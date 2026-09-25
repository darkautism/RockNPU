#!/usr/bin/env python3
"""Run isolated CPU/NPU ABBA measurements on RK3588 and retain userspace evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot():
    paths = [
        *Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor"),
        *Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_cur_freq"),
        *Path("/sys/class/thermal").glob("thermal_zone*/temp"),
    ]
    for node in sorted(Path("/sys/class/devfreq").glob("*")):
        for field in ("governor", "cur_freq", "target_freq", "min_freq", "max_freq", "available_frequencies"):
            paths.append(node / field)
    result = {"machine": platform.machine()}
    for path in paths:
        try:
            result[str(path)] = path.read_text().strip()
        except OSError as error:
            result[str(path)] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("decode", "prefill", "request"), default="decode")
    parser.add_argument("--prefill-cache", action="store_true")
    parser.add_argument("--npu-decode", choices=("on", "off"), default="on")
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--npu-threads", type=int, default=4)
    parser.add_argument("--allow-generic-cpu", action="store_true")
    parser.add_argument("--no-warmup", action="store_true",
                        help="disable llama-bench warmup so first-use resident-cache setup is timed")
    parser.add_argument("--expected-npu-freq", type=int, default=None,
                        help="require both NPU cur_freq and target_freq to equal this value")
    args = parser.parse_args()

    if platform.machine() not in ("aarch64", "arm64"):
        parser.error("run on an RK3588 host")
    for name in ("bench", "plugin", "model"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.sidecar is not None:
        args.sidecar = args.sidecar.resolve(strict=True)
    if args.npu_decode == "on" and args.sidecar is None:
        parser.error("NPU decode requires --sidecar; prefill-only acceleration does not")
    if min(args.tokens, args.reps, args.blocks, args.prompt) < 1:
        parser.error("token, prompt, repeat and block counts must be positive")
    if not all(1 <= n <= 4 for n in (args.cpu_threads, args.npu_threads)):
        parser.error("use 1..4 threads on the four A76 cores")

    cache = args.bench.parent.parent / "CMakeCache.txt"
    native = "GGML_NATIVE:BOOL=ON" in cache.read_text()
    if not native and not args.allow_generic_cpu:
        parser.error("CPU reference must use GGML_NATIVE=ON; generic builds are diagnostic only")

    initial = snapshot()
    for policy in (0, 4, 6):
        if initial.get(f"/sys/devices/system/cpu/cpufreq/policy{policy}/scaling_governor") != "performance":
            parser.error("set CPU policies 0/4/6 to performance before timing")

    def check_npu_frequency(label, snap):
        if args.expected_npu_freq is None:
            return
        node = "fdab0000.npu"
        cur = snap.get(f"/sys/class/devfreq/{node}/cur_freq")
        target = snap.get(f"/sys/class/devfreq/{node}/target_freq")
        if cur != str(args.expected_npu_freq) or target != str(args.expected_npu_freq):
            parser.error(f"{label} NPU frequency is not pinned to {args.expected_npu_freq}: cur={cur}, target={target}")

    check_npu_frequency("initial", initial)

    model_hash = sha256(args.model)
    if args.sidecar is not None:
        manifest = json.loads((args.sidecar / "manifest.json").read_text())
        if manifest.get("format") != "rocknpu-w8-sidecar-v2" or manifest.get("source_sha256") != model_hash:
            parser.error("sidecar v2 must match the exact GGUF SHA-256")

    args.output.mkdir(parents=True, exist_ok=False)
    base_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("ROCKNPU_") and k not in ("GGML_BACKEND_PATH", "GGML_SCHED_DEBUG")
    }
    workload = ["-p", "0", "-n", str(args.tokens)] if args.mode == "decode" else [
        "-p", "0", "-n", "0", "-pg", f"{args.prompt},{args.tokens}"
    ]
    if args.mode == "prefill":
        workload = ["-p", str(args.prompt), "-n", "0"]

    results = []
    def safe_env(environ):
        redacted = {}
        for key, value in sorted(environ.items()):
            upper = key.upper()
            redacted[key] = "<redacted>" if any(word in upper for word in ("TOKEN", "PASSWORD", "SECRET", "API_KEY")) else value
        return redacted

    recorded_env = safe_env(os.environ)
    (args.output / "environment.json").write_text(json.dumps(recorded_env, indent=2) + "\n")
    metadata = {
        "argv": sys.argv,
        "command": [str(args.bench)] + sys.argv[1:],
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": recorded_env,
        "source_hashes": {
            "model": model_hash,
            "bench": sha256(args.bench),
            "plugin": sha256(args.plugin),
            "cpu_backend": sha256(args.bench.parent / "libggml-cpu.so"),
        },
        "cpu_native": native,
        "initial_environment": initial,
        "frequency_gate": {
            "expected_npu_freq": args.expected_npu_freq,
            "initial_cur_freq": initial.get("/sys/class/devfreq/fdab0000.npu/cur_freq"),
            "initial_target_freq": initial.get("/sys/class/devfreq/fdab0000.npu/target_freq"),
        },
        "warmup_policy": (
            "disabled; first timed repetition includes first-use backend/cache setup"
            if args.no_warmup else
            "llama-bench default warmup runs before timing; context-owned resident caches are warm"
        ),
        "timing_scope": (
            "llama-bench timed workload; model loading excluded; warmup disabled"
            if args.no_warmup else
            "llama-bench timed workload; model loading and default warmup excluded; resident caches are warm"
        ),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "run-manifest.json").write_text(json.dumps({"argv": sys.argv, "command": [str(args.bench)] + sys.argv[1:], "environment": recorded_env, "initial_environment": initial, "frequency_gate": metadata["frequency_gate"]}, indent=2) + "\n")

    for index, kind in enumerate(["cpu", "npu", "npu", "cpu"] * args.blocks, 1):
        env = base_env.copy()
        if kind == "npu":
            env["GGML_BACKEND_PATH"] = str(args.plugin)
            env["ROCKNPU_DISPATCH_SUMMARY"] = "1"
            if args.npu_decode == "on":
                env.update(
                    ROCKNPU_W8_SIDECAR_DIR=str(args.sidecar),
                    ROCKNPU_W8_DIRECT_SUBMIT="1",
                    ROCKNPU_EXPERIMENT_DIRECT_SCRATCH="1",
                )
            env["ROCKNPU_DECODE"] = "1" if args.npu_decode == "on" else "0"
            if args.prefill_cache:
                env["ROCKNPU_PREFILL_CACHE"] = "1"

        threads = args.cpu_threads if kind == "cpu" else args.npu_threads
        command = [
            "taskset", "-c", "4-7", str(args.bench),
            "-m", str(args.model),
            *workload,
            "-r", str(args.reps),
            "-t", str(threads),
            "-fa", "on",
            "-dev", "none" if kind == "cpu" else "ROCKNPU0",
            "-nopo", "1" if kind == "cpu" else "0",
            "-o", "json",
        ]
        if args.no_warmup:
            command.append("--no-warmup")

        before = snapshot()
        prefix = args.output / f"{index:02d}-{kind}"
        print(f"START {index} {kind}", flush=True)
        start = time.monotonic()
        with prefix.with_suffix(".stdout").open("wb") as out, prefix.with_suffix(".stderr").open("wb") as err:
            process = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=1800)

        check_npu_frequency(f"before run {index} {kind}", before)
        record = {
            "kind": kind,
            "command": command,
            "environment": {k: v for k, v in sorted(env.items()) if k.startswith("ROCKNPU_") or k in ("GGML_BACKEND_PATH", "LD_LIBRARY_PATH", "GGML_SCHED_DEBUG")},
            "exit_code": process.returncode,
            "process_wall_seconds": time.monotonic() - start,
            "before": before,
            "after": snapshot(),
        }
        check_npu_frequency(f"after run {index} {kind}", record["after"])
        prefix.with_suffix(".meta.json").write_text(json.dumps(record, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"{kind} exited {process.returncode}; see {prefix}.stderr")

        data = json.loads(prefix.with_suffix(".stdout").read_text())
        if len(data) != 1:
            raise RuntimeError("expected exactly one workload result per process")
        row = data[0]
        if kind == "cpu" and (row["backends"] != "CPU" or row["devices"] != "none" or row["no_op_offload"] != 1):
            raise RuntimeError("CPU result is contaminated by an accelerator backend")
        if kind == "npu" and ("ROCKNPU" not in row["backends"] or row["devices"] != "ROCKNPU0"):
            raise RuntimeError("NPU backend did not load")

        if kind == "npu":
            summaries = re.findall(
                r"ROCKNPU GGML TRACE summary q4_K_mul_mat=(\d+) q6_K_mul_mat=(\d+) f16_mul_mat=(\d+)",
                prefix.with_suffix(".stderr").read_text(errors="replace"),
            )
            dispatches = sum(sum(map(int, counts)) for counts in summaries)
            record["npu_mul_mat_dispatches"] = dispatches
            prefix.with_suffix(".meta.json").write_text(json.dumps(record, indent=2) + "\n")
            expects_npu_work = args.npu_decode == "on" or args.mode != "decode"
            if expects_npu_work and dispatches == 0:
                raise RuntimeError(f"NPU backend loaded but executed no matmuls; see {prefix}.stderr")

        expected_prompt = 0 if args.mode == "decode" else args.prompt
        expected_tokens = 0 if args.mode == "prefill" else args.tokens
        if (row["n_prompt"], row["n_gen"]) != (expected_prompt, expected_tokens):
            raise RuntimeError("reported workload differs from requested workload")

        results.append({"kind": kind, "result": row})
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"DONE {index} {kind}: {row['avg_ts']:.6f} +/- {row['stddev_ts']:.6f} tok/s", flush=True)

    means = {
        kind: sum(r["result"]["avg_ns"] for r in results if r["kind"] == kind)
        / sum(r["kind"] == kind for r in results)
        for kind in ("cpu", "npu")
    }
    summary = {"mean_ns": means, "speedup_from_mean_time": means["cpu"] / means["npu"]}
    summary["block_speedups"] = []
    for offset in range(0, len(results), 4):
        block = results[offset:offset + 4]
        cpu_ns = sum(r["result"]["avg_ns"] for r in block if r["kind"] == "cpu")
        npu_ns = sum(r["result"]["avg_ns"] for r in block if r["kind"] == "npu")
        summary["block_speedups"].append(cpu_ns / npu_ns)

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
