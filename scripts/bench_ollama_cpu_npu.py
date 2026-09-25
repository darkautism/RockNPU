#!/usr/bin/env python3
"""Reproducible stock-Ollama CPU/NPU ABBA benchmark and quality oracle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    result = {
        "timestamp_unix": time.time(),
        "machine": platform.machine(),
        "uname": " ".join(platform.uname()),
    }
    paths = []
    for policy in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        for field in ("scaling_governor", "scaling_cur_freq", "cpuinfo_cur_freq",
                      "cpuinfo_min_freq", "cpuinfo_max_freq", "related_cpus", "online"):
            paths.append(policy / field)
    for node in sorted(Path("/sys/class/devfreq").glob("*")):
        for field in ("governor", "cur_freq", "target_freq", "min_freq", "max_freq",
                      "available_frequencies"):
            paths.append(node / field)
    paths.extend(sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")))
    for path in paths:
        try:
            result[str(path)] = path.read_text().strip()
        except OSError as e:
            result[str(path)] = f"<unavailable: {e}>"
    return result


def safe_env(environ):
    out = {}
    for key, value in sorted(environ.items()):
        upper = key.upper()
        out[key] = "<redacted>" if any(x in upper for x in ("TOKEN", "PASSWORD", "SECRET", "API_KEY")) else value
    return out


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def raw_sha(path):
    return sha256(path)


def wait_http(url, deadline, process=None):
    last = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"service exited early: {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                body = r.read()
                return r.status, body
        except Exception as e:  # readiness is intentionally retried
            last = e
            time.sleep(0.25)
    raise TimeoutError(f"service did not become ready: {url}: {last}")


def api_get(port, path, raw_path):
    url = f"http://127.0.0.1:{port}{path}"
    start = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
    elapsed = time.monotonic() - start
    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    return {"status": status, "elapsed_s": elapsed, "raw": str(raw_path),
            "sha256": hashlib.sha256(raw).hexdigest(), "json": parsed}


def generate(port, payload, stem, keep_warm=True):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    request = dict(payload)
    if keep_warm:
        request["keep_alive"] = "30m"
    write_json(stem.with_suffix(".request.json"), request)
    body = json.dumps(request).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    before = snapshot()
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
    wall = time.monotonic() - start
    raw_path = stem.with_suffix(".response.raw")
    raw_path.write_bytes(raw)
    try:
        parsed = json.loads(raw)
        write_json(stem.with_suffix(".response.json"), parsed)
    except Exception:
        parsed = None
    after = snapshot()
    record = {
        "request_file": str(stem.with_suffix(".request.json")),
        "response_raw_file": str(raw_path),
        "response_json_file": str(stem.with_suffix(".response.json")) if parsed is not None else None,
        "status": status, "wall_s": wall, "before": before, "after": after,
        "response_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "response": parsed,
    }
    if isinstance(parsed, dict):
        record["response_text"] = parsed.get("response")
        record["eval_count"] = parsed.get("eval_count")
        record["eval_duration_ns"] = parsed.get("eval_duration")
        if parsed.get("eval_count") is not None and parsed.get("eval_duration"):
            record["tok_s"] = parsed["eval_count"] / (parsed["eval_duration"] / 1e9)
    write_json(stem.with_suffix(".record.json"), record)
    return record


def start_service(name, binary, env, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = log_path.with_suffix(".stdout.log").open("wb")
    err = log_path.with_suffix(".stderr.log").open("wb")
    process = subprocess.Popen([str(binary), "serve"], env=env, stdout=out, stderr=err,
                               start_new_session=True)
    out.close()
    err.close()
    return {"name": name, "process": process, "pid": process.pid,
            "command": [str(binary), "serve"], "env": safe_env(env),
            "stdout": str(log_path.with_suffix(".stdout.log")),
            "stderr": str(log_path.with_suffix(".stderr.log"))}


def stop_service(service):
    process = service.get("process")
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def service_env(args, port, kind):
    env = os.environ.copy()
    env.update({
        "OLLAMA_HOST": f"127.0.0.1:{port}",
        "OLLAMA_MODELS": str(args.models),
        "OLLAMA_NUM_GPU": "0" if kind == "cpu" else "999",
        "OLLAMA_KEEP_ALIVE": "30m",
        "LLAMA_ARG_DEVICE": "none" if kind == "cpu" else "ROCKNPU0",
    })
    lib = str(args.lib_dir)
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = lib + (":" + old_ld if old_ld else "")
    if kind == "npu":
        env.update({
            "GGML_BACKEND_PATH": str(args.plugin),
            "ROCKNPU_W8_SIDECAR_DIR": str(args.sidecar),
            "ROCKNPU_W8_DIRECT_SUBMIT": "1",
            "ROCKNPU_EXPERIMENT_DIRECT_SCRATCH": "1",
            "ROCKNPU_DECODE": "1",
            "ROCKNPU_PREFILL_CACHE": "1",
            "ROCKNPU_DISPATCH_SUMMARY": "1",
        })
    else:
        env.pop("GGML_BACKEND_PATH", None)
        for key in list(env):
            if key.startswith("ROCKNPU_"):
                env.pop(key)
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ollama", type=Path, required=True)
    ap.add_argument("--lib-dir", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-blob", type=Path, required=True)
    ap.add_argument("--plugin", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--prompt", default="Paris")
    ap.add_argument("--num-predict", type=int, default=8)
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--num-thread", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--npu-port", type=int, default=11440)
    ap.add_argument("--cpu-port", type=int, default=11441)
    ap.add_argument("--oracle-port", type=int, default=11442)
    args = ap.parse_args()
    for name in ("ollama", "lib_dir", "models", "model_blob", "plugin", "sidecar"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.blocks < 3 or args.warmups < 1:
        ap.error("use at least three blocks and one warmup")
    args.output.mkdir(parents=True, exist_ok=False)
    initial = snapshot()
    payload = {"model": args.model, "prompt": args.prompt, "stream": False,
               "options": {"temperature": 0, "num_predict": args.num_predict,
                           "num_ctx": args.num_ctx, "num_thread": args.num_thread}}
    hashes = {"ollama": sha256(args.ollama), "plugin": sha256(args.plugin),
              "model": sha256(args.model_blob)}
    manifest = {
        "board": args.board, "argv": sys.argv,
        "executable_command": [str(args.ollama), "serve"],
        "workload": payload, "blocks": args.blocks, "warmups": args.warmups,
        "ports": {"npu": args.npu_port, "cpu": args.cpu_port, "oracle": args.oracle_port},
        "hashes": hashes, "initial_snapshot": initial,
        "environment": safe_env(os.environ),
    }
    write_json(args.output / "run-manifest.json", manifest)
    npu_env = service_env(args, args.npu_port, "npu")
    cpu_env = service_env(args, args.cpu_port, "cpu")
    oracle_env = service_env(args, args.oracle_port, "cpu")
    manifest["service_environments"] = {"npu": safe_env(npu_env), "cpu": safe_env(cpu_env), "oracle": safe_env(oracle_env)}
    manifest["service_commands"] = {
        "npu": [str(args.ollama), "serve"], "cpu": [str(args.ollama), "serve"],
        "oracle": [str(args.ollama), "serve"],
    }
    write_json(args.output / "run-manifest.json", manifest)
    command_lines = ["#!/bin/sh", "set -eu"]
    for name, env, port in (("npu", npu_env, args.npu_port), ("cpu", cpu_env, args.cpu_port), ("oracle", oracle_env, args.oracle_port)):
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()) if k in {
            "OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_NUM_GPU", "OLLAMA_KEEP_ALIVE", "LLAMA_ARG_DEVICE",
            "GGML_BACKEND_PATH", "LD_LIBRARY_PATH", "ROCKNPU_W8_SIDECAR_DIR", "ROCKNPU_W8_DIRECT_SUBMIT",
            "ROCKNPU_EXPERIMENT_DIRECT_SCRATCH", "ROCKNPU_DECODE", "ROCKNPU_PREFILL_CACHE", "ROCKNPU_DISPATCH_SUMMARY",
        })
        command_lines += [f"# {name} service", f"env {exports} {shlex.quote(str(args.ollama))} serve >{args.output.name}-{name}.stdout.log 2>{args.output.name}-{name}.stderr.log &", f"{name}_pid=$!"]
    command_lines += ["# Run requests using the Python harness; this file records the exact service launch commands.", "wait"]
    (args.output / "commands.sh").write_text("\n".join(command_lines) + "\n")
    (args.output / "commands.sh").chmod(0o755)
    services = []
    oracle = None
    try:
        npu = start_service("npu", args.ollama, npu_env, args.output / "npu-server")
        cpu = start_service("cpu", args.ollama, cpu_env, args.output / "cpu-server")
        services.extend([npu, cpu])
        for service, port in ((npu, args.npu_port), (cpu, args.cpu_port)):
            wait_http(f"http://127.0.0.1:{port}/api/tags", time.monotonic() + 180, service["process"])
            api_get(port, "/api/ps", args.output / f"{service['name']}-ps-before.json")
        for kind, port in (("npu", args.npu_port), ("cpu", args.cpu_port)):
            for i in range(args.warmups):
                generate(port, payload, args.output / "warmup" / f"{kind}-{i+1:02d}")
        oracle = start_service("oracle", args.ollama, oracle_env, args.output / "oracle-server")
        services.append(oracle)
        wait_http(f"http://127.0.0.1:{args.oracle_port}/api/tags", time.monotonic() + 180, oracle["process"])
        oracle_record = generate(args.oracle_port, payload, args.output / "oracle" / "request")
        blocks = []
        for block in range(args.blocks):
            order = ("npu", "cpu", "npu", "cpu") if block % 2 == 0 else ("cpu", "npu", "cpu", "npu")
            records = []
            for index, kind in enumerate(order, 1):
                port = args.npu_port if kind == "npu" else args.cpu_port
                result = generate(port, payload, args.output / "blocks" / f"block-{block+1:02d}" / f"{kind}-{index:02d}")
                records.append({"kind": kind, "order": index, "record": result})
            blocks.append({"block": block + 1, "order": list(order), "requests": records})
            for service, port in ((npu, args.npu_port), (cpu, args.cpu_port)):
                api_get(port, "/api/ps", args.output / f"{service['name']}-ps-block-{block+1:02d}.json")
        for service, port in ((npu, args.npu_port), (cpu, args.cpu_port)):
            api_get(port, "/api/ps", args.output / f"{service['name']}-ps-after.json")
        npu_log = Path(npu["stderr"]).read_text(errors="replace")
        cpu_log = Path(cpu["stderr"]).read_text(errors="replace")
        npu_responses = [x["record"]["response_text"] for b in blocks for x in b["requests"] if x["kind"] == "npu"]
        cpu_responses = [x["record"]["response_text"] for b in blocks for x in b["requests"] if x["kind"] == "cpu"]
        oracle_text = oracle_record.get("response_text")
        quality = {
            "status": "passed" if npu_responses and all(x == oracle_text for x in npu_responses) else "failed",
            "cpu_oracle_response": oracle_text,
            "npu_responses": npu_responses,
            "cpu_responses": cpu_responses,
            "npu_matches_oracle": all(x == oracle_text for x in npu_responses),
            "cpu_matches_oracle": all(x == oracle_text for x in cpu_responses),
            "oracle_request": str(args.output / "oracle" / "request.record.json"),
        }
        result = {
            "board": args.board, "manifest": manifest, "blocks": blocks,
            "quality_gate": quality,
            "npu_log": {"stderr": npu["stderr"], "sha256": hashlib.sha256(npu_log.encode()).hexdigest(),
                        "w8a8_m1_lines": npu_log.count("path=w8a8_m1"),
                        "host_lines": npu_log.count("ROCKNPU_HOST"),
                        "dispatch_summary_lines": npu_log.count("ROCKNPU GGML TRACE summary")},
            "cpu_log": {"stderr": cpu["stderr"], "sha256": hashlib.sha256(cpu_log.encode()).hexdigest()},
            "oracle": oracle_record,
        }
        write_json(args.output / "result.json", result)
    finally:
        for service in reversed(services):
            stop_service(service)
    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(args.output))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    write_json(args.output / "artifact-manifest.json", {"board": args.board, "files": files})
    print(json.dumps({"output": str(args.output), "quality": "passed" if result["quality_gate"]["status"] == "passed" else "failed", "npu_log": result["npu_log"], "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
