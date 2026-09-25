#!/usr/bin/env python3
"""Build a machine-readable index for versioned frontend benchmark evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frequency_values(snapshots):
    values = set()
    for snap in snapshots:
        for key in ("/sys/class/devfreq/fdab0000.npu/cur_freq",
                    "/sys/class/devfreq/fdab0000.npu/target_freq"):
            if key in snap:
                values.add(str(snap[key]))
    return sorted(values)


def profile_lines(path):
    return [line.strip() for line in Path(path).read_text(errors="replace").splitlines()
            if "ROCKNPU M1 PROFILE" in line]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.raw_root.resolve()
    board = args.board
    dirs = {
        "llama_formal": root / f"llama-formal-repro-{board}",
        "llama_quality": root / f"llama-quality-repro-{board}",
        "ollama_formal": root / f"ollama-formal-repro-{board}",
        "ollama_trace": root / f"ollama-trace-repro-{board}",
        "llama_profile": root / f"llama-profile-repro-{board}",
        "candidate_matrix": root / f"candidate-matrix-{board}",
        "ollama_concurrency": root / f"ollama-concurrency-{board}",
    }
    for name, path in dirs.items():
        if not path.is_dir():
            raise SystemExit(f"missing evidence directory: {name}: {path}")
    formal = read_json(dirs["llama_formal"] / "summary.json")
    formal_meta = read_json(dirs["llama_formal"] / "metadata.json")
    formal_metas = sorted(dirs["llama_formal"].glob("*.meta.json"))
    llama_dispatch = sum(int(read_json(p).get("npu_mul_mat_dispatches", 0)) for p in formal_metas)
    quality = read_json(dirs["llama_quality"] / "result.json")
    ollama_formal = read_json(dirs["ollama_formal"] / "result.json")
    ollama_trace = read_json(dirs["ollama_trace"] / "result.json")
    profile = read_json(dirs["llama_profile"] / "summary.json")
    profile_records = profile_lines(dirs["llama_profile"] / "02-npu.stderr")
    candidates = read_json(dirs["candidate_matrix"] / "candidate-matrix.json")
    concurrency = read_json(dirs["ollama_concurrency"] / "run-manifest.json")
    all_snapshots = []
    for p in formal_metas:
        m = read_json(p)
        all_snapshots += [m.get("before", {}), m.get("after", {})]
    for result in (ollama_formal, ollama_trace):
        all_snapshots.append(result.get("manifest", {}).get("initial_snapshot", {}))
        for block in result.get("blocks", []):
            for req in block.get("requests", []):
                r = req.get("record", {})
                all_snapshots += [r.get("before", {}), r.get("after", {})]
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != args.output:
            files[str(path.relative_to(root))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        source_commit = None
    index = {
        "schema": "rocknpu-frontend-evidence-v2",
        "board": board,
        "source_commit_at_index": source_commit,
        "fixed_conditions": {
            "model_sha256": "5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583",
            "sidecar_manifest": "w8a8-models/native-w8-gguf/manifest.json",
            "npu_devfreq": "/sys/class/devfreq/fdab0000.npu",
            "required_npu_cur_and_target": "700000000",
            "cpu_policies": {"0": "performance@1800000", "4": "performance@2400000", "6": "performance@2400000"},
        },
        "llama_cpp": {
            "formal_ab": {
                "directory": str(dirs["llama_formal"].relative_to(root)),
                "command_manifest": "run-manifest.json",
                "environment": "environment.json",
                "summary": formal,
                "frequency_gate": formal_meta.get("frequency_gate"),
                "observed_npu_frequency_values": frequency_values(all_snapshots),
                "npu_dispatch_count": llama_dispatch,
                "raw_runs": "01..12 CPU/NPU stdout/stderr/meta",
            },
            "quality_oracle": {
                "directory": str(dirs["llama_quality"].relative_to(root)),
                "result": "result.json",
                "status": quality.get("quality_gate", {}).get("status"),
                "oracle": quality.get("quality_gate", {}).get("cpu_oracle_response"),
                "trace": quality.get("npu_trace"),
            },
        },
        "ollama": {
            "fair_performance_ab": {
                "directory": str(dirs["ollama_formal"].relative_to(root)),
                "result": "result.json",
                "quality": ollama_formal.get("quality_gate"),
                "npu_trace": ollama_formal.get("npu_log"),
                "trace_enabled": ollama_formal.get("npu_log", {}).get("trace_enabled", False),
            },
            "instrumented_dispatch_ab": {
                "directory": str(dirs["ollama_trace"].relative_to(root)),
                "result": "result.json",
                "quality": ollama_trace.get("quality_gate"),
                "npu_trace": ollama_trace.get("npu_log"),
                "trace_enabled": ollama_trace.get("npu_log", {}).get("trace_enabled", False),
            },
        },
        "profile": {
            "directory": str(dirs["llama_profile"].relative_to(root)),
            "summary": profile,
            "records": profile_records,
            "frequency_gate": read_json(dirs["llama_profile"] / "metadata.json").get("frequency_gate"),
        },
        "candidate_matrix": {
            "directory": str(dirs["candidate_matrix"].relative_to(root)),
            "matrix": candidates,
        },
        "ollama_concurrency": {
            "directory": str(dirs["ollama_concurrency"].relative_to(root)),
            "runs": concurrency.get("runs", []),
        },
        "raw_files": files,
    }
    if "700000000" not in index["llama_cpp"]["formal_ab"]["observed_npu_frequency_values"]:
        raise SystemExit("llama formal evidence does not contain NPU 700 MHz frequency")
    if not index["ollama"]["instrumented_dispatch_ab"]["npu_trace"].get("w8a8_m1_lines", 0):
        raise SystemExit("instrumented Ollama evidence has no native W8 dispatch")
    if not index["profile"]["records"]:
        raise SystemExit("profile evidence has no M1 records")
    if len(index["candidate_matrix"]["matrix"].get("rows", [])) < 8:
        raise SystemExit("candidate matrix is incomplete")
    if {x.get("concurrency") for x in index["ollama_concurrency"]["runs"]} != {8, 16}:
        raise SystemExit("concurrency evidence lacks c8/c16")
    args.output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"board": board, "files": len(files), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
