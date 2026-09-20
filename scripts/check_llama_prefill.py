#!/usr/bin/env python3
"""Compare deterministic CPU, uncached NPU and cached NPU prompt processing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("completion", "plugin", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paragraph = (
        "Alice works at a library in a small town. Each morning she opens the doors at nine, "
        "returns books to their shelves, and helps visitors find stories to read. Bob repairs "
        "bicycles at the shop next door. On Friday they meet at the park and talk about their "
        "week. Alice brings a book about birds and Bob brings a map of nearby trails. "
    )
    prompts = ["The capital of France is", paragraph * 12 + "\nSummarize Alice's job in one sentence:\n"]
    results = []
    for index, prompt in enumerate(prompts):
        prompt_file = args.output / f"prompt-{index}.txt"
        prompt_file.write_text(prompt)
        outputs = {}
        for variant in ("cpu", "bridge", "cached"):
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("ROCKNPU_") and k not in ("GGML_BACKEND_PATH", "GGML_SCHED_DEBUG")}
            if variant != "cpu":
                env.update(GGML_BACKEND_PATH=str(args.plugin), ROCKNPU_DECODE="0",
                           ROCKNPU_PREFILL_CACHE="1" if variant == "cached" else "0",
                           ROCKNPU_GGML_TRACE="1")
            command = ["taskset", "-c", "4-7", str(args.completion), "-m", str(args.model),
                       "-f", str(prompt_file), "-n", "64", "-t", "4", "-fa", "on",
                       "-fit", "off", "-no-cnv", "--temp", "0", "--seed", "1",
                       "--no-warmup", "--no-display-prompt", "-c", "2048", "-ub", "512",
                       "-dev", "none" if variant == "cpu" else "ROCKNPU0"]
            prefix = args.output / f"{index}-{variant}"
            print(f"START {index} {variant}", flush=True)
            with prefix.with_suffix(".stdout").open("wb") as out, prefix.with_suffix(".stderr").open("wb") as err:
                process = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=900)
            raw = prefix.with_suffix(".stdout").read_bytes()
            trace = prefix.with_suffix(".stderr").read_text(errors="replace")
            record = {"prompt": index, "variant": variant, "command": command,
                      "exit_code": process.returncode, "stdout_sha256": hashlib.sha256(raw).hexdigest(),
                      "stdout_bytes": len(raw), "npu_prefill_calls": trace.count("path=fp16_bridge"),
                      "npu_decode_calls": trace.count("path=w8a8_m1") + trace.count("path=w4a4_m1")}
            results.append(record)
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            if process.returncode or not raw:
                raise RuntimeError(f"{prefix} failed or generated empty output")
            if variant != "cpu" and ((index == 1 and record["npu_prefill_calls"] == 0) or record["npu_decode_calls"] != 0):
                raise RuntimeError(f"{prefix} did not use the intended prefill-only route")
            outputs[variant] = raw
            print(json.dumps(record), flush=True)
        comparison = {"prompt": index, "cached_equals_bridge": outputs["cached"] == outputs["bridge"],
                      "cached_equals_cpu": outputs["cached"] == outputs["cpu"]}
        (args.output / f"comparison-{index}.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison), flush=True)
    print("DONE: inspect both comparisons; token equality is a regression check, not a quality benchmark", flush=True)


if __name__ == "__main__":
    main()
