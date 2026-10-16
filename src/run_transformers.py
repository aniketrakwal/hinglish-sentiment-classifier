"""Run all 9 transformer fine-tuning runs (3 models x seeds 42, 43, 44), VALIDATION only.

    python src/run_transformers.py

Each run is a separate process (src/train_transformer.py), so the GPU memory is fully freed between
runs and one crash cannot stop the others. A run is skipped if its results already exist AND were made
with the same max_length, so the script can simply be started again after a crash or an interruption.
"""

import json
import os
import subprocess
import sys
import time

MODELS = ["google/muril-base-cased", "l3cube-pune/hing-roberta", "xlm-roberta-base"]
SEEDS = [42, 43, 44]
MAX_LENGTH = 64
RESULTS_DIR = "results"
FAILURE_LOG = os.path.join(RESULTS_DIR, "transformer_failures.log")


def run_name(model, seed):
    return f"{model.split('/')[-1]}_seed{seed}"


def summary_path(model, seed):
    return os.path.join(RESULTS_DIR, f"{run_name(model, seed)}_val_summary.json")


def already_done(model, seed):
    """True if this run has a results file made with the current max_length (an old 48-token file does not count)."""
    path = summary_path(model, seed)
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    return summary.get("max_length") == MAX_LENGTH and summary.get("model") == model


def main():
    # Seed-major order: all 3 models with seed 42 first, then seed 43, then seed 44.
    # If something goes wrong half-way we still have a complete comparison for the first seeds.
    runs = [(model, seed) for seed in SEEDS for model in MODELS]
    started = time.time()
    failed = []

    for number, (model, seed) in enumerate(runs, start=1):
        name = run_name(model, seed)
        elapsed = (time.time() - started) / 60
        if already_done(model, seed):
            print(f"[run {number} of {len(runs)} | {elapsed:.1f} min elapsed] {name}: results exist, skipping", flush=True)
            continue

        print(f"\n[run {number} of {len(runs)} | {elapsed:.1f} min elapsed] starting {name}", flush=True)
        log_path = os.path.join(RESULTS_DIR, f"{name}_train_log.txt")
        command = [sys.executable, "-u", os.path.join("src", "train_transformer.py"),
                   "--model", model, "--seed", str(seed), "--max_length", str(MAX_LENGTH)]
        run_start = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        minutes = (time.time() - run_start) / 60

        if result.returncode != 0 or not already_done(model, seed):
            # Do not stop: write down what happened and go on with the next run
            message = f"FAILED {name} (exit code {result.returncode}) after {minutes:.1f} min, see {log_path}"
            print(message, flush=True)
            with open(FAILURE_LOG, "a", encoding="utf-8") as f:
                f.write(message + "\n")
            failed.append(name)
            continue

        with open(summary_path(model, seed), encoding="utf-8") as f:
            summary = json.load(f)
        print(f"  done in {minutes:.1f} min | best val macro-F1 {summary['best_val']['macro_f1']:.4f} | "
              f"peak GPU memory {summary['peak_gpu_allocated_gib']:.2f} GiB allocated, "
              f"{summary['peak_gpu_reserved_gib']:.2f} GiB reserved", flush=True)

    total = (time.time() - started) / 60
    print(f"\nFinished all {len(runs)} runs in {total:.1f} min. Failed: {failed if failed else 'none'}")


if __name__ == "__main__":
    main()
