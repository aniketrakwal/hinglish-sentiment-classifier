"""Run the LoRA experiments for HingRoBERTa, VALIDATION only.

    python src/run_lora.py

  a) learning-rate check: r=8, seed 42, lr in 2e-4 and 5e-4 -> the lr with the better val macro-F1 wins
  b) rank ablation at that lr: r=8 and r=16, seeds 42, 43, 44 (the run from (a) is reused)

Each run is a separate process (src/train_transformer.py --lora), like in src/run_transformers.py.
A run is skipped if its results already exist, so the script can be started again after an interruption.
"""

import json
import os
import subprocess
import sys
import time

MODEL = "l3cube-pune/hing-roberta"
LEARNING_RATES = [2e-4, 5e-4]
RANKS = [8, 16]
SEEDS = [42, 43, 44]
MAX_LENGTH = 64
EPOCHS = 6
RESULTS_DIR = "results"
FAILURE_LOG = os.path.join(RESULTS_DIR, "lora_failures.log")


def run_name(r, lr, seed):
    """Same name that src/train_transformer.py gives the run (it is used in the result file names)."""
    return f"{MODEL.split('/')[-1]}_lora_r{r}_lr{lr:g}_seed{seed}"


def summary_path(r, lr, seed):
    return os.path.join(RESULTS_DIR, f"{run_name(r, lr, seed)}_val_summary.json")


def already_done(r, lr, seed):
    """True if this run has a results file made with the same settings."""
    path = summary_path(r, lr, seed)
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    return (summary.get("lora") is True and summary.get("r") == r and summary.get("max_length") == MAX_LENGTH
            and summary.get("num_epochs") == EPOCHS and abs(summary.get("learning_rate", 0) - lr) < 1e-12)


def best_val_f1(r, lr, seed):
    with open(summary_path(r, lr, seed), encoding="utf-8") as f:
        return json.load(f)["best_val"]["macro_f1"]


def run_one(r, lr, seed, started):
    """Run one training (or skip it). Returns True if its results exist afterwards."""
    name = run_name(r, lr, seed)
    elapsed = (time.time() - started) / 60
    if already_done(r, lr, seed):
        print(f"[{elapsed:.1f} min elapsed] {name}: results exist, skipping", flush=True)
        return True

    print(f"\n[{elapsed:.1f} min elapsed] starting {name}", flush=True)
    log_path = os.path.join(RESULTS_DIR, f"{name}_train_log.txt")
    command = [sys.executable, "-u", os.path.join("src", "train_transformer.py"), "--model", MODEL,
               "--seed", str(seed), "--max_length", str(MAX_LENGTH), "--lora", "--r", str(r),
               "--lr", str(lr), "--epochs", str(EPOCHS)]
    run_start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    minutes = (time.time() - run_start) / 60

    if result.returncode != 0 or not already_done(r, lr, seed):
        # Do not stop: write down what happened and go on
        message = f"FAILED {name} (exit code {result.returncode}) after {minutes:.1f} min, see {log_path}"
        print(message, flush=True)
        with open(FAILURE_LOG, "a", encoding="utf-8") as f:
            f.write(message + "\n")
        return False

    with open(summary_path(r, lr, seed), encoding="utf-8") as f:
        summary = json.load(f)
    seconds = [row["epoch_time_s"] for row in summary["epochs"]]
    print(f"  done in {minutes:.1f} min | best val macro-F1 {summary['best_val']['macro_f1']:.4f} | "
          f"{sum(seconds) / len(seconds):.0f} s per epoch | peak GPU memory "
          f"{summary['peak_gpu_allocated_gib']:.2f} GiB allocated, {summary['peak_gpu_reserved_gib']:.2f} GiB reserved | "
          f"saved weights {summary['saved_model_mb']} MB", flush=True)
    return True


def main():
    started = time.time()

    # ---- (a) learning-rate check: r=8, seed 42
    for lr in LEARNING_RATES:
        run_one(8, lr, 42, started)
    finished = [lr for lr in LEARNING_RATES if already_done(8, lr, 42)]
    if not finished:
        sys.exit("no learning-rate run finished, see " + FAILURE_LOG)
    best_lr = max(finished, key=lambda lr: best_val_f1(8, lr, 42))      # judged on VALIDATION macro-F1
    for lr in finished:
        print(f"learning-rate check: lr {lr:g} -> val macro-F1 {best_val_f1(8, lr, 42):.4f}")
    print(f"chosen learning rate: {best_lr:g}", flush=True)
    with open(os.path.join(RESULTS_DIR, "lora_lr_choice.txt"), "w", encoding="utf-8") as f:
        f.write(f"LEARNING_RATE: {best_lr:g}\n")
        for lr in finished:
            f.write(f"lr {lr:g} (r=8, seed 42): val macro-F1 {best_val_f1(8, lr, 42):.4f}\n")

    # ---- (b) rank ablation at the chosen learning rate, seed-major order
    failed = []
    for seed in SEEDS:
        for r in RANKS:
            if not run_one(r, best_lr, seed, started):
                failed.append(run_name(r, best_lr, seed))

    print(f"\nFinished in {(time.time() - started) / 60:.1f} min. Failed: {failed if failed else 'none'}")


if __name__ == "__main__":
    main()
