"""Fine-tune a pretrained transformer for 3-class Hinglish sentiment. Works for any of:
    google/muril-base-cased    l3cube-pune/hing-roberta    xlm-roberta-base

Run from the project root, e.g.:
    python src/train_transformer.py --model google/muril-base-cased --seed 42

VALIDATION ONLY: this script never reads data/processed/test.csv.
Memory settings are made for a 4 GB GPU (frozen word embeddings, fp16, gradient checkpointing,
batch 16 x 2 accumulation) and are the same for every model.
"""

import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # silence a harmless Windows warning

import argparse
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)

from preprocess import clean_text

# The label order is the same as for the BiLSTM: negative=0, neutral=1, positive=2
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
# results/tokenizer_report.txt: 48 is the smallest length covering >= 99% of tweets for all 3 tokenizers; we use 64
# (99.8% coverage). With dynamic padding a longer limit costs almost nothing, because batches are
# only padded to their own longest tweet.
DEFAULT_MAX_LENGTH = 64


class TweetDataset(Dataset):
    """Cleaned tweets, tokenized with the model's OWN tokenizer. No padding here: the data collator
    pads each batch only up to its longest tweet (dynamic padding), which saves time and memory."""

    def __init__(self, texts, labels, tokenizer, max_length=DEFAULT_MAX_LENGTH):
        # truncation=True cuts any tweet longer than max_length tokens (specials included)
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = [LABEL2ID[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.encodings.items()}   # input_ids, attention_mask, ...
        item["labels"] = self.labels[index]
        return item


def load_split(path):
    """Read a CSV and add the cleaned text (the shared clean_text() from src/preprocess.py)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["cleaned_text"] = df["text"].apply(clean_text)
    return df


def compute_metrics(eval_prediction):
    """Called by the Trainer after every epoch on the validation set."""
    logits, labels = eval_prediction
    preds = np.argmax(logits, axis=1)
    per_class = f1_score(labels, preds, labels=[0, 1, 2], average=None)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, labels=[0, 1, 2], average="macro"),
        "f1_negative": per_class[0], "f1_neutral": per_class[1], "f1_positive": per_class[2],
    }


def freeze_word_embeddings(model):
    """Freeze the word-embedding table (one vector per vocabulary word). It is a huge part of these
    models (e.g. 150M of 238M weights for MuRIL), so not training it saves a lot of GPU memory.
    The 12 transformer layers, the position embeddings and the classifier stay trainable.
    Returns (total parameters, trainable parameters, embedding-table parameters)."""
    embedding = model.get_input_embeddings()          # the word-embedding layer
    for parameter in embedding.parameters():
        parameter.requires_grad = False               # no gradient, no optimizer state for these
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = sum(p.numel() for p in embedding.parameters())
    return total, trainable, embedding_params


class EpochLogger(TrainerCallback):
    """Prints one line per epoch: train loss, val loss, val macro-F1, time."""

    def __init__(self):
        self.epoch_start = None
        self.losses = []          # training losses logged during the current epoch
        self.rows = []            # one dict per epoch, saved to a JSON file at the end

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:                       # a training-loss log entry
            # This transformers version logs the training loss multiplied by gradient_accumulation_steps
            # (at the start, with 3 classes, it logs 2.197 = 2 x ln 3 instead of 1.099). Divide it back
            # so the number is the real average loss per tweet and comparable with the validation loss.
            self.losses.append(logs["loss"] / args.gradient_accumulation_steps)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # The Trainer evaluates right after each training epoch, so this is the end of the epoch
        row = {
            "epoch": int(round(state.epoch)),
            "train_loss": float(np.mean(self.losses)) if self.losses else float("nan"),
            "val_loss": metrics["eval_loss"],
            "val_macro_f1": metrics["eval_macro_f1"],
            "epoch_time_s": round(time.time() - self.epoch_start, 1),   # training + evaluation
        }
        self.rows.append(row)
        gpu_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"EPOCH {row['epoch']}: train loss {row['train_loss']:.4f}  val loss {row['val_loss']:.4f}  "
              f"val macro-F1 {row['val_macro_f1']:.4f}  time {row['epoch_time_s']}s  "
              f"peak GPU memory so far {gpu_gb:.2f} GiB", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Fine-tune a transformer (validation only).")
    parser.add_argument("--model", required=True, help="e.g. google/muril-base-cased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--models_dir", default="models")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    run_name = f"{model_name}_seed{args.seed}"
    save_dir = os.path.join(args.models_dir, run_name)                  # git-ignored
    os.makedirs(args.results_dir, exist_ok=True)

    print("python:", sys.executable)
    print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found: stopping (training a transformer on CPU would take hours).")
    print("GPU:", torch.cuda.get_device_name(0),
          f"| total memory {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")

    set_seed(args.seed)                                                  # python, numpy and torch seeds

    # ---- Data: train and validation only (the test file is never opened here)
    train_df = load_split(os.path.join(args.data_dir, "train.csv"))
    val_df = load_split(os.path.join(args.data_dir, "val.csv"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = TweetDataset(train_df["cleaned_text"], train_df["label"], tokenizer, args.max_length)
    val_ds = TweetDataset(val_df["cleaned_text"], val_df["label"], tokenizer, args.max_length)
    print(f"train {len(train_ds)} tweets, val {len(val_ds)} tweets, max_length {args.max_length}")

    # ---- Model: pretrained body + a fresh 3-class classification layer on top
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)
    total, trainable, embedding_params = freeze_word_embeddings(model)
    print(f"parameters: total {total:,} | trainable {trainable:,} | "
          f"word-embedding table {embedding_params:,} ({100 * embedding_params / total:.1f}% of all parameters)")

    # ---- Training settings
    # Loss: plain cross-entropy (the Trainer default). The classes are only mildly imbalanced
    # (30% / 37% / 33%), so we do not add class weights here.
    training_args = TrainingArguments(
        output_dir=os.path.join(save_dir, "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,        # 16 tweets per step ...
        gradient_accumulation_steps=2,         # ... collected over 2 steps = effective batch of 32
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        warmup_steps=0.1,                      # a float below 1 = share of all steps (10% warm-up)
        weight_decay=0.01,
        fp16=True,                             # half-precision maths: faster, less memory
        gradient_checkpointing=True,           # recompute activations in the backward pass: less memory, slower
        gradient_checkpointing_kwargs={"use_reentrant": False},   # needed because the embeddings are frozen
        eval_strategy="epoch",                 # evaluate on validation after every epoch
        save_strategy="epoch",                 # save after every epoch ...
        save_only_model=True,                  # ... weights only (no optimizer state): smaller files
        save_total_limit=2,
        load_best_model_at_end=True,           # at the end, reload the epoch with the best val macro-F1
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        dataloader_num_workers=0,              # Windows: no extra worker processes
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=25,
        report_to="none",                      # no wandb / tensorboard
        disable_tqdm=True,                     # progress bars make the log unreadable
    )
    logger = EpochLogger()
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),   # pads each batch to its longest tweet
        compute_metrics=compute_metrics, callbacks=[logger],
    )

    # ---- Train
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    train_seconds = time.time() - start
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\ntraining finished in {train_seconds / 60:.1f} min | peak GPU memory: "
          f"{peak_allocated:.2f} GiB allocated, {peak_reserved:.2f} GiB reserved")

    # ---- Best model (already reloaded by load_best_model_at_end): save it and score validation
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    # The per-epoch checkpoints (about 1 GB each) are no longer needed: the best model is saved above
    shutil.rmtree(os.path.join(save_dir, "checkpoints"), ignore_errors=True)
    output = trainer.predict(val_ds)
    probs = torch.softmax(torch.tensor(output.predictions), dim=1).numpy()
    preds = probs.argmax(axis=1)
    true_ids = np.array(val_ds.labels)
    final = compute_metrics((output.predictions, true_ids))
    print("best model on validation:", {k: round(float(v), 4) for k, v in final.items()})

    wrong = preds != true_ids
    errors = pd.DataFrame({
        "id": val_df["id"], "text": val_df["text"], "cleaned_text": val_df["cleaned_text"],
        "true_label": val_df["label"], "predicted_label": [LABELS[i] for i in preds],
        "confidence": probs.max(axis=1).round(4),
    })[wrong].sort_values("confidence", ascending=False)
    errors.to_csv(os.path.join(args.results_dir, f"{run_name}_val_errors.csv"), index=False)

    summary = {
        "model": args.model, "seed": args.seed, "max_length": args.max_length,
        "total_params": total, "trainable_params": trainable, "embedding_params": embedding_params,
        "peak_gpu_allocated_gib": round(peak_allocated, 3), "peak_gpu_reserved_gib": round(peak_reserved, 3),
        "train_minutes": round(train_seconds / 60, 1), "best_val": {k: float(v) for k, v in final.items()},
        "epochs": logger.rows,
    }
    with open(os.path.join(args.results_dir, f"{run_name}_val_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved model to {save_dir}, {len(errors)} validation errors, and the summary JSON")


if __name__ == "__main__":
    main()
