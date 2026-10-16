"""Train and compare BiLSTM vs unidirectional LSTM, 3 seeds each. Run from the project root:

    python src/train_lstm.py

All choices are made on VALIDATION. The test set is touched once per checkpoint of the chosen
configuration, at the very end.
"""

import argparse
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from lstm_data import LABELS, make_loader, prepare_train_val, load_test_dataset
from lstm_model import BiLSTMClassifier

SEEDS = [42, 43, 44]
CONFIGS = {"bilstm": True, "lstm": False}   # name -> bidirectional?
MAX_EPOCHS = 20
PATIENCE = 3          # stop if val macro-F1 has not improved for 3 epochs in a row
LEARNING_RATE = 1e-3
CLIP_NORM = 1.0


def set_seed(seed):
    """Make a run repeatable: fix every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balanced_class_weights(labels):
    """weight = n_samples / (n_classes * count of the class): rarer class -> bigger weight.
    This is the same formula as class_weight="balanced" in the scikit-learn baseline."""
    counts = torch.bincount(labels, minlength=len(LABELS)).float()
    return len(labels) / (len(LABELS) * counts)


def train_step(model, batch, loss_fn, optimizer, device):
    """One update of the model on one batch. Returns the batch loss as a plain number."""
    ids, lengths, labels = batch
    ids, labels = ids.to(device), labels.to(device)          # lengths stay on the CPU (needed by packing)
    model.train()                                            # turn dropout ON
    optimizer.zero_grad()                                    # forget the gradients of the previous batch
    logits = model(ids, lengths)                             # forward pass: [batch, 3] scores
    loss = loss_fn(logits, labels)                           # how wrong are the scores?
    loss.backward()                                          # backward pass: gradient of the loss
    nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)  # cap the gradient size (stops exploding updates)
    optimizer.step()                                         # Adam changes the weights a little
    return loss.item()


def evaluate(model, loader, loss_fn, device):
    """Run the model over a whole loader without learning. Returns loss, predictions, probabilities."""
    model.eval()                                             # turn dropout OFF
    total_loss, n_seen = 0.0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():                                    # no gradients needed: faster, less memory
        for ids, lengths, labels in loader:
            ids, labels_d = ids.to(device), labels.to(device)
            logits = model(ids, lengths)
            loss = loss_fn(logits, labels_d)
            total_loss += loss.item() * len(labels)          # batch loss x batch size
            n_seen += len(labels)
            all_probs.append(torch.softmax(logits, dim=1).cpu())   # scores -> probabilities
            all_labels.append(labels)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = probs.argmax(axis=1)                             # the class with the highest probability
    return total_loss / n_seen, preds, probs, labels


def metrics(labels, preds):
    """Accuracy, macro-F1 and the F1 of each class (labels and preds are ids 0/1/2)."""
    per_class = f1_score(labels, preds, labels=[0, 1, 2], average=None)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, labels=[0, 1, 2], average="macro"),
        "f1_negative": per_class[0], "f1_neutral": per_class[1], "f1_positive": per_class[2],
    }


def train_one_run(config, seed, data, class_weights, device, model_dir):
    """Train one (config, seed) with early stopping. Returns a summary dict (incl. the history)."""
    set_seed(seed)
    model = BiLSTMClassifier(vocab_size=len(data["word2idx"]), bidirectional=CONFIGS[config]).to(device)
    # Weighted loss: mistakes on rarer classes cost more. Used for val loss too, so the curves compare.
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(data["train_ds"], shuffle=True, seed=seed)    # only train is shuffled
    val_loader = make_loader(data["val_ds"], shuffle=False)

    checkpoint = os.path.join(model_dir, f"lstm_{config}_seed{seed}.pt")
    best_f1, best_epoch, epochs_without_gain = -1.0, 0, 0
    history = []
    run_start = time.time()
    print(f"\n--- {config}, seed {seed} ---")

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        batch_losses = [train_step(model, batch, loss_fn, optimizer, device) for batch in train_loader]
        train_loss = float(np.mean(batch_losses))
        val_loss, val_preds, _, val_labels = evaluate(model, val_loader, loss_fn, device)
        val_f1 = metrics(val_labels, val_preds)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": val_f1})
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"val macro-F1 {val_f1:.4f}  ({time.time() - epoch_start:.1f}s)")

        if val_f1 > best_f1:                                 # new best: save a checkpoint
            best_f1, best_epoch, epochs_without_gain = val_f1, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:              # early stopping
                print(f"early stop: no improvement for {PATIENCE} epochs")
                break
    train_time = time.time() - run_start

    # Reload the best checkpoint and score it, so the reported numbers come from the saved model
    model.load_state_dict(torch.load(checkpoint))
    _, preds, probs, labels = evaluate(model, val_loader, loss_fn, device)
    result = {"config": config, "seed": seed, "best_epoch": best_epoch}
    result.update({"val_" + k: v for k, v in metrics(labels, preds).items()})
    result["train_time_s"] = round(train_time, 1)
    return result, history, preds, probs, checkpoint


def plot_curves(history, title, path):
    """Left: train and val loss per epoch. Right: val macro-F1 per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", label="train loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], marker="o", label="val loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], marker="o", color="green")
    best = max(history, key=lambda h: h["val_macro_f1"])
    axes[1].axvline(best["epoch"], linestyle="--", color="grey", label=f"best epoch {best['epoch']}")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("macro-F1"); axes[1].legend()
    axes[1].set_title("Validation macro-F1")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def mean_std(series):
    return f"{series.mean():.4f} ± {series.std():.4f}"       # std over the 3 seeds (sample std)


def main():
    parser = argparse.ArgumentParser(description="BiLSTM vs LSTM, 3 seeds each.")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()
    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA GPU available:", torch.cuda.is_available(), "-> using", device)

    data = prepare_train_val(args.data_dir)                       # test is NOT loaded yet
    with open(os.path.join(args.model_dir, "lstm_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(data["word2idx"], f, ensure_ascii=False)        # needed later to serve the model
    class_weights = balanced_class_weights(data["train_ds"].labels)
    print("class weights (negative, neutral, positive):", [round(w, 3) for w in class_weights.tolist()])

    # ---- Run every (config, seed)
    rows, histories, val_outputs, checkpoints = [], {}, {}, {}
    for config in CONFIGS:
        for seed in SEEDS:
            result, history, preds, probs, ckpt = train_one_run(
                config, seed, data, class_weights, device, args.model_dir)
            rows.append(result)
            histories[(config, seed)] = history
            val_outputs[(config, seed)] = (preds, probs)
            checkpoints[(config, seed)] = ckpt

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.results_dir, "lstm_experiments.csv"), index=False, float_format="%.4f")
    print("\n" + table.round(4).to_string(index=False))

    print("\nMean ± std over seeds (validation)")
    for config in CONFIGS:
        part = table[table["config"] == config]
        print(f"  {config:7s} macro-F1 {mean_std(part['val_macro_f1'])}   accuracy {mean_std(part['val_accuracy'])}   "
              f"best epoch {part['best_epoch'].mean():.1f}   time {part['train_time_s'].mean():.0f}s")

    # ---- Training curves: seed 42 of each config
    for config in CONFIGS:
        plot_curves(histories[(config, SEEDS[0])], f"{config}, seed {SEEDS[0]}",
                    os.path.join(fig_dir, f"lstm_curves_{config}.png"))

    # ---- Choose the config with the best MEAN validation macro-F1
    means = table.groupby("config")["val_macro_f1"].mean()
    chosen = means.idxmax()
    print(f"\nChosen config (best mean val macro-F1): {chosen}")

    # ---- Test set: used once per checkpoint of the chosen config
    test_df, test_ds = load_test_dataset(data["word2idx"], args.data_dir)
    test_loader = make_loader(test_ds, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    test_rows = []
    for seed in SEEDS:
        model = BiLSTMClassifier(vocab_size=len(data["word2idx"]), bidirectional=CONFIGS[chosen]).to(device)
        model.load_state_dict(torch.load(checkpoints[(chosen, seed)]))
        _, preds, _, labels = evaluate(model, test_loader, loss_fn, device)
        row = {"config": chosen, "seed": seed}
        row.update({"test_" + k: v for k, v in metrics(labels, preds).items()})
        test_rows.append(row)
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(os.path.join(args.results_dir, "lstm_test.csv"), index=False, float_format="%.4f")
    print("\nTEST results of the chosen config")
    print(test_table.round(4).to_string(index=False))
    print(f"  test macro-F1 {mean_std(test_table['test_macro_f1'])}   test accuracy {mean_std(test_table['test_accuracy'])}")

    # ---- Validation errors of the best seed of the chosen config
    part = table[table["config"] == chosen]
    best_seed = int(part.loc[part["val_macro_f1"].idxmax(), "seed"])
    preds, probs = val_outputs[(chosen, best_seed)]
    val_df = data["val_df"]
    true_ids = data["val_ds"].labels.numpy()
    wrong = preds != true_ids
    errors = pd.DataFrame({
        "id": val_df["id"], "text": val_df["text"], "cleaned_text": val_df["cleaned_text"],
        "true_label": val_df["label"], "predicted_label": [LABELS[i] for i in preds],
        "confidence": probs.max(axis=1).round(4),
    })[wrong].sort_values("confidence", ascending=False)
    errors.to_csv(os.path.join(args.results_dir, "lstm_val_errors.csv"), index=False)
    print(f"\nSaved {len(errors)} misclassified validation tweets (config {chosen}, seed {best_seed})")


if __name__ == "__main__":
    main()
