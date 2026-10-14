# Hinglish Sentiment Classifier

## Goal
End-to-end sentiment classifier for Hinglish (Hindi-English code-mixed) text,
using the SemEval-2020 Task 9 SentiMix dataset (labels: positive / neutral / negative).
Three approaches are compared on the SAME data splits:
1. TF-IDF + Logistic Regression (baseline)
2. BiLSTM (PyTorch)
3. Fine-tuned transformers: MuRIL, HingRoBERTa, XLM-R

Later: deploy with FastAPI + Gradio.

## Folder structure
- `data/raw/`        original downloaded dataset files (git-ignored)
- `data/processed/`  train.csv / val.csv / test.csv produced by `src/load_data.py` (git-ignored)
- `src/`             all reusable scripts (parsing, preprocessing, training, evaluation)
- `notebooks/`       exploration only (EDA, plots)
- `app/`             FastAPI + Gradio demo (later)
- `configs/`         YAML configs (hyperparameters, paths)
- `tests/`           pytest tests

## Rules
- ALL models must use the same preprocessing function (one shared function in `src/`)
  and the same saved train/val/test splits from `data/processed/`. Never re-split per model.
- Scripts go in `src/`. Notebooks are for exploration only; no logic lives only in a notebook.
- Keep code simple and well commented. The author must be able to explain every line in interviews:
  no clever one-liners, no unnecessary abstractions or dependencies.
- Processed CSV columns: `id, text, label, lang_tags` (`lang_tags` is space-separated, aligned with tokens in `text`).
