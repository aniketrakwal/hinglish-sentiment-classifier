# Hinglish Sentiment Classifier

## Overview
End-to-end sentiment classification for Hinglish (Hindi-English code-mixed) text.
Compares a TF-IDF + Logistic Regression baseline, a PyTorch BiLSTM, and fine-tuned
transformers (MuRIL, HingRoBERTa, XLM-R) on identical data splits.

## Dataset
SemEval-2020 Task 9 SentiMix (Hinglish). _TODO: source link, size, label distribution._

## Pipeline
_TODO: parsing -> shared preprocessing -> models -> evaluation -> deployment._

## Results

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
|       |          |          |

## How to run
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python src/load_data.py --train data/raw/<train> --val data/raw/<val> --test data/raw/<test>
pytest
```

## Demo
_TODO: FastAPI + Gradio demo._
