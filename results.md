# Results

All models use the same `clean_text()` (`src/preprocess.py`) and the same saved train/val/test splits.
Hyper-parameters are tuned on validation only; test is evaluated once per final model.

| Model | Setup | Val macro-F1 | Test macro-F1 | Notes |
|-------|-------|--------------|---------------|-------|
| Dummy | `most_frequent` (always "neutral") | 0.1822 | 0.1789 | Floor. Accuracy: val 0.3760, test 0.3667. |
| TF-IDF + Logistic Regression | char_wb (2,5) n-grams, C=0.5, `class_weight="balanced"`, 66,534 features | 0.6226 | 0.6849 | Accuracy: val 0.6223, test 0.6837. Per-class val F1: neg 0.659, neu 0.519, pos 0.690. Neutral is the weak class. |
| BiLSTM (PyTorch) | bidirectional, emb 100, hidden 128, dropout 0.3, class-weighted loss, Adam 1e-3, early stop on val macro-F1; mean ± std over seeds 42/43/44 | 0.5870 ± 0.0045 | 0.6318 ± 0.0018 | Accuracy: val 0.5866 ± 0.0053, test 0.6306 ± 0.0021. Best epoch about 5. Below the TF-IDF baseline. |
| LSTM (unidirectional) | same, `bidirectional=False`; 3 seeds | 0.5850 ± 0.0092 | not run | Ablation only. Not chosen (lower mean val macro-F1), so test was not evaluated. |

Notes
- Run `python src/train_lstm.py` to reproduce the BiLSTM rows (about 25 minutes on CPU). Per-run numbers are in `results/lstm_experiments.csv` and `results/lstm_test.csv`; the per-epoch log is `results/lstm_train_log.txt`.
- Two runs (BiLSTM seed 44 and LSTM seed 43) each contain one epoch where the machine paused (about 700 s instead of about 22 s), so `train_time_s` is inflated for those two rows. Scores are not affected.
- Run `python src/train_ml.py` to reproduce. The full 15-run sweep (3 feature setups x 5 values of C) is in `results/ml_experiments.csv`.
- Test scores are about 6 points higher than validation scores. The cause has not been investigated. Do not read this as a sign of tuning on test: the test set was used exactly once, after the setup was chosen on validation.
- Validation has 3,000 tweets, so differences of about 1 point between setups are within noise.
