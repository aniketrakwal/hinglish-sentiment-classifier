# Results

All models use the same `clean_text()` (`src/preprocess.py`) and the same saved train/val/test splits.
Hyper-parameters are tuned on validation only; test is evaluated once per final model.

| Model | Setup | Val macro-F1 | Test macro-F1 | Notes |
|-------|-------|--------------|---------------|-------|
| Dummy | `most_frequent` (always "neutral") | 0.1822 | 0.1789 | Floor. Accuracy: val 0.3760, test 0.3667. |
| TF-IDF + Logistic Regression | char_wb (2,5) n-grams, C=0.5, `class_weight="balanced"`, 66,534 features | 0.6226 | 0.6849 | Accuracy: val 0.6223, test 0.6837. Per-class val F1: neg 0.659, neu 0.519, pos 0.690. Neutral is the weak class. |
| BiLSTM (PyTorch) | bidirectional, emb 100, hidden 128, dropout 0.3, class-weighted loss, Adam 1e-3, early stop on val macro-F1; mean ± std over seeds 42/43/44 | 0.5870 ± 0.0045 | 0.6318 ± 0.0018 | Accuracy: val 0.5866 ± 0.0053, test 0.6306 ± 0.0021. Best epoch about 5. Below the TF-IDF baseline. |
| LSTM (unidirectional) | same, `bidirectional=False`; 3 seeds | 0.5850 ± 0.0092 | not run | Ablation only. Not chosen (lower mean val macro-F1), so test was not evaluated. |
| MuRIL | `google/muril-base-cased`, max_length 64, lr 2e-5, effective batch 32 (16 x 2), 4 epochs, warm-up 10%, weight decay 0.01, fp16, gradient checkpointing, word embeddings frozen; mean ± std over seeds 42/43/44 | 0.6447 ± 0.0030 | not run | Val accuracy 0.6408 ± 0.0027. Per-class val F1: neg 0.664, neu 0.562, pos 0.708. 237.6M parameters (86.0M trainable). Its tokenizer maps every emoji to [UNK]. About 108 s/epoch, peak GPU memory 2.13 GiB. |
| HingRoBERTa | `l3cube-pune/hing-roberta`, same settings; mean ± std over seeds 42/43/44 | **0.6796 ± 0.0028** | **0.7436 ± 0.0070** | **Winner (chosen on validation before test was used).** Val accuracy 0.6770 ± 0.0019; per-class val F1: neg 0.723, neu 0.591, pos 0.725. Test accuracy 0.7404 ± 0.0081; per-class test F1: neg 0.772, neu 0.662, pos 0.797. 278.0M parameters (86.0M trainable). About 108 s/epoch, peak GPU memory 2.29 GiB. |
| XLM-R | `xlm-roberta-base`, same settings; mean ± std over seeds 42/43/44 | 0.6411 ± 0.0040 | not run | Val accuracy 0.6391 ± 0.0030. Per-class val F1: neg 0.665, neu 0.549, pos 0.709. Same architecture and tokenizer as HingRoBERTa without the Hinglish pre-training: HingRoBERTa is ahead by +0.0385 val macro-F1 (3 of 3 seeds). |

Notes
- Run `python src/train_lstm.py` to reproduce the BiLSTM rows (about 25 minutes on CPU). Per-run numbers are in `results/lstm_experiments.csv` and `results/lstm_test.csv`; the per-epoch log is `results/lstm_train_log.txt`.
- Two runs (BiLSTM seed 44 and LSTM seed 43) each contain one epoch where the machine paused (about 700 s instead of about 22 s), so `train_time_s` is inflated for those two rows. Scores are not affected.
- Transformers: `python src/run_transformers.py` (9 runs, about 75 min on an RTX 3050 Laptop), then `python src/compare_transformers.py` (validation comparison; writes `results/transformer_choice.txt`), then `python src/evaluate_winner.py` (test, winner only). Tokenizer comparison: `python src/tokenizer_report.py`.
- Only HingRoBERTa was evaluated on test. MuRIL and XLM-R were not, because the winner was chosen on validation first.
- Run `python src/train_ml.py` to reproduce. The full 15-run sweep (3 feature setups x 5 values of C) is in `results/ml_experiments.csv`.
- Test scores are about 6 points higher than validation scores. The cause has not been investigated. Do not read this as a sign of tuning on test: the test set was used exactly once, after the setup was chosen on validation.
- Validation has 3,000 tweets, so differences of about 1 point between setups are within noise.
