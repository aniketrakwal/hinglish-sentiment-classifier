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
| HingRoBERTa + LoRA r=8 | `l3cube-pune/hing-roberta`, LoRA on query/value (alpha 2r, dropout 0.1), lr 2e-4, 6 epochs, other settings as the full fine-tune; mean ± std over seeds 42/43/44 | 0.6759 ± 0.0014 | not run | 0.887M trainable parameters (0.32%). Saved weights 3.4 MB (full fine-tune: 1060.7 MB). Per-class val F1: neg 0.711, neu 0.592, pos 0.725. Peak GPU memory 1.17 GiB (full: 2.29 GiB). |
| HingRoBERTa + LoRA r=16 | same, r=16 | 0.6774 ± 0.0035 | **0.7515 ± 0.0006** | **Best LoRA setup (chosen on validation before test was used).** 1.183M trainable parameters (0.42%). Saved weights 4.5 MB. Per-class val F1: neg 0.713, neu 0.595, pos 0.724. Test accuracy 0.7476 ± 0.0010; per-class test F1: neg 0.781, neu 0.678, pos 0.796. About 99 s/epoch, peak GPU memory 1.17 GiB. |
| XLM-R | `xlm-roberta-base`, same settings; mean ± std over seeds 42/43/44 | 0.6411 ± 0.0040 | not run | Val accuracy 0.6391 ± 0.0030. Per-class val F1: neg 0.665, neu 0.549, pos 0.709. Same architecture and tokenizer as HingRoBERTa without the Hinglish pre-training: HingRoBERTa is ahead by +0.0385 val macro-F1 (3 of 3 seeds). |

Notes
- Run `python src/train_lstm.py` to reproduce the BiLSTM rows (about 25 minutes on CPU). Per-run numbers are in `results/lstm_experiments.csv` and `results/lstm_test.csv`; the per-epoch log is `results/lstm_train_log.txt`.
- Two runs (BiLSTM seed 44 and LSTM seed 43) each contain one epoch where the machine paused (about 700 s instead of about 22 s), so `train_time_s` is inflated for those two rows. Scores are not affected.
- Transformers: `python src/run_transformers.py` (9 runs, about 75 min on an RTX 3050 Laptop), then `python src/compare_transformers.py` (validation comparison; writes `results/transformer_choice.txt`), then `python src/evaluate_winner.py` (test, winner only). Tokenizer comparison: `python src/tokenizer_report.py`.
- Only HingRoBERTa was evaluated on test. MuRIL and XLM-R were not, because the winner was chosen on validation first.
- Run `python src/train_ml.py` to reproduce. The full 15-run sweep (3 feature setups x 5 values of C) is in `results/ml_experiments.csv`.
- Test scores are about 6 points higher than validation scores. The cause has not been investigated. Do not read this as a sign of tuning on test: the test set was used exactly once, after the setup was chosen on validation.
- Validation has 3,000 tweets, so differences of about 1 point between setups are within noise.
- LoRA (HingRoBERTa): `python src/run_lora.py` (7 runs, about 90 min when the GPU runs at full speed), then `python src/compare_lora.py` (validation comparison; writes `results/lora_comparison.csv`, `results/lora_per_seed.csv` and `results/lora_choice.txt`), then `python src/evaluate_lora.py` (test, chosen setup only). The learning-rate check (r=8, seed 42) gave val macro-F1 0.6774 for lr 2e-4 and 0.6770 for 5e-4, so 2e-4 was used (`results/lora_lr_choice.txt`).
- LoRA vs full fine-tune on validation: r=8 is 0.0037 lower and r=16 is 0.0022 lower on average, and LoRA is lower in 3 of 3 seeds for both ranks. The seed std is 0.0014-0.0035, so the gap is about one seed-noise unit. r=16 beat r=8 by 0.0015 on average (higher in only 1 of 3 seeds): within noise.
- On test, LoRA r=16 (0.7515) is 0.0079 above the full fine-tune (0.7436 ± 0.0070). That is about one std of the full fine-tune's test seeds; LoRA was not better on validation, so do not read it as LoRA being better. Only LoRA r=16 and the full fine-tune were evaluated on test.
- Time per epoch is not a clean LoRA-vs-full comparison. The LoRA r=8 seed 42 run (and the 5e-4 check run) ran while the laptop GPU was power-throttled (about 223 s and 699 s per epoch), which inflates the r=8 average (141.8 s). Unthrottled, LoRA took about 95-102 s per epoch against 108 s for the full fine-tune, because gradient checkpointing (kept the same for both) dominates the time.

## Statistical evaluation

Step 8a. No model was trained and no model was chosen here. The saved probabilities (`results/predictions/`, made by `src/collect_predictions.py`) reproduce every macro-F1 above exactly. All intervals and tests below use these files.

### 95% bootstrap confidence intervals of macro-F1

1,000 resamples of the 3,000 tweets of a split, fixed seed. For models with 3 seeds each resample uses the mean F1 over the seeds. The interval only reflects *which tweets* are in the split, not the randomness of training. (`src/bootstrap.py` -> `results/bootstrap.csv`)

| Model | Val macro-F1 | Val 95% CI | Test macro-F1 | Test 95% CI |
|-------|--------------|------------|---------------|-------------|
| TF-IDF + LogReg | 0.6226 | [0.6059, 0.6395] | 0.6849 | [0.6681, 0.7013] |
| BiLSTM | 0.5870 | [0.5725, 0.6005] | 0.6318 | [0.6175, 0.6453] |
| LSTM (uni) | 0.5850 | [0.5708, 0.5982] | not run | |
| MuRIL | 0.6447 | [0.6292, 0.6598] | not run | |
| XLM-R | 0.6411 | [0.6262, 0.6556] | not run | |
| HingRoBERTa (full) | 0.6796 | [0.6647, 0.6936] | 0.7436 | [0.7304, 0.7585] |
| HingRoBERTa LoRA r=8 | 0.6759 | [0.6609, 0.6907] | not run | |
| HingRoBERTa LoRA r=16 | 0.6774 | [0.6623, 0.6917] | 0.7515 | [0.7379, 0.7659] |

Every interval is about 3 points wide (+-1.5). The val and test intervals of the same model do not overlap, so the 6-point test-over-val gap is not sampling noise.

### Paired bootstrap of differences (same resample for both models)

| A - B | Split | Difference | 95% CI | Clear? |
|-------|-------|-----------|--------|--------|
| HingRoBERTa (full) - TF-IDF | val | +0.0569 | [+0.0413, +0.0725] | yes |
| HingRoBERTa (full) - MuRIL | val | +0.0349 | [+0.0221, +0.0478] | yes |
| HingRoBERTa (full) - XLM-R | val | +0.0385 | [+0.0259, +0.0500] | yes |
| HingRoBERTa (full) - LoRA r=16 | val | +0.0022 | [-0.0054, +0.0095] | no (contains 0) |
| MuRIL - XLM-R | val | +0.0036 | [-0.0084, +0.0145] | no (contains 0) |
| HingRoBERTa (full) - TF-IDF | test | +0.0587 | [+0.0420, +0.0759] | yes |
| HingRoBERTa (full) - LoRA r=16 | test | -0.0079 | [-0.0148, -0.0004] | borderline (the upper end is almost 0) |

The test difference between full and LoRA is the only surprise (LoRA ahead). It comes from 3 seeds per model whose own seed-to-seed std on test is 0.0006 (LoRA) and 0.0070 (full); the bootstrap does not include training randomness, so treat it as "LoRA is at least as good", not as a proven win. It was not ahead on validation.

### Calibration (`src/calibration.py` -> `results/calibration.csv`, figures `results/figures/calibration_*.png`)

Temperature T fitted on validation only (minimise validation log-loss), then applied to test. Predictions (argmax) are identical before and after in every case, so accuracy and F1 do not change. Diagrams show HingRoBERTa seed 42 and TF-IDF.

| Model | T | Val ECE before -> after | Test ECE before -> after |
|-------|---|-------------------------|--------------------------|
| HingRoBERTa seed 42 | 1.62 | 0.128 -> 0.026 | 0.079 -> 0.047 |
| HingRoBERTa seed 43 | 1.34 | 0.079 -> 0.022 | 0.022 -> 0.058 |
| HingRoBERTa seed 44 | 1.44 | 0.098 -> 0.037 | 0.043 -> 0.045 |
| HingRoBERTa mean of 3 seeds | 1.47 | 0.102 -> 0.028 | 0.048 -> 0.050 |
| TF-IDF + LogReg | 0.86 | 0.035 -> 0.019 | 0.097 -> 0.070 |

- HingRoBERTa is over-confident on validation (T > 1 for every seed). Temperature scaling fixes this on validation (ECE 0.10 -> 0.03).
- On test the fitted T does not help on average (ECE 0.048 -> 0.050; worse for seeds 43 and 44). The model was already better calibrated on test, because it is more accurate there, so a T fitted on validation corrects too much.
- Mean confidence of HingRoBERTa (mean over seeds): correct predictions 0.80 (val) / 0.81 (test), wrong predictions 0.72 / 0.70. After scaling: correct 0.71 / 0.72, wrong 0.63 / 0.61. The model is only slightly more sure when right than when wrong (about 0.1), so its confidence is a weak warning sign for errors.
- TF-IDF is slightly under-confident (T = 0.86 sharpens it).

### Why is test easier than validation? (`src/val_test_gap.py`)

| Check | Val | Test |
|-------|-----|------|
| Label share neg / neu / pos (%) | 29.7 / 37.6 / 32.7 | 30.0 / 36.7 / 33.3 |
| Mean tokens per cleaned tweet | 20.3 | 20.3 |
| Tweets with emoji / URL / mention (%) | 18.9 / 56.8 / 81.0 | 19.9 / 53.3 / 81.5 |
| Devanagari / rough profanity / greeting words (%) | 0.5 / 8.4 / 7.3 | 0.5 / 8.3 / 7.4 |
| Tweets with a near-twin in train (%) | 1.9 (57 tweets) | 1.9 (56 tweets) |
| Tweets that all four systems get wrong (%) | 18.6 | 12.3 |
| Tweets that all four systems get right (%) | 43.2 | 47.9 |

- HingRoBERTa per-class F1, val -> test: negative 0.723 -> 0.772 (+0.049), neutral 0.591 -> 0.662 (+0.071), positive 0.725 -> 0.797 (+0.072). The gain is spread over all classes, a little less for negative. TF-IDF and the BiLSTM gain too (macro +0.062 and +0.045), so it is not specific to one model.
- Adversarial validation (TF-IDF word + char n-grams + logistic regression, 5-fold, val vs test): ROC-AUC 0.513 +- 0.010; with shuffled labels 0.520. A classifier cannot tell validation tweets from test tweets.

What the evidence supports or rules out:
- Ruled out: sampling noise (the intervals do not overlap); a different label mix, tweet length, emoji / URL / mention / Devanagari / profanity / greeting content (all nearly identical); a different kind of text (adversarial AUC about 0.5); near-duplicate leakage from train (same 1.9% in both, only about 56 tweets, worth about 0.1 point at most).
- Supported: test tweets simply get fewer hard cases. The share of tweets every system gets wrong falls from 18.6% to 12.3%, and all models improve, with a similar size in the three model families. Since the texts look the same, the likely cause is on the label side: the test labels are more predictable from the text than the validation labels (for example less label noise or a slightly different annotation of the test set, which was released with its labels in a separate file). Weak extra hint: for tweets with a near-twin in train, the test label agrees with a train label for 70% of them, against 58% for validation (only about 56 tweets each, so this alone is not significant).
- Cannot be shown from this data: the exact reason for the label difference. This is a consequence of using only the files we have, not a model problem.
- Practical consequence: compare models within one split, never across val and test. Test scores are about 6 points higher for every model, and the *ranking* of models is the same on both.

Reproduce: `python src/collect_predictions.py` (needs the GPU and the saved models, about 15 min; the other three scripts only read the saved prediction files and run in under a minute each on the CPU), then `python src/bootstrap.py`, `python src/calibration.py`, `python src/val_test_gap.py`.
