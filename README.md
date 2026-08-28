# Geology Forecast Challenge

My work on Kaggle's [Geology Forecast Challenge (open)](https://www.kaggle.com/competitions/geology-forecast-challenge-open) — forecasting the next 300 points of a geological horizon's depth from the previous 300, with 10 output realizations per well to express uncertainty.

Competition data isn't included in this repo (see `.gitignore`) — download `train.csv`, `test.csv`, `sample_submission.csv` and `train_raw/` from the [competition's Data tab](https://www.kaggle.com/competitions/geology-forecast-challenge-open/data) and place them under `data/`.

## Notebooks

- **`data/baseline_trend_forecast.ipynb`** — dependency-free baseline (Python standard library only): persistence vs. linear-trend extrapolation, honest 9-realization uncertainty via calibrated random walk, first valid `submission.csv`.
- **`data/advanced_conv_attention_forecast.ipynb`** — the advanced pipeline: sliding-window augmentation grouped by well, enriched features (missing-value mask, history length, local slope, sinusoidal positional encoding), a Conv1D + BiLSTM + self-attention model trained with a weighted-MSE loss, well-grouped cross-validation with 5-fold ensembling, and MC-Dropout for genuinely diverse realizations.
- **`data/geology_forecast_complete.ipynb`** — both of the above combined into one executable pipeline, in the order they were built.
- **`data/public-11st-private-4th.ipynb`** — a public write-up (not mine) of a 4th-place private-leaderboard LSTM solution, kept for reference/comparison.

## Environment

Validated locally on Python 3.12 (64-bit) — see `requirements.txt`. `scikit-learn` is deliberately not required; `GroupKFold` is replaced with a small hand-written `group_k_fold()`.

```
uv pip install --break-system-packages -r requirements.txt
```

`.vscode/settings.json` points the workspace at the interpreter used to validate everything, and `.env` quiets TensorFlow's noisy startup logs.

## Known limitations

- The exact weighted-MSE formula used by the leaderboard wasn't accessible from this environment (Kaggle's *Evaluation* page is client-rendered); the notebooks use a documented linear-decay proxy weighting instead.
- Local runs are CPU-only (native TensorFlow on Windows has no GPU support outside WSL) — training is faster on Kaggle's own GPU accelerator.
