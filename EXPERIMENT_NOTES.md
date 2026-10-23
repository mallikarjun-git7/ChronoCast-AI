# Experiment Notes

## Evidence and provenance

The final LSTM notebook, exploratory notebooks, saved outputs and submitted report are separate artifacts. No models were retrained when preparing this repository. The code and its saved output take precedence when describing what was actually implemented.

| Published artifact | Original role | Treatment |
| --- | --- | --- |
| `notebooks/01_lstm_final.ipynb` | Final code submission | Model logic preserved; repository paths and a setup cell added |
| `notebooks/02_lstm_exploration.ipynb` | Earlier PyTorch LSTM experiment | Same path cleanup; includes the plotting code |
| `notebooks/03_lstm_keras_baseline.ipynb` | Initial TensorFlow/Keras experiment | Machine-specific data paths replaced |
| `notebooks/04_prophet_baseline.ipynb` | Prophet experiment | Machine-specific data paths replaced |
| `notebooks/05_ensemble_exploration.ipynb` | LSTM, Transformer and ARIMA comparison | Separate generated-output directory |
| `results/lstm/training_log.txt` | Final notebook stdout | Preserved as text; metrics transcribed to JSON |
| `assets/lstm_forecast.png` | LSTM exploration figure | Copied without altering the image |
| `results/lstm-exploration/submission.csv` | Saved exploratory forecast | Historical file; not identified as the final submission |
| `results/ensemble/*.csv` | Archived comparison and forecasts | Copied without changing numeric values |
| `docs/course_report.pdf` | Submitted course report | Identifying cover removed; body preserved; metadata replaced |

The original notebook hashes are recorded in [source_manifest.json](source_manifest.json). Notebook output was cleared in the curated copies, and kernels were renamed to a standard Python kernel. Original comments may remain in Chinese. The original local course folder is retained separately.

## What the final LSTM actually uses

The code calls `create_features`, which generates 37 additional columns: 12 lags, 20 rolling statistics, two differences and three weekly/cyclical columns. The resulting frames are never passed into `create_sequences` or `LSTMModel`.

The active data flow is:

```text
Target column → StandardScaler → 12-value windows → LSTM(input_size=1)
```

The report's attribution of the observed accuracy to engineered features is therefore unsupported by the submitted implementation. This repository documents that mismatch without changing the original model.

## Validation and forecasting are different tasks

- Standardization for validation is fitted on training targets only.
- Validation windows are formed within the validation split. The first 12 rows are context, leaving 154 predictions.
- Each validation prediction uses the preceding observed values, including values inside the validation split.
- Validation loss is also used for learning-rate scheduling, early stopping and checkpoint selection.
- The future forecast comes from a separate model retrained on all 1,491 available targets. A new scaler is fitted on those combined targets.
- Forecasting then feeds predictions back into the input window for all 162 future timestamps.

The recorded 0.790081% MAPE describes the one-step validation protocol. It is not a measured 162-step test error, a competition score or a guarantee of future accuracy.

## Historical baseline results

The archived ensemble table records:

| Model | MSE | RMSE | MAE | MAPE (%) |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 8958.242971 | 94.647995 | 94.546733 | 49.427237 |
| Transformer | 8812.546023 | 93.875162 | 93.772663 | 49.022040 |
| ARIMA | 109.182117 | 10.449025 | 9.330337 | 4.928522 |

Source: [archived CSV](../results/ensemble/model_comparison.csv). These numbers are not recomputed results.

The neural models in that notebook predict 154 validation targets from observed windows, while ARIMA forecasts all 166 validation targets from the training boundary. These different horizons and context rules prevent treating the table as a controlled comparison. The cause of the large discrepancy between the earlier neural results and the final notebook has not been established by retraining or systematic debugging.

The original Prophet notebook records a validation MSE of 459.8065398324315. The submitted report instead lists other baseline values, including a GRU result for which no implementation was found among the project notebooks. Those report values are not adopted as verified benchmarks in the README.

## Reproduction status

The repository contains the original algorithms with path and notebook-metadata cleanup. Its setup files reconstruct dependencies from imports; they are not the original environment lock. The final notebook metadata records Python 3.10.19, while the report describes Python 3.12 and PyTorch 2.7.0. This inconsistency is preserved here rather than resolved by assumption.

The main notebook seeds NumPy and PyTorch with 42, but does not configure full deterministic execution. Its model and tensors remain on CPU. The saved checkpoint represents the validation model; the final full-data model and fitted scalers are not exported for standalone inference.

The source notebooks, input schema, archived predictions, documentation links and repository paths can be checked without rerunning training. A complete training reproduction is outside the scope of this publication pass.

Publication checks passed for all five notebook schemas and Python syntax. After normalizing file paths, their Python abstract syntax trees match the original code. Local data checks confirmed the original 1,325 / 166 / 162 row counts. A small CPU forward-pass check with the available PyTorch 2.8.0 environment confirmed 154 validation windows and a finite `(32, 1)` batch output. This smoke check does not validate the reconstructed environment or reproduce the saved metrics; a fresh installation of the documented dependencies has not been tested.

## Excluded materials

The public repository omits raw competition data, course specifications and lecture materials, report drafts, presentation drafts, original ZIP submissions, local checkpoints and unrelated files. Access to the original dataset is described in [data/README.md](../data/README.md).
