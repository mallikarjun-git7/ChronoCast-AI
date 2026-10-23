# AI Time Series Forecasting

A course project exploring weekly time series forecasting with a stacked **PyTorch LSTM**, alongside Keras LSTM, Prophet, ARIMA and Transformer experiments.

Developed for **COMP 5511: Artificial Intelligence Concepts** at **The Hong Kong Polytechnic University** in 2025. This repository presents the implementation, selected experiment artifacts and an English guide to the results.

![Archived LSTM training, validation and recursive forecast](assets/lstm_forecast.png)

*Archived figure from the LSTM exploration notebook. The upper-right panel shows one-step validation predictions; the lower panel shows a recursive forecast without test ground truth.*

## Project at a glance

| Item | Description |
| --- | --- |
| Task | Forecast an anonymized weekly indicator |
| Main implementation | Two-layer LSTM with 128 hidden units and a 64 → 32 → 1 prediction head |
| Input | Previous 12 standardized target values |
| Data | 1,325 training rows, 166 validation rows, 162 future timestamps |
| Training | Adam, MSE loss, learning-rate scheduling and early stopping |
| Final forecast | Retrain on training + validation data, then predict 162 steps recursively |
| Other experiments | Keras LSTM, Prophet, ARIMA(5, 1, 2), Transformer and an equal-weight ensemble |

Start with [the final LSTM notebook](notebooks/01_lstm_final.ipynb). The [experiment notes](docs/EXPERIMENT_NOTES.md) explain how the saved results relate to the code and historical report.

## Method

The main notebook loads the chronological training and validation splits, fits a `StandardScaler` on the training targets, and forms sliding windows of 12 observations. Each window predicts the next value.

```text
12 observed target values
          ↓
Training-fitted standardization
          ↓
Two-layer LSTM · 128 hidden units · dropout 0.2
          ↓
Dense 64 → ReLU → dropout → Dense 32 → ReLU → dropout → Dense 1
          ↓
Inverse scaling → next target value
```

Training uses a batch size of 32, an initial learning rate of 0.001, a maximum of 100 epochs and early stopping after 15 epochs without improvement. A separate final model is trained for 50 epochs on all 1,491 observed values. Its predictions are fed back into the input window to produce the future sequence.

The notebook also computes lag, rolling, difference and cyclical features. **These engineered features are not passed to the final model:** the actual network has `input_size=1` and uses only the target history.

## Recorded results

The final notebook's saved output reports the following values. They are historical results, not measurements from a new training run during repository preparation.

| Metric | Saved value |
| --- | ---: |
| MSE | 3.431400 |
| RMSE | 1.852404 |
| MAE | 1.513808 |
| MAPE | 0.790081% |
| R² | 0.821874 |

Source: [saved training log](results/lstm/training_log.txt) and [structured metrics](results/lstm/metrics.json).

These metrics cover **154 one-step predictions**: the first 12 of the 166 validation rows provide the initial history. Each prediction uses observed validation values from its preceding window. The validation split also selects the checkpoint, so it is not an untouched test set. The **162-step recursive forecast is a different evaluation setting**; its accuracy is not established by this table.

The earlier multi-model experiment is preserved in [the ensemble results](results/ensemble/model_comparison.csv). It uses different validation horizons for neural models and ARIMA, and its neural-model results differ substantially from the final notebook. It is not presented as a controlled ranking. No verified Kaggle leaderboard score is included.

## Run locally

### 1. Set up an environment

Use Python 3.10 or 3.11 for the documented setup. The original final notebook records Python 3.10.19; the historical report names PyTorch 2.7.0. The dependency files reconstruct the imported packages rather than reproduce a preserved environment lockfile.

```bash
git clone https://github.com/langezhang/AI-Time-Series-Forecasting.git
cd AI-Time-Series-Forecasting
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install the main dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The original notebook runs on CPU. For a CPU-specific PyTorch installation on Windows or Linux, use `python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu` before installing the requirements. See the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).

### 2. Provide the data

Place the course competition files in `data/raw/`:

```text
data/raw/
├── train.csv    # Relative Date, Predict Target
├── val.csv      # Relative Date, Predict Target
└── test.csv     # Relative Date
```

The source dataset is not distributed in this public repository. See [data/README.md](data/README.md) for its expected format and access information. Do not substitute the unrelated text file from the original course folder.

Validate the inputs without loading any machine-learning dependencies:

```bash
python scripts/run_notebook.py lstm --check-data
```

### 3. Run the final notebook

```bash
python scripts/run_notebook.py lstm
```

This executes the notebook's Python cells in order and starts the original training procedure. It writes generated files to `outputs/lstm/`:

- `best_model.pth`: checkpoint selected on validation loss.
- `submission2.csv`: 162 forecast rows for the original test file, with `Relative Date` and `Predict Target` columns.

The final retrained forecasting model is not saved separately by the original code; `best_model.pth` belongs to the validation stage. Training time and numerical results depend on the environment.

For interactive exploration, run `jupyter lab` from the repository root, open a notebook and run all cells from the top.

### Other experiments

| Command | Notebook | Additional setup |
| --- | --- | --- |
| `python scripts/run_notebook.py lstm-exploration` | [LSTM exploration](notebooks/02_lstm_exploration.ipynb) | Main requirements |
| `python scripts/run_notebook.py prophet` | [Prophet baseline](notebooks/04_prophet_baseline.ipynb) | `python -m pip install -r requirements/prophet.txt` |
| `python scripts/run_notebook.py ensemble` | [Multi-model experiment](notebooks/05_ensemble_exploration.ipynb) | `python -m pip install -r requirements/ensemble.txt` |
| `python scripts/run_notebook.py keras` | [Keras baseline](notebooks/03_lstm_keras_baseline.ipynb) | Separate Python 3.10/3.11 environment with `requirements/keras.txt` |

Each experiment writes into its own `outputs/<experiment>/` directory. These files are ignored by Git. The exploratory notebooks preserve their original modeling behavior and are not guaranteed to reproduce the final LSTM's results.

## Repository layout

```text
AI-Time-Series-Forecasting/
├── notebooks/                 # Five curated course notebooks
├── scripts/run_notebook.py    # Notebook runner and input checks
├── data/README.md             # Input schema and access notes
├── assets/lstm_forecast.png   # Selected historical visualization
├── results/                  # Saved logs, metrics and selected forecasts
├── docs/
│   ├── EXPERIMENT_NOTES.md    # Evidence, limitations and artifact provenance
│   ├── course_report.pdf     # Archived report body, identifying cover removed
│   └── source_manifest.json  # Hashes of source notebooks
├── requirements/             # Dependencies for optional experiments
└── requirements.txt          # Main LSTM dependencies
```

## Scope and limitations

This is a coursework portfolio, with the original model logic preserved. Repository preparation standardizes paths, separates generated outputs, clears notebook execution output and documents the available evidence. It does not introduce a new model or rerun the experiments.

The [archived report](docs/course_report.pdf) contains historical claims that do not all match the submitted code or available logs, including its engineered-feature interpretation and baseline comparisons. Read it alongside [the implementation notes](docs/EXPERIMENT_NOTES.md). Long-horizon backtesting, a consistent cross-model evaluation protocol and a fully locked training environment remain future work.

Original classroom handouts, duplicate report drafts, student identifiers, local model binaries and unrelated materials are excluded from the public repository. No open-source license is declared in this archive.
