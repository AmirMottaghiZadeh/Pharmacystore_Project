# PharmacyStore

Forecast weekly unique prescriptions per drug (UPW) for an Iranian pharmacy using XGBoost on tabular data.

## Highlights
- Limited 10-month dataset and tabular features
- SQL Server extraction pipeline
- Feature engineering: weekly aggregates, rolling stats, ATC classification, holiday effects
- XGBoost training with time-based validation and walk-forward evaluation

## Prerequisites
- Python >= 3.11
- ODBC Driver 18 for SQL Server installed on the host
  - Linux: `msodbcsql18`
  - Windows: "ODBC Driver 18 for SQL Server"
- Docker
  - Linux: `restore_mssql_docker.sh` can install Docker if missing
  - Windows: Docker Desktop must be installed and running
- Network access for pip and Docker image pulls

## Quick Start (run_all.py)
1. Put exactly one `.bak` file in `data/external/`.
2. Run:
   `python run_all.py`
3. When prompted, enter the SQL Server SA password (it is not stored).
4. Outputs are written to `data/processed/` and `artifacts/`.

## run_all.py Execution Flow (Detailed)
1. `run_all.py` (root) calls `scripts/run_all.py`.
2. It prompts for the SQL password and exports it only for this process:
   - `SQL_PASS` (for restore scripts)
   - `PHARMACYSTORE_SQL_PASSWORD` (for the pipeline)
3. Restore step (OS-specific):
   - Windows: runs
     `powershell -ExecutionPolicy Bypass -File .\restore_mssql_docker.ps1 -DB_NAME PharmacyStore`
   - Linux: runs `bash scripts/restore_db.sh`, which calls `restore_mssql_docker.sh PharmacyStore`
4. The restore script:
   - Creates/recreates a Docker container `mssql_restore` (SQL Server 2022)
   - Maps host port `14333` to container port `1433`
   - Copies the single `.bak` from `data/external/` into the container
   - Detects logical names via `RESTORE FILELISTONLY`
   - Restores the database with `REPLACE` and returns to MULTI_USER
   - Updates `.env` with connection settings (except the password)
5. `scripts/run_all.py` validates required external files and installs dependencies.
6. It runs `python -m pharmacystore.pipeline full` to extract data, build features, train models, and export outputs.

## Docker Restore Scripts
Linux:
- `restore_mssql_docker.sh` (root): does the full Docker restore
- `scripts/restore_db.sh`: small wrapper used by `run_all.py`

Windows:
- `restore_mssql_docker.ps1` (root): should restore the `.bak` using Docker and accept `-DB_NAME`

If you want a different database name, run the restore script directly with your name and update `.env`:
- Linux: `bash restore_mssql_docker.sh MyDb`
- Windows: `powershell -ExecutionPolicy Bypass -File .\restore_mssql_docker.ps1 -DB_NAME MyDb`

## Required Input Files
- Exactly one `.bak` file in `data/external/` (used by restore scripts)
- `data/external/who_atc_ddd.csv`
- `data/external/holidays/weekly_official_holidays.csv`
- Optional: `data/external/holidays/persian_holidays.csv`

## Configuration (.env)
The app reads `.env` automatically. See `.env.example` for a template.

Required keys:
- `PHARMACYSTORE_SQL_SERVER`
- `PHARMACYSTORE_SQL_DATABASE`
- `PHARMACYSTORE_SQL_USERNAME`
- `PHARMACYSTORE_SQL_PASSWORD`

Defaults (when using Docker restore):
- `PHARMACYSTORE_SQL_SERVER=localhost,14333` (note the comma for port)
- `PHARMACYSTORE_SQL_DATABASE=PharmacyStore`
- `PHARMACYSTORE_SQL_USERNAME=sa`
- `PHARMACYSTORE_SQL_DRIVER=ODBC Driver 18 for SQL Server`
- `PHARMACYSTORE_SQL_ENCRYPT=false`
- `PHARMACYSTORE_SQL_TRUST_CERT=true`

Password handling:
- Restore scripts do not write the password to `.env`.
- `run_all.py` passes it via environment variables.
- If you run the pipeline manually, export `PHARMACYSTORE_SQL_PASSWORD` yourself or add it to `.env`.

## Repository Layout
- `src/pharmacystore/`: core package
- `scripts/`: runnable entry points
- `restore_mssql_docker.sh`: Linux Docker restore
- `restore_mssql_docker.ps1`: Windows Docker restore
- `data/external/`: backup and reference data
- `data/processed/`: generated outputs (gitignored)
- `artifacts/`: trained models and artifacts (gitignored)
- `artifacts/metrics/`: per-run metrics exports (gitignored)

## Data Sources
SQL Server tables used by the pipeline:
- `dr_Drugs`: drug metadata and ATC codes
- `dr_PrescriptionDetailLog`: prescription line items (PrID, DrugId, PacketQuantity, LogDateTime, SalePrice)
- `dr_FactorDetail`: factor/price details for optional merges

## Feature Engineering (Weekly)
- Weekly aggregation: `UPW = nunique(PrID)` per `DrugId` and `week_start` (Monday bucket).
- Calendar features: week-of-year, month, quarter, start/end flags.
- Holiday features: weekly official holiday count (if data is provided).
- ATC classification: normalized ATC codes and grouping.
- Sales/price categories: computed from a training subset to avoid leakage.
- Rolling features: lagged UPW, rolling means/stds, trend and z-score features.

## Forecast Target & Evaluation
| Component | Standard |
| --- | --- |
| Target (UPW) | Weekly unique prescriptions per drug: `UPW = nunique(PrID)` aggregated by `DrugId` and `week_start`, filtered to `SalePrice > 0`. |
| Time split | Train = oldest weeks, validation = middle weeks, test = newest weeks (70/15/15 by week). |
| Baselines | Naive (last week), Moving average (last 4 weeks), Seasonal naive (t-52 weeks, if available). |
| Metrics | MAE, RMSE, WAPE, sMAPE. |

## Run (Manual)
- Full pipeline (features + model training):
  `python -m pharmacystore.pipeline full`
- Build features and export weekly dataset:
  `python -m pharmacystore.pipeline run`
- Train the main XGBoost model:
  `python -m pharmacystore.pipeline train`
- Train the baseline XGBoost model:
  `python -m pharmacystore.pipeline train-baseline`
- Optional run tag to label artifacts:
  `python -m pharmacystore.pipeline train --run-tag exp1`

Legacy script entry points still work:
- `python scripts/run_pipeline.py`
- `python scripts/train_model.py`
- `python scripts/train_baseline.py`
- Scale sales categories:
  `python scripts/scale_sales.py`
- Preview ATC classification:
  `python scripts/classify_atc.py`

## Outputs
- `data/processed/weekly_features.csv`: weekly feature dataset.
- `data/processed/upw_valid_predictions.csv`: validation predictions.
- `data/processed/upw_walkforward_predictions.csv`: walk-forward backtest (optional).
- `data/processed/upw_fold_drift.csv`: drift diagnostics (train vs valid).
- `artifacts/models/upw_xgb_model.json`: latest trained model.
- `artifacts/runs/<run_id>/`: run-scoped configs, metrics, predictions, diagnostics.

## Reproducibility
- Fixed seeds for numpy/random/xgboost via `PHARMACYSTORE_RANDOM_SEED`.
- Each run writes artifacts to `artifacts/runs/<run_id>/` with:
  `config.json`, `train_params.json`, `metrics.json`, `data_manifest.json`,
  plus per-run predictions and model files.
- Metrics are also exported to `artifacts/metrics/<run_id>/metrics.json`.
- `data_manifest.json` records dataset hash and train/test week ranges.

## Troubleshooting
- pip SSL errors during install: check system time and CA certificates; try
  `python -m pip install -U pip setuptools wheel`.
- "Cannot open database ... (4060)": verify the Docker container is running and
  the database name in `.env` matches the restored DB.
- "No .bak file found": ensure exactly one `.bak` exists in `data/external/`.
- Missing ODBC driver: install "ODBC Driver 18 for SQL Server" on the host.

## Data Privacy
`data/processed/` and `artifacts/` are ignored by Git to keep sensitive outputs and model files out of the repository.
