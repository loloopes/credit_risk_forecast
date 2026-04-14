# Credit Risk Project

* This repository contains a credit risk data science project. It follows a practical, business-oriented approach to building data solutions for credit intelligence and risk decision-making.

## Project Scope

* Build a realistic credit risk modeling workflow
* Combine technical quality with business applicability
* Demonstrate clear reasoning, structured problem-solving, and well-justified modeling choices


## Data Assets
* Inside the data/ directory, you will find:

* Four datasets used in analysis and modeling dicionario_dados.csv with a complete description of all variables

* This project prioritizes:

* Clear problem framing
* Coherent analytical structure
* Well-justified decisions
* Practical methods that deliver business value
* The goal is not complexity for its own sake, but effective and explainable credit risk analytics.

## Airflow + NannyML Pipeline

This repository now includes a baseline training and monitoring pipeline:

- `dags/pipelines/train_model.py`: trains a LightGBM pipeline, logs metrics, and registers the model in MLflow.
- `dags/pipelines/monitor_model_decay.py`: runs NannyML drift monitoring comparing reference vs analysis data.
- `dags/model_training_monitoring_dag.py`: runs monitoring first and retrains only when drift is detected.

Flow logic:

- Airflow runs NannyML monitoring first (via `DockerOperator` container).
- If drift/decay is detected above `DRIFT_ALERT_THRESHOLD`, retraining is triggered automatically.
- If no drift is detected, retraining is skipped.
- Schedule is daily (`@daily`), so checks run every day.

Model decay check (optional but enabled when columns exist):

- Uses `MONITOR_TARGET_COLUMN` (default `target`) and `MONITOR_PREDICTION_COLUMN` (default `prediction_proba`).
- Compares reference AUC vs analysis AUC.
- Triggers retraining when `reference_auc - analysis_auc >= MODEL_DECAY_MIN_AUC_DROP`.

### Expected input files

Place files in MinIO (recommended) or local mounted directory:

- MinIO:
  - `s3://mlflow/data/base_submissao.parquet`
  - `s3://mlflow/data/base_cadastral.parquet`
  - `s3://mlflow/data/historico_emprestimos.parquet`
  - `s3://mlflow/data/historico_parcelas.parquet`
- Local fallback:
  - `dags/data/base_submissao.parquet`
  - `dags/data/base_cadastral.parquet`
  - `dags/data/historico_emprestimos.parquet`
  - `dags/data/historico_parcelas.parquet`

By default, training builds a dataset from those raw sources using `id_cliente` and `id_contrato`.
If you provide `TRAIN_DATA_PATH`, that file is used directly instead.

`TARGET_COLUMN` must exist in the final training dataset (from merged raw files or explicit train file).

NannyML output report is written to:

- `s3://mlflow/artifacts/nannyml_drift_report.csv` (default)

Notes on monitoring:

- Drift check runs every day from `REFERENCE_DATA_PATH` (old baseline) and `ANALYSIS_DATA_PATH` (actuals).
- Default mode expects folder-style prefixes:
  - old: `s3://mlflow/data` with files like `base_submissao.parquet`
  - actuals: `s3://mlflow/actuals` with files like `base_submissao_new.parquet`
- The DockerOperator container merges the four raw sources by `id_cliente`/`id_contrato` before running NannyML.
- Analysis rows are scored by calling the prediction API (`PREDICTION_API_URL`) for each record.
- DockerOperator runs with `NANNYML_DOCKER_NETWORK_MODE=container:credit_scoring_api` by default, so API calls use `http://localhost:8000/predict` inside that network namespace.
- Model decay check requires actual targets + predictions (`MONITOR_TARGET_COLUMN`, `MONITOR_PREDICTION_COLUMN`).
- If actuals are not available yet, decay check is skipped and drift-only logic is applied.
- Build the NannyML evaluator image once before running the DAG:
  - `docker build -t datarisk/nannyml-evaluator:latest -f docker/nannyml-evaluator/Dockerfile docker/nannyml-evaluator`
- If your actuals schema is missing API-required fields, set fallback payload values with `API_PAYLOAD_DEFAULTS_JSON` in `.env`.

### Running with Airflow

1. Copy `env-sample` to `.env` and adjust values.
2. Ensure Airflow installs extra python libs using `_PIP_ADDITIONAL_REQUIREMENTS`.
3. Start Airflow with:
   - `docker compose -f docker-compose.airflow.yml up -d`
4. In Airflow UI, enable DAG:
   - `credit_model_training_and_nannyml_monitoring`
5. Optional manual run:
   - `docker compose -f docker-compose.airflow.yml exec airflow-apiserver airflow dags trigger credit_model_training_and_nannyml_monitoring`
