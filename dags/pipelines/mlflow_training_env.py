"""Shared Airflow task env vars for MLflow model training (train_model.py)."""

import os


def get_training_env() -> dict:
    return {
        "TRAIN_DATA_PATH": os.getenv("TRAIN_DATA_PATH", ""),
        "RAW_SUBMISSAO_PATH": os.getenv("RAW_SUBMISSAO_PATH", "s3://mlflow/data/base_submissao.parquet"),
        "RAW_CADASTRAL_PATH": os.getenv("RAW_CADASTRAL_PATH", "s3://mlflow/data/base_cadastral.parquet"),
        "RAW_EMPRESTIMOS_PATH": os.getenv(
            "RAW_EMPRESTIMOS_PATH", "s3://mlflow/data/historico_emprestimos.parquet"
        ),
        "RAW_PARCELAS_PATH": os.getenv("RAW_PARCELAS_PATH", "s3://mlflow/data/historico_parcelas.parquet"),
        "ID_CLIENTE_COLUMN": os.getenv("ID_CLIENTE_COLUMN", "id_cliente"),
        "ID_CONTRATO_COLUMN": os.getenv("ID_CONTRATO_COLUMN", "id_contrato"),
        "TARGET_COLUMN": os.getenv("TARGET_COLUMN", "target"),
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        "MLFLOW_EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME", "credit_risk_training"),
        "MLFLOW_REGISTERED_MODEL_NAME": os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME", "credit_model_pipeline_v2"
        ),
        "LGBM_CLASSIFIER_PARAMS_PATH": os.getenv("LGBM_CLASSIFIER_PARAMS_PATH", ""),
        "LGBM_CLASSIFIER_PARAMS_JSON": os.getenv("LGBM_CLASSIFIER_PARAMS_JSON", ""),
        "MLFLOW_PROMOTION_GATE_DISABLED": os.getenv("MLFLOW_PROMOTION_GATE_DISABLED", ""),
        "MLFLOW_GATE_MIN_VALID_AUC": os.getenv("MLFLOW_GATE_MIN_VALID_AUC", ""),
        "MLFLOW_GATE_MAX_AUC_REGRESSION": os.getenv("MLFLOW_GATE_MAX_AUC_REGRESSION", ""),
        "MLFLOW_GATE_REQUIRE_IMPROVEMENT": os.getenv("MLFLOW_GATE_REQUIRE_IMPROVEMENT", ""),
        "MLFLOW_EXIT_ON_GATE_FAILURE": os.getenv("MLFLOW_EXIT_ON_GATE_FAILURE", ""),
        "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }
