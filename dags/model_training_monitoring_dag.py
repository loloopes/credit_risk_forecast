from datetime import datetime
import json
import os

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.docker.operators.docker import DockerOperator

def _monitoring_env() -> dict:
    return {
        "REFERENCE_DATA_PATH": os.getenv("REFERENCE_DATA_PATH", "s3://mlflow/data"),
        "ANALYSIS_DATA_PATH": os.getenv("ANALYSIS_DATA_PATH", "s3://mlflow/actuals"),
        "REFERENCE_SUFFIX": os.getenv("REFERENCE_SUFFIX", ".parquet"),
        "ANALYSIS_SUFFIX": os.getenv("ANALYSIS_SUFFIX", "_new.parquet"),
        "MONITOR_JOIN_ID_CLIENTE": os.getenv("MONITOR_JOIN_ID_CLIENTE", "id_cliente"),
        "MONITOR_JOIN_ID_CONTRATO": os.getenv("MONITOR_JOIN_ID_CONTRATO", "id_contrato"),
        "MONITOR_OUTPUT_PATH": os.getenv(
            "MONITOR_OUTPUT_PATH", "s3://mlflow/artifacts/nannyml_drift_report.csv"
        ),
        "DRIFT_ALERT_THRESHOLD": os.getenv("DRIFT_ALERT_THRESHOLD", "0.7"),
        "MONITOR_TIMESTAMP_COLUMN": os.getenv("MONITOR_TIMESTAMP_COLUMN", "data_decisao"),
        "MONITOR_TARGET_COLUMN": os.getenv("MONITOR_TARGET_COLUMN", "target"),
        "MONITOR_PREDICTION_COLUMN": os.getenv("MONITOR_PREDICTION_COLUMN", "prediction_proba"),
        "MODEL_DECAY_MIN_AUC_DROP": os.getenv("MODEL_DECAY_MIN_AUC_DROP", "0.03"),
        "PREDICTION_API_URL": os.getenv("PREDICTION_API_URL", "http://localhost:8000/predict"),
        "REQUEST_TIMEOUT_SECONDS": os.getenv("REQUEST_TIMEOUT_SECONDS", "15"),
        "MONITOR_MAX_ROWS": os.getenv("MONITOR_MAX_ROWS", "0"),
        "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }


@dag(
    dag_id="credit_model_training_and_nannyml_monitoring",
    description="Check data drift and model decay daily, retrain only when detected.",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["ml", "training", "monitoring", "nannyml"],
)
def credit_model_training_and_nannyml_monitoring():
    monitor_with_nannyml_api = DockerOperator(
        task_id="monitor_with_nannyml_api",
        image=os.getenv("NANNYML_EVAL_IMAGE", "datarisk/nannyml-evaluator:latest"),
        user="root",
        container_name="nannyml",
        api_version="auto",
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        network_mode=os.getenv("NANNYML_DOCKER_NETWORK_MODE", "container:credit_scoring_api"),
        tty=True,
        xcom_all=False,
        mount_tmp_dir=False,
        environment=_monitoring_env(),
        do_xcom_push=True,
        command="python /app/evaluate.py",
    )

    @task.branch(task_id="decide_retraining")
    def decide_retraining(monitor_output: str) -> str:
        try:
            output_lines = [line.strip() for line in str(monitor_output).splitlines() if line.strip()]
            result = json.loads(output_lines[-1]) if output_lines else {"should_retrain": False}
        except Exception:
            result = {"should_retrain": False}
        if result.get("should_retrain"):
            return "trigger_retrain_on_decay_or_drift"
        return "skip_retraining"

    trigger_retrain_on_decay_or_drift = TriggerDagRunOperator(
        task_id="trigger_retrain_on_decay_or_drift",
        trigger_dag_id="credit_model_retrain_dag",
        wait_for_completion=False,
    )

    @task(task_id="skip_retraining")
    def skip_retraining():
        print("No drift above threshold. Retraining skipped.", flush=True)

    decision = decide_retraining(monitor_with_nannyml_api.output)
    skip_task = skip_retraining()

    decision >> [trigger_retrain_on_decay_or_drift, skip_task]

dag = credit_model_training_and_nannyml_monitoring()
