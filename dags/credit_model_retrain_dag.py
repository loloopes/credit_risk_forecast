"""Manual / GitHub trigger: retrain and register model (no NannyML monitoring)."""

from datetime import datetime

from airflow.decorators import dag, task

from mlflow_training_env import get_training_env


@dag(
    dag_id="credit_model_retrain_from_github_actions",
    description="Retrain and register the credit scoring model (trigger from GitHub Actions).",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "training", "manual", "ci", "github"],
)
def credit_model_retrain_from_github_actions():
    @task.bash(task_id="train_and_register_model", env=get_training_env())
    def train_and_register_model() -> str:
        return "python /opt/airflow/dags/pipelines/train_model.py"

    train_and_register_model()


dag = credit_model_retrain_from_github_actions()
