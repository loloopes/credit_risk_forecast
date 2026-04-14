"""Retrain DAG dedicated to model decay / data drift events."""

from datetime import datetime

from airflow.decorators import dag, task

from mlflow_training_env import get_training_env


@dag(
    dag_id="credit_model_retrain_on_decay_or_drift",
    description="Retrain and register model when drift/decay monitoring triggers it.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "training", "monitoring", "drift", "decay"],
)
def credit_model_retrain_on_decay_or_drift():
    @task.bash(task_id="train_and_register_model", env=get_training_env())
    def train_and_register_model() -> str:
        return "python /opt/airflow/dags/pipelines/train_model.py"

    train_and_register_model()


dag = credit_model_retrain_on_decay_or_drift()
