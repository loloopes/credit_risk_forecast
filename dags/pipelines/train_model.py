import os
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import boto3
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _is_s3_path(path_value: str) -> bool:
    return path_value.startswith("s3://")


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _build_s3_client():
    endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
    region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client("s3", endpoint_url=endpoint_url, region_name=region_name)


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() == ".csv":
        return pd.read_csv(dataset_path)
    if dataset_path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(dataset_path)

    raise ValueError("Unsupported dataset format. Use .csv or .parquet.")


def _load_dataset_from_uri(dataset_uri: str) -> pd.DataFrame:
    if _is_s3_path(dataset_uri):
        bucket, key = _parse_s3_uri(dataset_uri)
        s3_client = _build_s3_client()
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()

        if dataset_uri.lower().endswith(".csv"):
            return pd.read_csv(BytesIO(data))
        if dataset_uri.lower().endswith((".parquet", ".pq")):
            return pd.read_parquet(BytesIO(data))
        raise ValueError("Unsupported dataset format. Use .csv or .parquet.")

    return _load_dataset(Path(dataset_uri))


def _load_if_provided(dataset_uri: str | None) -> pd.DataFrame | None:
    if dataset_uri and dataset_uri.strip():
        return _load_dataset_from_uri(dataset_uri)
    return None


def _build_training_dataset_from_raw_sources() -> pd.DataFrame:
    id_cliente_col = os.getenv("ID_CLIENTE_COLUMN", "id_cliente")
    id_contrato_col = os.getenv("ID_CONTRATO_COLUMN", "id_contrato")

    submissao_path = os.getenv("RAW_SUBMISSAO_PATH", "s3://mlflow/data/base_submissao.parquet")
    cadastral_path = os.getenv("RAW_CADASTRAL_PATH", "s3://mlflow/data/base_cadastral.parquet")
    emprestimos_path = os.getenv("RAW_EMPRESTIMOS_PATH", "s3://mlflow/data/historico_emprestimos.parquet")
    parcelas_path = os.getenv("RAW_PARCELAS_PATH", "s3://mlflow/data/historico_parcelas.parquet")

    base_df = _load_dataset_from_uri(submissao_path)
    cadastral_df = _load_if_provided(cadastral_path)
    emprestimos_df = _load_if_provided(emprestimos_path)
    parcelas_df = _load_if_provided(parcelas_path)

    if cadastral_df is not None and id_cliente_col in base_df.columns and id_cliente_col in cadastral_df.columns:
        cadastral_cols = [c for c in cadastral_df.columns if c != id_cliente_col and c not in base_df.columns]
        base_df = base_df.merge(
            cadastral_df[[id_cliente_col] + cadastral_cols],
            on=id_cliente_col,
            how="left",
        )

    if emprestimos_df is not None and id_contrato_col in base_df.columns and id_contrato_col in emprestimos_df.columns:
        emprestimos_cols = [c for c in emprestimos_df.columns if c != id_contrato_col and c not in base_df.columns]
        base_df = base_df.merge(
            emprestimos_df[[id_contrato_col] + emprestimos_cols],
            on=id_contrato_col,
            how="left",
        )

    if parcelas_df is not None and id_contrato_col in base_df.columns and id_contrato_col in parcelas_df.columns:
        numeric_cols = parcelas_df.select_dtypes(include=["number"]).columns.tolist()
        agg_cols = [c for c in numeric_cols if c != id_contrato_col]
        if agg_cols:
            agg_dict = {c: "mean" for c in agg_cols}
            parcelas_agg = parcelas_df.groupby(id_contrato_col, as_index=False).agg(agg_dict)
            parcelas_agg = parcelas_agg.rename(columns={c: f"{c}_parcelas_mean" for c in agg_cols})
            base_df = base_df.merge(parcelas_agg, on=id_contrato_col, how="left")

    return base_df


def _resolve_training_dataset() -> tuple[pd.DataFrame, str]:
    dataset_uri = os.getenv("TRAIN_DATA_PATH", "").strip()
    if dataset_uri:
        return _load_dataset_from_uri(dataset_uri), dataset_uri

    df = _build_training_dataset_from_raw_sources()
    return df, "raw_sources(base_submissao+base_cadastral+historico_emprestimos+historico_parcelas)"


def main() -> None:
    dataset_uri = os.getenv("TRAIN_DATA_PATH", "").strip()
    target_col = os.getenv("TARGET_COLUMN", "target")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit_risk_training")
    model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "credit_model_pipeline_v2")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    df, source_used = _resolve_training_dataset()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(random_state=42, n_estimators=300)),
        ]
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_valid_proba)

        mlflow.log_param("train_data_path", source_used if source_used else dataset_uri)
        mlflow.log_param("target_column", target_col)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_metric("valid_auc", auc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="credit_model_pipeline_v2",
            registered_model_name=model_name,
        )

        print(f"Training finished. Validation AUC: {auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
