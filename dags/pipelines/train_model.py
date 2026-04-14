import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import boto3
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
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


def _derive_targets_from_parcelas(hp: pd.DataFrame) -> pd.DataFrame:
    """Match model/lgbm_test.ipynb: FPD, EVER30MOB03, OVER60MOB06 at id_contrato grain."""
    id_contrato_col = os.getenv("ID_CONTRATO_COLUMN", "id_contrato")
    needed = {
        id_contrato_col,
        "numero_parcela",
        "data_real_pagamento",
        "data_prevista_pagamento",
    }
    missing = needed - set(hp.columns)
    if missing:
        raise ValueError(f"historico_parcelas missing columns required for targets: {sorted(missing)}")

    hp = hp.copy()
    hp["data_real_pagamento"] = pd.to_datetime(hp["data_real_pagamento"], errors="coerce")
    hp["data_prevista_pagamento"] = pd.to_datetime(hp["data_prevista_pagamento"], errors="coerce")
    hp["atraso"] = (hp["data_real_pagamento"] - hp["data_prevista_pagamento"]).dt.days
    hp["atraso"] = hp["atraso"].fillna(999)

    mob03 = hp[hp["numero_parcela"] <= 3]
    ever30 = (mob03.groupby(id_contrato_col, sort=False)["atraso"].max() > 30).astype(int)
    ever30 = ever30.reset_index()
    ever30.columns = [id_contrato_col, "target_ever30mob03"]

    mob06 = hp[hp["numero_parcela"] <= 6]
    over60 = mob06.groupby(id_contrato_col, sort=False)["atraso"].apply(lambda x: x[x > 0].sum()) > 60
    over60 = over60.astype(int).reset_index()
    over60.columns = [id_contrato_col, "target_over60mob06"]

    fpd = hp[hp["numero_parcela"] == 1].copy()
    fpd["atraso_fpd"] = (fpd["data_real_pagamento"] - fpd["data_prevista_pagamento"]).dt.days
    fpd["target_fpd"] = np.where((fpd["atraso_fpd"] > 1) | fpd["atraso_fpd"].isna(), 1, 0)
    fpd_small = fpd[[id_contrato_col, "target_fpd"]].drop_duplicates(subset=[id_contrato_col])

    out = ever30.merge(over60, on=id_contrato_col, how="outer")
    out = out.merge(fpd_small, on=id_contrato_col, how="outer")
    return out.fillna(0)


def _build_training_dataset_from_raw_sources() -> pd.DataFrame:
    """
    Grain is one row per contrato (see dicionario_dados.csv: base_submissao has no id_contrato).
    Labels come from historico_parcelas; features from historico_emprestimos + base_cadastral (+ optional base_submissao).
    """
    id_cliente_col = os.getenv("ID_CLIENTE_COLUMN", "id_cliente")
    id_contrato_col = os.getenv("ID_CONTRATO_COLUMN", "id_contrato")

    submissao_path = os.getenv("RAW_SUBMISSAO_PATH", "s3://mlflow/data/base_submissao.parquet")
    cadastral_path = os.getenv("RAW_CADASTRAL_PATH", "s3://mlflow/data/base_cadastral.parquet")
    emprestimos_path = os.getenv("RAW_EMPRESTIMOS_PATH", "s3://mlflow/data/historico_emprestimos.parquet")
    parcelas_path = os.getenv("RAW_PARCELAS_PATH", "s3://mlflow/data/historico_parcelas.parquet")

    emprestimos_df = _load_if_provided(emprestimos_path)
    parcelas_df = _load_if_provided(parcelas_path)
    if emprestimos_df is None or emprestimos_df.empty:
        raise ValueError(
            "RAW_EMPRESTIMOS_PATH must load a non-empty historico_emprestimos dataset when TRAIN_DATA_PATH is unset."
        )
    if parcelas_df is None or parcelas_df.empty:
        raise ValueError(
            "RAW_PARCELAS_PATH must load a non-empty historico_parcelas dataset when TRAIN_DATA_PATH is unset."
        )
    if id_contrato_col not in emprestimos_df.columns:
        raise ValueError(f"historico_emprestimos missing {id_contrato_col!r}.")

    targets_df = _derive_targets_from_parcelas(parcelas_df)
    df = emprestimos_df.merge(targets_df, on=id_contrato_col, how="inner")

    cadastral_df = _load_if_provided(cadastral_path)
    if cadastral_df is not None and id_cliente_col in df.columns and id_cliente_col in cadastral_df.columns:
        ccols = [c for c in cadastral_df.columns if c != id_cliente_col]
        df = df.merge(cadastral_df[[id_cliente_col] + ccols], on=id_cliente_col, how="left")

    submissao_df = _load_if_provided(submissao_path)
    if submissao_df is not None and id_cliente_col in df.columns and id_cliente_col in submissao_df.columns:
        sub_dup = submissao_df.drop_duplicates(subset=[id_cliente_col], keep="last").copy()
        renames = {
            c: f"{c}_submissao"
            for c in sub_dup.columns
            if c != id_cliente_col and c in df.columns
        }
        if renames:
            sub_dup = sub_dup.rename(columns=renames)
        df = df.merge(sub_dup, on=id_cliente_col, how="left")

    return df


def _resolve_training_dataset() -> tuple[pd.DataFrame, str]:
    dataset_uri = os.getenv("TRAIN_DATA_PATH", "").strip()
    if dataset_uri:
        return _load_dataset_from_uri(dataset_uri), dataset_uri

    df = _build_training_dataset_from_raw_sources()
    return df, "raw_sources(historico_emprestimos+historico_parcelas_targets+base_cadastral[+base_submissao])"


# Order matches common names from project notebooks / engineered exports.
_TARGET_COLUMN_FALLBACKS: tuple[str, ...] = (
    "target",
    "target_over60mob06",
    "target_ever30mob03",
    "target_fpd",
    "inadimplente",
    "bad",
)


def _resolve_target_column(df: pd.DataFrame) -> str:
    preferred = (os.getenv("TARGET_COLUMN") or "").strip()
    if preferred and preferred in df.columns:
        return preferred

    for name in _TARGET_COLUMN_FALLBACKS:
        if name in df.columns:
            if preferred:
                print(
                    f"TARGET_COLUMN={preferred!r} not in dataset; using '{name}'.",
                    flush=True,
                )
            return name

    # Single column named like target_*
    candidates = [c for c in df.columns if str(c).startswith("target")]
    if len(candidates) == 1:
        c = candidates[0]
        if preferred:
            print(
                f"TARGET_COLUMN={preferred!r} not in dataset; using sole target-like column '{c}'.",
                flush=True,
            )
        return c

    cols_preview = sorted(df.columns.astype(str).tolist())[:80]
    raise ValueError(
        f"Target column not found. Set TARGET_COLUMN to one of the dataset columns. "
        f"Tried TARGET_COLUMN={preferred!r} and fallbacks {_TARGET_COLUMN_FALLBACKS}. "
        f"Columns (first 80): {cols_preview}"
    )


def main() -> None:
    dataset_uri = os.getenv("TRAIN_DATA_PATH", "").strip()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit_Prisk_training")
    model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "credit_model_pipeline_v2")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    df, source_used = _resolve_training_dataset()
    target_col = _resolve_target_column(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    strat = y if y.nunique() > 1 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
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

    with mlflow.start_run() as run:
        mlflow.set_tag("training_started_at_utc", datetime.now(timezone.utc).isoformat())
        model.fit(X_train, y_train)
        y_valid_pred = model.predict(X_valid)
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_valid_proba)

        labels = np.unique(np.concatenate([np.asarray(y_valid), np.asarray(y_valid_pred)]))
        precisions = precision_score(
            y_valid, y_valid_pred, labels=labels, average=None, zero_division=0
        )
        recalls = recall_score(
            y_valid, y_valid_pred, labels=labels, average=None, zero_division=0
        )
        f1s = f1_score(y_valid, y_valid_pred, labels=labels, average=None, zero_division=0)

        mlflow.log_param("train_data_path", source_used if source_used else dataset_uri)
        mlflow.log_param("target_column", target_col)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_metric("valid_auc", auc)
        for i, lbl in enumerate(labels):
            mlflow.log_metric(f"valid_precision_class_{lbl}", float(precisions[i]))
            mlflow.log_metric(f"valid_recall_class_{lbl}", float(recalls[i]))
            mlflow.log_metric(f"valid_f1_class_{lbl}", float(f1s[i]))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        mlflow.set_tag("registered_model_name", model_name)
        mlflow.set_tag("registered_model_version", model_version.version)
        mlflow.set_tag("logged_model_uri", model_info.model_uri)

        print(
            f"Training finished. Validation AUC: {auc:.4f}. "
            f"Persisted to MLflow run {run.info.run_id} and model {model_name} v{model_version.version}",
            flush=True,
        )


if __name__ == "__main__":
    main()
