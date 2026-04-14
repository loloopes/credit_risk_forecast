import os
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd
from sklearn.metrics import roc_auc_score


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


def _save_report_to_uri(report_df: pd.DataFrame, output_uri: str) -> None:
    if _is_s3_path(output_uri):
        bucket, key = _parse_s3_uri(output_uri)
        s3_client = _build_s3_client()
        csv_buffer = StringIO()
        report_df.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue().encode("utf-8"))
        return

    output_path = Path(output_uri)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)


def _run_nannyml_drift(reference_df: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    import nannyml as nml

    timestamp_col = os.getenv("MONITOR_TIMESTAMP_COLUMN")
    feature_cols = [c for c in reference_df.columns if c not in {timestamp_col}]
    if not feature_cols:
        raise ValueError("No feature columns available for monitoring.")

    kwargs = {"column_names": feature_cols, "chunk_size": 500}
    if timestamp_col and timestamp_col in reference_df.columns and timestamp_col in analysis_df.columns:
        kwargs["timestamp_column_name"] = timestamp_col

    try:
        calculator = nml.DataReconstructionDriftCalculator(**kwargs)
    except AttributeError:
        calculator = nml.UnivariateDriftCalculator(**kwargs)

    calculator.fit(reference_df)
    result = calculator.calculate(analysis_df)
    result_df = result.to_df()
    return result_df


def _compute_model_decay(
    reference_df: pd.DataFrame, analysis_df: pd.DataFrame
) -> dict:
    target_col = os.getenv("MONITOR_TARGET_COLUMN", "target")
    prediction_col = os.getenv("MONITOR_PREDICTION_COLUMN", "prediction_proba")
    min_auc_drop = float(os.getenv("MODEL_DECAY_MIN_AUC_DROP", "0.03"))

    required_cols = {target_col, prediction_col}
    if not required_cols.issubset(reference_df.columns) or not required_cols.issubset(analysis_df.columns):
        return {
            "decay_checked": False,
            "decay_detected": False,
            "reference_auc": None,
            "analysis_auc": None,
            "auc_drop": None,
            "min_auc_drop": min_auc_drop,
        }

    try:
        reference_auc = float(roc_auc_score(reference_df[target_col], reference_df[prediction_col]))
        analysis_auc = float(roc_auc_score(analysis_df[target_col], analysis_df[prediction_col]))
    except Exception:
        return {
            "decay_checked": False,
            "decay_detected": False,
            "reference_auc": None,
            "analysis_auc": None,
            "auc_drop": None,
            "min_auc_drop": min_auc_drop,
        }

    auc_drop = reference_auc - analysis_auc
    decay_detected = auc_drop >= min_auc_drop
    return {
        "decay_checked": True,
        "decay_detected": decay_detected,
        "reference_auc": reference_auc,
        "analysis_auc": analysis_auc,
        "auc_drop": auc_drop,
        "min_auc_drop": min_auc_drop,
    }


def run_monitoring() -> dict:
    reference_path = os.getenv("REFERENCE_DATA_PATH", "s3://mlflow/data/reference.csv")
    analysis_path = os.getenv("ANALYSIS_DATA_PATH", "s3://mlflow/data/analysis.csv")
    output_path = os.getenv("MONITOR_OUTPUT_PATH", "s3://mlflow/artifacts/nannyml_drift_report.csv")
    alert_threshold = float(os.getenv("DRIFT_ALERT_THRESHOLD", "0.7"))

    reference_df = _load_dataset_from_uri(reference_path)
    analysis_df = _load_dataset_from_uri(analysis_path)

    drift_report_df = _run_nannyml_drift(reference_df, analysis_df)
    _save_report_to_uri(drift_report_df, output_path)

    should_retrain = False
    drift_detected = False
    alert_count = 0
    alert_rate = 0.0
    raw_alert_count = drift_report_df.get("alert")
    if raw_alert_count is not None:
        alert_series = raw_alert_count.fillna(False).astype(bool)
        alert_count = int(alert_series.sum())
        alert_rate = float(alert_series.mean())
        print(f"NannyML alerts found: {alert_count}", flush=True)
        drift_detected = alert_rate >= alert_threshold

    decay_result = _compute_model_decay(reference_df, analysis_df)
    if decay_result["decay_checked"]:
        print(
            "Model decay check: "
            f"reference_auc={decay_result['reference_auc']:.4f}, "
            f"analysis_auc={decay_result['analysis_auc']:.4f}, "
            f"auc_drop={decay_result['auc_drop']:.4f}",
            flush=True,
        )
    else:
        print(
            "Model decay check skipped: missing target/prediction columns.",
            flush=True,
        )

    should_retrain = drift_detected or decay_result["decay_detected"]

    print(f"NannyML monitoring completed. Report saved to: {output_path}", flush=True)
    return {
        "should_retrain": should_retrain,
        "drift_detected": drift_detected,
        "decay_detected": decay_result["decay_detected"],
        "decay_checked": decay_result["decay_checked"],
        "alert_count": alert_count,
        "alert_rate": alert_rate,
        "alert_threshold": alert_threshold,
        "reference_auc": decay_result["reference_auc"],
        "analysis_auc": decay_result["analysis_auc"],
        "auc_drop": decay_result["auc_drop"],
        "min_auc_drop": decay_result["min_auc_drop"],
        "output_path": output_path,
    }


def main() -> None:
    result = run_monitoring()
    if result["should_retrain"]:
        raise RuntimeError(
            "Model/data decay detected by NannyML "
            f"(alert_rate={result['alert_rate']:.3f}, threshold={result['alert_threshold']})."
        )


if __name__ == "__main__":
    main()
