import asyncio
import json
import os
import socket
import threading
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Iterable, Optional
from uuid import uuid4

import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# MLflow (and similar) may define Pydantic fields named `model_name`; silence that noise.
warnings.filterwarnings(
    "ignore",
    message=r'Field "model_name".*protected namespace',
    category=UserWarning,
)

_spark_lock = threading.Lock()
_spark_session = None
_prediction_ddl_done = False
_prediction_hms_registered = False

# ==========================================
# Carregamento do modelo via MLflow Model Registry
# ==========================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI")
RUN_ID = os.getenv("RUN_ID")
MLFLOW_MODEL_ARTIFACT_PATH = os.getenv("MLFLOW_MODEL_ARTIFACT_PATH", "credit_model_pipeline_v2")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "credit_model_pipeline_v2")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "latest")
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "false").lower() in {"1", "true", "yes"}
PREDICTION_LOG_ENABLED = os.getenv("PREDICTION_LOG_ENABLED", "true").lower() in {"1", "true", "yes"}
PREDICTION_LOG_STRICT = os.getenv("PREDICTION_LOG_STRICT", "true").lower() in {"1", "true", "yes"}

# Spark cluster (driver runs in this process; executors on existing workers)
SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
SPARK_HIVE_METASTORE_URIS = os.getenv(
    "SPARK_HIVE_METASTORE_URIS", "thrift://hive-metastore:9083"
)
SPARK_SQL_WAREHOUSE_DIR = os.getenv("SPARK_SQL_WAREHOUSE_DIR", "s3a://lakehouse/")
# Use container_name by default because this API and spark-cluster run in different compose projects.
SPARK_DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "credit-scoring-api")
SPARK_S3A_ENDPOINT = os.getenv("SPARK_S3A_ENDPOINT", "http://minio:9000")
SPARK_S3A_ACCESS_KEY = os.getenv(
    "SPARK_S3A_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID", "")
)
SPARK_S3A_SECRET_KEY = os.getenv(
    "SPARK_S3A_SECRET_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "")
)
SPARK_PYTHON_BIN = os.getenv("PYSPARK_PYTHON", "python3")
SPARK_DRIVER_PYTHON_BIN = os.getenv("PYSPARK_DRIVER_PYTHON", SPARK_PYTHON_BIN)
SPARK_EXTRA_CLASSPATH = os.getenv(
    "SPARK_EXTRA_CLASSPATH",
    "/opt/extra-jars/hadoop-aws-3.3.4.jar:"
    "/opt/extra-jars/aws-java-sdk-bundle-1.12.262.jar:"
    "/opt/extra-jars/woodstox-core-6.2.8.jar:"
    "/opt/extra-jars/stax2-api-4.2.1.jar:"
    "/opt/extra-jars/iceberg-spark-runtime-3.5_2.12-1.10.1.jar",
)
PREDICTION_LOG_DATABASE = os.getenv("PREDICTION_LOG_DATABASE", "forecast")
PREDICTION_LOG_TABLE = os.getenv("PREDICTION_LOG_TABLE", "prediction_events")
ICEBERG_MAIN_CATALOG = os.getenv("ICEBERG_MAIN_CATALOG", "iceberg")
ICEBERG_HMS_CATALOG = os.getenv("ICEBERG_HMS_CATALOG", "iceberg_hms")


def _empty_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _sql_literal(value: Optional[str]) -> str:
    if value is None:
        return "NULL"
    escaped = value.replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


def _resolve_spark_driver_host(raw_host: str) -> str:
    """
    Spark Standalone rejects underscores in spark:// host URLs.
    Resolve invalid hostnames to a routable IP before building SparkSession.
    """
    candidate = raw_host.strip()
    if not candidate:
        candidate = "127.0.0.1"
    if "_" not in candidate:
        return candidate
    try:
        return socket.gethostbyname(candidate)
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def _split_model_uris(raw: str) -> list[str]:
    # Allow multiple fallbacks, e.g.:
    # MLFLOW_MODEL_URI="models:/credit_risk_forecast/1,models:/credit_riks_forecast/1"
    parts: list[str] = []
    for chunk in raw.replace(";", ",").split(","):
        uri = chunk.strip()
        if uri:
            parts.append(uri)
    return parts


def _dedupe_preserve_order(uris: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for uri in uris:
        if uri in seen:
            continue
        seen.add(uri)
        out.append(uri)
    return out


def _iter_candidate_model_uris() -> list[str]:
    uris: list[str] = []

    explicit = _empty_to_none(MLFLOW_MODEL_URI)
    if explicit:
        uris.extend(_split_model_uris(explicit))

    run_id = _empty_to_none(RUN_ID)
    if run_id:
        artifact_path = _empty_to_none(MLFLOW_MODEL_ARTIFACT_PATH)
        if artifact_path:
            uris.append(f"runs:/{run_id}/{artifact_path}")

        # If training logged `mlflow.set_tag("logged_model_uri", ...)`, prefer it.
        try:
            from mlflow.tracking import MlflowClient

            run = MlflowClient().get_run(run_id)
            logged_uri = _empty_to_none(run.data.tags.get("logged_model_uri"))
            if logged_uri:
                uris.append(logged_uri)
        except Exception:
            pass

        # Common artifact_path values across this repo.
        for fallback in ("credit_model_pipeline_v2", "model"):
            if fallback != artifact_path:
                uris.append(f"runs:/{run_id}/{fallback}")

    model_name = _empty_to_none(MLFLOW_MODEL_NAME)
    model_stage = _empty_to_none(MLFLOW_MODEL_STAGE)
    if model_name and model_stage:
        uris.append(f"models:/{model_name}/{model_stage}")

    return _dedupe_preserve_order(uris)


def _stop_spark_session() -> None:
    global _spark_session, _prediction_ddl_done, _prediction_hms_registered
    with _spark_lock:
        if _spark_session is not None:
            try:
                _spark_session.stop()
            except Exception as e:
                print(f"Spark stop error: {e}", flush=True)
            _spark_session = None
            _prediction_ddl_done = False
            _prediction_hms_registered = False


def _ensure_spark_session_locked():
    """Create SparkSession if missing. Caller must hold ``_spark_lock``."""
    global _spark_session
    if _spark_session is not None:
        return _spark_session

    from pyspark.sql import SparkSession

    os.makedirs("/tmp/spark-local", exist_ok=True)
    extra_cp = SPARK_EXTRA_CLASSPATH.strip()
    spark_jars = ",".join([item for item in extra_cp.split(":") if item])
    driver_host = _resolve_spark_driver_host(SPARK_DRIVER_HOST)
    builder = (
        SparkSession.builder.appName("credit-scoring-api-lakehouse-log")
        .master(SPARK_MASTER_URL)
        .config("spark.submit.deployMode", "client")
        .config("spark.driver.host", driver_host)
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.sql.warehouse.dir", SPARK_SQL_WAREHOUSE_DIR)
        .config("spark.hadoop.hive.metastore.uris", SPARK_HIVE_METASTORE_URIS)
        .config("spark.sql.catalogImplementation", "hive")
        .config("spark.hadoop.fs.s3a.endpoint", SPARK_S3A_ENDPOINT)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.access.key", SPARK_S3A_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", SPARK_S3A_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.local.dir", "/tmp/spark-local")
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "512m"))
        .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", "512m"))
        .config("spark.pyspark.python", SPARK_PYTHON_BIN)
        .config("spark.pyspark.driver.python", SPARK_DRIVER_PYTHON_BIN)
        .config("spark.driver.userClassPathFirst", "true")
        .config("spark.executor.userClassPathFirst", "true")
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.iceberg.vectorization.enabled", "false")
        .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.iceberg.type", "hadoop")
        .config("spark.sql.catalog.iceberg.warehouse", SPARK_SQL_WAREHOUSE_DIR)
        .config("spark.sql.catalog.iceberg_hms", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.iceberg_hms.type", "hive")
        .config("spark.sql.catalog.iceberg_hms.uri", SPARK_HIVE_METASTORE_URIS)
        .config("spark.sql.catalog.iceberg_hms.warehouse", SPARK_SQL_WAREHOUSE_DIR)
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
    )
    if extra_cp:
        builder = (
            builder.config("spark.driver.extraClassPath", extra_cp)
            .config("spark.executor.extraClassPath", extra_cp)
            .config("spark.jars", spark_jars)
        )
    _spark_session = builder.getOrCreate()
    return _spark_session


def _latest_iceberg_metadata_file(spark, schema: str, table: str) -> str:
    warehouse_root = SPARK_SQL_WAREHOUSE_DIR.rstrip("/")
    metadata_dir = f"{warehouse_root}/{schema}/{table}/metadata"
    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    path = jvm.org.apache.hadoop.fs.Path(metadata_dir)
    fs = path.getFileSystem(hconf)
    if not fs.exists(path):
        raise RuntimeError(f"Iceberg metadata directory not found: {metadata_dir}")

    files = fs.listStatus(path)
    candidates: list[str] = []
    for file_status in files:
        file_name = file_status.getPath().getName()
        if file_name.endswith(".metadata.json"):
            candidates.append(file_name)
    if not candidates:
        raise RuntimeError(f"No Iceberg metadata file found in {metadata_dir}")

    latest = sorted(candidates)[-1]
    return f"{metadata_dir}/{latest}"


def _ensure_prediction_table_locked(spark) -> None:
    """Idempotent DDL. Caller must hold ``_spark_lock``."""
    global _prediction_ddl_done
    if _prediction_ddl_done:
        return
    db = PREDICTION_LOG_DATABASE
    # Keep both syntaxes for compatibility across catalog implementations.
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {ICEBERG_MAIN_CATALOG}.{db}")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {ICEBERG_MAIN_CATALOG}.{db}")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {ICEBERG_HMS_CATALOG}.{db}")
    spark.sql(
        f"CREATE NAMESPACE IF NOT EXISTS {ICEBERG_HMS_CATALOG}.{db} "
        f"LOCATION '{SPARK_SQL_WAREHOUSE_DIR.rstrip('/')}/{db}'"
    )
    _prediction_ddl_done = True


def _append_prediction_to_lakehouse(
    request_payload: dict,
    response_payload: dict,
    request_id: str,
) -> None:
    if not PREDICTION_LOG_ENABLED:
        return

    event_ts = datetime.now(timezone.utc).isoformat()
    model_name = _empty_to_none(MLFLOW_MODEL_NAME) or ""
    model_stage = _empty_to_none(MLFLOW_MODEL_STAGE) or ""
    request_json = json.dumps(request_payload, ensure_ascii=False)
    response_json = json.dumps(response_payload, ensure_ascii=False)

    global _prediction_hms_registered
    with _spark_lock:
        from pyspark.sql.utils import AnalysisException

        spark = _ensure_spark_session_locked()
        _ensure_prediction_table_locked(spark)
        table_name = f"{PREDICTION_LOG_DATABASE}.{PREDICTION_LOG_TABLE}"
        # Write through Spark (Iceberg HMS catalog) to persist in lakehouse.
        full_table_name = f"{ICEBERG_HMS_CATALOG}.{table_name}"
        event_df = spark.createDataFrame(
            [
                {
                    "event_id": request_id,
                    "event_ts": event_ts,
                    "model_name": model_name,
                    "model_stage": model_stage,
                    "request_json": request_json,
                    "response_json": response_json,
                }
            ]
        )
        writer = (
            event_df.writeTo(full_table_name)
            .tableProperty("format-version", "2")
            .tableProperty("write.format.default", "parquet")
        )
        try:
            writer.append()
        except AnalysisException:
            # Some HMS setups do not auto-create namespace via V2 writer path.
            spark.sql(
                f"CREATE DATABASE IF NOT EXISTS {ICEBERG_HMS_CATALOG}.{PREDICTION_LOG_DATABASE}"
            )
            spark.sql(
                f"CREATE NAMESPACE IF NOT EXISTS {ICEBERG_HMS_CATALOG}.{PREDICTION_LOG_DATABASE}"
            )
            spark.sql(
                f"CREATE TABLE IF NOT EXISTS {full_table_name} ("
                "event_id STRING, "
                "event_ts STRING, "
                "model_name STRING, "
                "model_stage STRING, "
                "request_json STRING, "
                "response_json STRING"
                ") USING iceberg"
            )
            writer.append()

        # Already writing directly to HMS catalog; no manual register needed.
        if not _prediction_hms_registered:
            _prediction_hms_registered = True


if SKIP_MODEL_LOAD:
    print("SKIP_MODEL_LOAD enabled. Starting API without loading model.", flush=True)
    model = None
else:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        candidates = _iter_candidate_model_uris()
        if not candidates:
            raise RuntimeError(
                "No model URI candidates resolved. Set MLFLOW_MODEL_URI, "
                "or RUN_ID (+ MLFLOW_MODEL_ARTIFACT_PATH), "
                "or MLFLOW_MODEL_NAME + MLFLOW_MODEL_STAGE."
            )

        last_error: Optional[Exception] = None
        for model_uri in candidates:
            print(f"Loading model from mlflow: {model_uri}...", flush=True)
            try:
                model = mlflow.sklearn.load_model(model_uri=model_uri)
                print("Model loaded successfully!", flush=True)
                last_error = None
                break
            except Exception as e:
                last_error = e
                print(f"Failed loading {model_uri}: {e}", flush=True)

        if last_error is not None:
            raise last_error
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        model = None

# ==========================================
# Schema de entrada (payload bruto → DataFrame para o Pipeline)
# ==========================================


class CreditApplication(BaseModel):
    id_cliente: str
    id_contrato: Optional[str] = None
    tipo_contrato: str
    status_contrato: str
    tipo_pagamento: str
    finalidade_emprestimo: str
    tipo_cliente: str
    tipo_portfolio: str
    tipo_produto: str
    categoria_bem: str
    setor_vendedor: str
    canal_venda: str
    faixa_rendimento: Optional[str] = None
    combinacao_produto: Optional[str] = None
    area_venda: Optional[str] = None
    dia_semana_solicitacao: Optional[str] = None
    data_nascimento: str
    data_decisao: str
    data_liberacao: Optional[str] = None
    data_primeiro_vencimento: Optional[str] = None
    data_ultimo_vencimento_original: Optional[str] = None
    data_ultimo_vencimento: Optional[str] = None
    data_encerramento: Optional[str] = None
    valor_solicitado: float
    valor_credito: float
    valor_bem: float
    valor_parcela: float
    valor_entrada: float
    percentual_entrada: float
    qtd_parcelas_planejadas: int
    taxa_juros_padrao: float
    taxa_juros_promocional: float
    hora_solicitacao: int
    flag_ultima_solicitacao_contrato: int
    flag_ultima_solicitacao_dia: int
    acompanhantes_cliente: int
    flag_seguro_contratado: int
    motivo_recusa: Optional[str] = None
    # Cadastral (merge com base_cadastral no treino — opcionais se não enviados)
    renda_anual: Optional[float] = None
    qtd_membros_familia: Optional[int] = None
    possui_carro: Optional[str] = None
    possui_imovel: Optional[str] = None


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    _stop_spark_session()


app = FastAPI(title="Datarisk Credit Scoring API", lifespan=_lifespan)


# ==========================================
# Endpoints
# ==========================================


@app.post("/predict")
async def predict(application: CreditApplication):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")

    try:
        request_payload = application.model_dump()
        input_df = pd.DataFrame([request_payload])
        probability = model.predict_proba(input_df)[0, 1]
        request_id = str(uuid4())

        decision = "Aprovado"
        if probability > 0.5:
            decision = "Negado"
        elif probability > 0.3:
            decision = "Revisão Manual"

        prediction_response = {
            "request_id": request_id,
            "probability": round(float(probability), 4),
            "threshold_decision": decision,
            "status": "success",
        }

        try:
            await asyncio.to_thread(
                _append_prediction_to_lakehouse,
                request_payload,
                prediction_response,
                request_id,
            )
        except Exception as log_error:
            print(f"Prediction log error: {log_error}", flush=True)
            if PREDICTION_LOG_STRICT:
                raise HTTPException(
                    status_code=503,
                    detail="Falha ao persistir request/response no lakehouse.",
                ) from log_error

        return prediction_response

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "online", "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
