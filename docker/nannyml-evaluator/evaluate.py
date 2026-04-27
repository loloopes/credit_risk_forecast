import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO, StringIO
from urllib.parse import urlparse

import boto3
import nannyml as nml
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

API_REQUIRED_FIELDS = [
    "id_cliente",
    "tipo_contrato",
    "status_contrato",
    "tipo_pagamento",
    "finalidade_emprestimo",
    "tipo_cliente",
    "tipo_portfolio",
    "tipo_produto",
    "categoria_bem",
    "setor_vendedor",
    "canal_venda",
    "data_nascimento",
    "data_decisao",
    "valor_solicitado",
    "valor_credito",
    "valor_bem",
    "valor_parcela",
    "valor_entrada",
    "percentual_entrada",
    "qtd_parcelas_planejadas",
    "taxa_juros_padrao",
    "taxa_juros_promocional",
    "hora_solicitacao",
    "flag_ultima_solicitacao_contrato",
    "flag_ultima_solicitacao_dia",
    "acompanhantes_cliente",
    "flag_seguro_contratado",
]

API_ALLOWED_FIELDS = set(
    API_REQUIRED_FIELDS
    + [
        "id_contrato",
        "faixa_rendimento",
        "combinacao_produto",
        "area_venda",
        "dia_semana_solicitacao",
        "data_liberacao",
        "data_primeiro_vencimento",
        "data_ultimo_vencimento_original",
        "data_ultimo_vencimento",
        "data_encerramento",
        "motivo_recusa",
        "renda_anual",
        "qtd_membros_familia",
        "possui_carro",
        "possui_imovel",
    ]
)

STRING_FIELDS = {
    "id_cliente",
    "id_contrato",
    "tipo_contrato",
    "status_contrato",
    "tipo_pagamento",
    "finalidade_emprestimo",
    "tipo_cliente",
    "tipo_portfolio",
    "tipo_produto",
    "categoria_bem",
    "setor_vendedor",
    "canal_venda",
    "faixa_rendimento",
    "combinacao_produto",
    "area_venda",
    "dia_semana_solicitacao",
    "data_nascimento",
    "data_decisao",
    "data_liberacao",
    "data_primeiro_vencimento",
    "data_ultimo_vencimento_original",
    "data_ultimo_vencimento",
    "data_encerramento",
    "motivo_recusa",
    "possui_carro",
    "possui_imovel",
}

INT_FIELDS = {
    "qtd_parcelas_planejadas",
    "hora_solicitacao",
    "flag_ultima_solicitacao_contrato",
    "flag_ultima_solicitacao_dia",
    "acompanhantes_cliente",
    "flag_seguro_contratado",
    "qtd_membros_familia",
}

FLOAT_FIELDS = {
    "valor_solicitado",
    "valor_credito",
    "valor_bem",
    "valor_parcela",
    "valor_entrada",
    "percentual_entrada",
    "taxa_juros_padrao",
    "taxa_juros_promocional",
    "renda_anual",
}

REQUIRED_FALLBACKS = {
    "id_cliente": "unknown",
    "tipo_contrato": "financiamento",
    "status_contrato": "ativo",
    "tipo_pagamento": "boleto",
    "finalidade_emprestimo": "compra_veiculo",
    "tipo_cliente": "pessoa_fisica",
    "tipo_portfolio": "varejo",
    "tipo_produto": "credito_pessoal",
    "categoria_bem": "automovel",
    "setor_vendedor": "digital",
    "canal_venda": "online",
    "data_nascimento": "1990-01-01",
    "data_decisao": "2024-01-01",
    "valor_solicitado": 0.0,
    "valor_credito": 0.0,
    "valor_bem": 0.0,
    "valor_parcela": 0.0,
    "valor_entrada": 0.0,
    "percentual_entrada": 0.0,
    "qtd_parcelas_planejadas": 12,
    "taxa_juros_padrao": 0.03,
    "taxa_juros_promocional": 0.03,
    "hora_solicitacao": 12,
    "flag_ultima_solicitacao_contrato": 0,
    "flag_ultima_solicitacao_dia": 0,
    "acompanhantes_cliente": 0,
    "flag_seguro_contratado": 0,
}


def _load_payload_defaults() -> dict:
    raw = os.getenv("API_PAYLOAD_DEFAULTS_JSON", "").strip()
    if not raw:
        return {}
    try:
        defaults = json.loads(raw)
        if isinstance(defaults, dict):
            return defaults
    except Exception:
        pass
    return {}


def is_s3(path: str) -> bool:
    return path.startswith("s3://")


def parse_s3(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(s3_uri)
    return parsed.netloc, parsed.path.lstrip("/")


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def load_df(uri: str) -> pd.DataFrame:
    if is_s3(uri):
        bucket, key = parse_s3(uri)
        data = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        if uri.lower().endswith((".parquet", ".pq")):
            return pd.read_parquet(BytesIO(data))
        return pd.read_csv(BytesIO(data))
    if uri.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(uri)
    return pd.read_csv(uri)


def list_s3_keys(prefix_uri: str) -> tuple[str, list[str]]:
    bucket, prefix = parse_s3(prefix_uri)
    paginator = s3_client().get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return bucket, keys


def load_merged_from_prefix(
    prefix_uri: str, suffix: str, id_cliente_col: str, id_contrato_col: str
) -> pd.DataFrame:
    bucket, keys = list_s3_keys(prefix_uri)
    wanted = {}
    for key in keys:
        name = key.split("/")[-1]
        if name == f"base_submissao{suffix}":
            wanted["submissao"] = key
        elif name == f"base_cadastral{suffix}":
            wanted["cadastral"] = key
        elif name == f"historico_emprestimos{suffix}":
            wanted["emprestimos"] = key
        elif name == f"historico_parcelas{suffix}":
            wanted["parcelas"] = key

    if "submissao" not in wanted:
        raise RuntimeError(f"Missing base_submissao file under {prefix_uri} with suffix {suffix}")

    def read_key(key: str) -> pd.DataFrame:
        data = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        if key.lower().endswith((".parquet", ".pq")):
            return pd.read_parquet(BytesIO(data))
        return pd.read_csv(BytesIO(data))

    base_df = read_key(wanted["submissao"])

    if "cadastral" in wanted:
        cad = read_key(wanted["cadastral"])
        if id_cliente_col in base_df.columns and id_cliente_col in cad.columns:
            extra = [c for c in cad.columns if c != id_cliente_col and c not in base_df.columns]
            base_df = base_df.merge(cad[[id_cliente_col] + extra], on=id_cliente_col, how="left")

    if "emprestimos" in wanted:
        emp = read_key(wanted["emprestimos"])
        if id_contrato_col in base_df.columns and id_contrato_col in emp.columns:
            extra = [c for c in emp.columns if c != id_contrato_col and c not in base_df.columns]
            base_df = base_df.merge(emp[[id_contrato_col] + extra], on=id_contrato_col, how="left")

    if "parcelas" in wanted:
        par = read_key(wanted["parcelas"])
        if id_contrato_col in base_df.columns and id_contrato_col in par.columns:
            num_cols = [c for c in par.select_dtypes(include=["number"]).columns if c != id_contrato_col]
            if num_cols:
                agg = par.groupby(id_contrato_col, as_index=False).agg({c: "mean" for c in num_cols})
                agg = agg.rename(columns={c: f"{c}_parcelas_mean" for c in num_cols})
                base_df = base_df.merge(agg, on=id_contrato_col, how="left")

    return base_df


def save_df(df: pd.DataFrame, uri: str) -> None:
    if is_s3(uri):
        bucket, key = parse_s3(uri)
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        s3_client().put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode("utf-8"))
        return
    df.to_csv(uri, index=False)


def _build_payload(row: pd.Series, defaults: dict) -> dict | None:
    payload = {}
    for key in API_ALLOWED_FIELDS:
        if key in row.index:
            value = row[key]
            payload[key] = None if pd.isna(value) else value

    for key, value in defaults.items():
        if key in API_ALLOWED_FIELDS and (key not in payload or payload[key] in (None, "")):
            payload[key] = value

    for key, value in REQUIRED_FALLBACKS.items():
        if key not in payload or payload[key] in (None, ""):
            payload[key] = value

    for required_key in API_REQUIRED_FIELDS:
        if required_key not in payload or payload[required_key] in (None, ""):
            return None

    for key, value in list(payload.items()):
        if value is None:
            continue
        try:
            if key in STRING_FIELDS:
                payload[key] = str(value)
            elif key in INT_FIELDS:
                payload[key] = int(value)
            elif key in FLOAT_FIELDS:
                payload[key] = float(value)
        except Exception:
            return None

    return payload


def _missing_required_fields(row: pd.Series) -> list[str]:
    missing = []
    for required_key in API_REQUIRED_FIELDS:
        if required_key not in row.index or pd.isna(row[required_key]) or row[required_key] == "":
            missing.append(required_key)
    return missing


def _lakehouse_negado_fraction(
    events_df: pd.DataFrame,
    column: str,
) -> tuple[float | None, int, int]:
    """Parse response_json strings; fraction of rows with threshold_decision == Negado among successful predictions."""
    if column not in events_df.columns:
        raise RuntimeError(
            f"Lakehouse prediction events export is missing column {column!r}. "
            f"Columns present: {list(events_df.columns)}"
        )

    negado_count = 0
    total = 0
    for raw in events_df[column]:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        if isinstance(raw, dict):
            obj = raw
        elif isinstance(raw, str):
            raw_stripped = raw.strip()
            if not raw_stripped:
                continue
            try:
                obj = json.loads(raw_stripped)
            except json.JSONDecodeError:
                continue
        else:
            continue

        status = obj.get("status")
        if status is not None and status != "success":
            continue

        total += 1
        if obj.get("threshold_decision") == "Negado":
            negado_count += 1

    if total == 0:
        return None, 0, 0
    return negado_count / total, negado_count, total


def _score_payload(
    row_dict: dict, payload: dict, api_url: str, timeout: int, prediction_col: str
) -> tuple[bool, dict | None, dict | None]:
    resp = requests.post(api_url, json=payload, timeout=timeout)
    if resp.status_code == 422:
        try:
            detail = resp.json()
        except Exception:
            detail = {"raw_text": resp.text[:500]}
        return False, None, detail
    resp.raise_for_status()

    scored_row = dict(row_dict)
    scored_row[prediction_col] = float(resp.json().get("probability"))
    return True, scored_row, None


def main() -> None:
    defaults = _load_payload_defaults()

    reference_uri = os.environ["REFERENCE_DATA_PATH"]
    analysis_uri = os.environ["ANALYSIS_DATA_PATH"]
    ref_suffix = os.getenv("REFERENCE_SUFFIX", ".parquet")
    ana_suffix = os.getenv("ANALYSIS_SUFFIX", "_new.parquet")
    id_cliente_col = os.getenv("MONITOR_JOIN_ID_CLIENTE", "id_cliente")
    id_contrato_col = os.getenv("MONITOR_JOIN_ID_CONTRATO", "id_contrato")

    if reference_uri.startswith("s3://") and not reference_uri.lower().endswith((".csv", ".parquet", ".pq")):
        reference_df = load_merged_from_prefix(reference_uri, ref_suffix, id_cliente_col, id_contrato_col)
    else:
        reference_df = load_df(reference_uri)

    if analysis_uri.startswith("s3://") and not analysis_uri.lower().endswith((".csv", ".parquet", ".pq")):
        analysis_df = load_merged_from_prefix(analysis_uri, ana_suffix, id_cliente_col, id_contrato_col)
    else:
        analysis_df = load_df(analysis_uri)

    max_rows = int(os.getenv("MONITOR_MAX_ROWS", "0"))
    if max_rows > 0:
        analysis_df = analysis_df.head(max_rows).copy()

    api_url = os.environ["PREDICTION_API_URL"]
    timeout = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
    prediction_col = os.getenv("MONITOR_PREDICTION_COLUMN", "prediction_proba")
    worker_count = max(1, int(os.getenv("MONITOR_API_WORKERS", "8")))
    progress_every = max(1, int(os.getenv("MONITOR_PROGRESS_EVERY", "500")))

    analysis_scored_rows: list[dict] = []
    dropped_rows = 0
    missing_counter: dict[str, int] = {}
    api_422_count = 0
    first_422_detail = None
    valid_rows: list[tuple[dict, dict]] = []
    for _, row in analysis_df.iterrows():
        payload = _build_payload(row, defaults)
        if payload is None:
            dropped_rows += 1
            for field in _missing_required_fields(row):
                missing_counter[field] = missing_counter.get(field, 0) + 1
            continue
        valid_rows.append((row.to_dict(), payload))

    total_valid_rows = len(valid_rows)
    if total_valid_rows > 0:
        print(
            (
                "Scoring %s rows via API with %s workers "
                "(progress every %s rows)"
            )
            % (total_valid_rows, worker_count, progress_every),
            flush=True,
        )
        completed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(_score_payload, row_dict, payload, api_url, timeout, prediction_col): idx
                for idx, (row_dict, payload) in enumerate(valid_rows, start=1)
            }
            for future in as_completed(future_to_idx):
                completed += 1
                ok, scored_row, detail = future.result()
                if ok and scored_row is not None:
                    analysis_scored_rows.append(scored_row)
                else:
                    dropped_rows += 1
                    api_422_count += 1
                    if first_422_detail is None:
                        first_422_detail = detail
                if completed % progress_every == 0 or completed == total_valid_rows:
                    print(
                        f"Scoring progress: {completed}/{total_valid_rows} rows processed",
                        flush=True,
                    )

    if not analysis_scored_rows:
        top_missing = sorted(missing_counter.items(), key=lambda x: x[1], reverse=True)[:8]
        raise RuntimeError(
            "No valid rows for API scoring. "
            f"Top missing required fields: {top_missing}. "
            f"API 422 rows: {api_422_count}. "
            f"First 422 detail: {first_422_detail}. "
            f"Available columns sample: {list(analysis_df.columns[:25])}"
        )

    analysis_scored = pd.DataFrame.from_records(analysis_scored_rows)
    if dropped_rows > 0:
        print(f"Dropped rows due to invalid API payload: {dropped_rows}", flush=True)

    timestamp_col = os.getenv("MONITOR_TIMESTAMP_COLUMN")
    ignore_cols = {timestamp_col, os.getenv("MONITOR_TARGET_COLUMN", "target"), prediction_col}
    feature_cols = [c for c in analysis_scored.columns if c not in ignore_cols]
    feature_cols = [c for c in feature_cols if c in reference_df.columns]

    kwargs = {"column_names": feature_cols, "chunk_size": 500}
    if timestamp_col and timestamp_col in reference_df.columns and timestamp_col in analysis_scored.columns:
        kwargs["timestamp_column_name"] = timestamp_col

    try:
        calc = nml.DataReconstructionDriftCalculator(**kwargs)
    except AttributeError:
        calc = nml.UnivariateDriftCalculator(**kwargs)

    ref_cols = feature_cols + ([timestamp_col] if timestamp_col in reference_df.columns else [])
    ana_cols = feature_cols + ([timestamp_col] if timestamp_col in analysis_scored.columns else [])
    calc.fit(reference_df[ref_cols])
    drift_result = calc.calculate(analysis_scored[ana_cols]).to_df()

    save_df(drift_result, os.environ["MONITOR_OUTPUT_PATH"])

    alert_series = drift_result.get("alert")
    alert_rate = float(alert_series.fillna(False).astype(bool).mean()) if alert_series is not None else 0.0
    drift_detected = alert_rate >= float(os.getenv("DRIFT_ALERT_THRESHOLD", "0.7"))

    target_col = os.getenv("MONITOR_TARGET_COLUMN", "target")
    decay_detected = False
    decay_checked = False
    auc_drop = None
    if target_col in analysis_scored.columns and prediction_col in analysis_scored.columns:
        if target_col in reference_df.columns and prediction_col in reference_df.columns:
            decay_checked = True
            ref_auc = float(roc_auc_score(reference_df[target_col], reference_df[prediction_col]))
            ana_auc = float(roc_auc_score(analysis_scored[target_col], analysis_scored[prediction_col]))
            auc_drop = ref_auc - ana_auc
            decay_detected = auc_drop >= float(os.getenv("MODEL_DECAY_MIN_AUC_DROP", "0.03"))

    negado_checked = False
    negado_fraction = None
    negado_below_threshold_retrain = False
    negado_count = 0
    negado_denominator = 0

    lake_uri = os.getenv("LAKEHOUSE_PREDICTION_EVENTS_URI", "").strip()
    if lake_uri:
        negado_checked = True
        lake_col = os.getenv("LAKEHOUSE_RESPONSE_JSON_COLUMN", "response_json")
        min_negado_rate = float(os.getenv("LAKEHOUSE_NEGADO_MIN_RATE", "0.90"))
        events_df = load_df(lake_uri)
        negado_fraction, negado_count, negado_denominator = _lakehouse_negado_fraction(
            events_df, lake_col
        )
        if negado_fraction is not None:
            negado_below_threshold_retrain = negado_fraction < min_negado_rate
            print(
                json.dumps(
                    {
                        "lakehouse_negado_check": True,
                        "negado_fraction": round(negado_fraction, 6),
                        "negado_count": negado_count,
                        "negado_denominator": negado_denominator,
                        "min_negado_rate_required": min_negado_rate,
                        "below_threshold_triggers_retrain": negado_below_threshold_retrain,
                    }
                ),
                flush=True,
            )
        else:
            print(
                json.dumps(
                    {
                        "lakehouse_negado_check": True,
                        "warning": "no countable prediction rows "
                        "(status success, valid response_json)",
                        "lakehouse_uri": lake_uri,
                    }
                ),
                flush=True,
            )

    result = {
        "should_retrain": bool(
            drift_detected or decay_detected or negado_below_threshold_retrain
        ),
        "drift_detected": bool(drift_detected),
        "decay_detected": bool(decay_detected),
        "decay_checked": bool(decay_checked),
        "alert_rate": alert_rate,
        "auc_drop": auc_drop,
        "lakehouse_negado_checked": negado_checked,
        "negado_fraction": negado_fraction,
        "negado_below_threshold_retrain": bool(negado_below_threshold_retrain),
        "negado_count": negado_count,
        "negado_denominator": negado_denominator,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
