import os
from typing import Iterable, Optional

import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Datarisk Credit Scoring API")

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


def _empty_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


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


# ==========================================
# Endpoints
# ==========================================


@app.post("/predict")
async def predict(application: CreditApplication):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")

    try:
        input_df = pd.DataFrame([application.model_dump()])
        probability = model.predict_proba(input_df)[0, 1]

        decision = "Aprovado"
        if probability > 0.5:
            decision = "Negado"
        elif probability > 0.3:
            decision = "Revisão Manual"

        return {
            "probability": round(float(probability), 4),
            "threshold_decision": decision,
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "online", "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
