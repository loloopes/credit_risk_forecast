import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

import __main__

# ==========================================
# Feature engineering (same as model/lgbm_test.ipynb — Pipeline step 'fe')
# ==========================================


class CreditFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.target_cat_cols = [
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
        ]
        self.feature_names = feature_names

    def fit(self, X, y=None):
        actual_cat = [c for c in self.target_cat_cols if c in X.columns]
        if actual_cat:
            self.encoder.fit(X[actual_cat].fillna("Missing").astype(str))
        return self

    def transform(self, X):
        X_c = X.copy()

        if "renda_anual" in X_c.columns:
            renda_mensal = X_c["renda_anual"] / 12
            if "valor_parcela" in X_c.columns:
                X_c["feat_comprometimento_renda"] = X_c["valor_parcela"] / renda_mensal.replace(0, np.nan)
            if "qtd_membros_familia" in X_c.columns:
                X_c["feat_renda_per_capita"] = X_c["renda_anual"] / X_c["qtd_membros_familia"].replace(0, np.nan)

        if "possui_carro" in X_c.columns:
            X_c["feat_possui_carro"] = X_c["possui_carro"].map({"Y": 1, "N": 0}).fillna(0)
        if "possui_imovel" in X_c.columns:
            X_c["feat_possui_imovel"] = X_c["possui_imovel"].map({"Y": 1, "N": 0}).fillna(0)

        if "data_nascimento" in X_c.columns and "data_decisao" in X_c.columns:
            X_c["data_nascimento"] = pd.to_datetime(X_c["data_nascimento"], errors="coerce")
            X_c["data_decisao"] = pd.to_datetime(X_c["data_decisao"], errors="coerce")
            X_c["feat_idade"] = (X_c["data_decisao"] - X_c["data_nascimento"]).dt.days // 365

        if "valor_credito" in X_c.columns and "valor_bem" in X_c.columns:
            X_c["feat_ltv"] = X_c["valor_credito"] / X_c["valor_bem"].replace(0, np.nan).fillna(X_c["valor_credito"])

        if "hora_solicitacao" in X_c.columns:
            X_c["feat_hora_pico_fraude"] = X_c["hora_solicitacao"].apply(lambda x: 1 if x < 7 or x > 21 else 0)

        actual_cat = [c for c in self.target_cat_cols if c in X_c.columns]
        if actual_cat:
            encoded = self.encoder.transform(X_c[actual_cat].fillna("Missing").astype(str))
            for i, col in enumerate(actual_cat):
                X_c[f"feat_{col}_enc"] = encoded[:, i]

        cols_to_drop = [
            "id_cliente",
            "id_contrato",
            "data_decisao",
            "data_liberacao",
            "data_primeiro_vencimento",
            "data_ultimo_vencimento",
            "data_ultimo_vencimento_original",
            "data_encerramento",
            "data_nascimento",
            "possui_carro",
            "possui_imovel",
        ]
        X_final = X_c.drop(columns=[c for c in cols_to_drop if c in X_c.columns], errors="ignore")
        X_final = X_final.select_dtypes(include=[np.number])

        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in X_final:
                    X_final[col] = 0
            X_final = X_final[self.feature_names]

        return X_final


__main__.CreditFeatureEngineering = CreditFeatureEngineering

app = FastAPI(title="Datarisk Credit Scoring API")

# ==========================================
# Carregamento do modelo (Pipeline: fe + LGBM)
# ==========================================

MODEL_PATH = "credit_model_pipeline.pkl"

try:
    print(f"Carregando modelo local de: {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro crítico ao carregar modelo: {e}")
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
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor local.")

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
