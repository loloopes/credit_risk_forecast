import os
from typing import Optional

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# IMPORTANTE: A classe CreditFeatureEngineering deve estar definida aqui 
# ou importada de um módulo comum para que o log_model funcione.

# ==========================================
# 1. CONFIGURAÇÃO DE AMBIENTE
# ==========================================
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio123"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

mlflow.set_tracking_uri("http://localhost:5000")

app = FastAPI(title="Datarisk Credit Scoring API")

# ==========================================
# 2. CARREGAMENTO DO MODELO
# ==========================================
# Substitua pelo seu RUN_ID real ou use o alias 'latest' se configurado no Registry
RUN_ID = "654bc08bc8194392822cd24b8549fa4f" 
model_uri = f"runs:/{RUN_ID}/credit_model_pipeline"
try:
    print(f"Baixando modelo de: {model_uri}...")
    model = mlflow.sklearn.load_model(model_uri)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    model = None

# ==========================================
# 3. SCHEMA DE ENTRADA (FEATURES)
# ==========================================
from typing import Optional

from pydantic import BaseModel


class CreditApplication(BaseModel):
    # --- Identificadores (removidos no transform, mas necessários no input) ---
    id_cliente: str
    id_contrato: Optional[str] = None
    
    # --- Colunas Categóricas (Processadas pelo OrdinalEncoder) ---
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
    
    # --- Datas (Usadas para calcular Idade e depois removidas) ---
    data_nascimento: str # Formato 'YYYY-MM-DD'
    data_decisao: str    # Formato 'YYYY-MM-DD'
    data_liberacao: Optional[str] = None
    data_primeiro_vencimento: Optional[str] = None
    data_ultimo_vencimento_original: Optional[str] = None
    data_ultimo_vencimento: Optional[str] = None
    data_encerramento: Optional[str] = None
    
    # --- Features Numéricas (Passadas direto ao LightGBM) ---
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
    
    # --- Outros ---
    motivo_recusa: Optional[str] = None
# ==========================================
# 4. ENDPOINTS
# ==========================================

@app.post("/predict")
async def predict(application: CreditApplication):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")
    
    try:
        # 1. Converter entrada para DataFrame (esperado pelo Pipeline)
        input_df = pd.DataFrame([application.model_dump()])
        
        # 2. Inferência (O Pipeline executa o CreditFeatureEngineering automaticamente)
        probability = model.predict_proba(input_df)[0, 1]
        
        # 3. Política de Crédito Simples baseada nos seus Thresholds
        decision = "Aprovado"
        if probability > 0.5:
            decision = "Negado"
        elif probability > 0.3:
            decision = "Revisão Manual"

        return {
            "probability": round(float(probability), 4),
            "threshold_decision": decision,
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "online", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)