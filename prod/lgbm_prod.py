import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ==========================================
# 1. DEFINIÇÃO DA CLASSE (Obrigatório para o Pipeline funcionar)
# ==========================================
# Se a classe CreditFeatureEngineering estiver em outro arquivo, importe-a aqui:
# from feature_engineering import CreditFeatureEngineering

app = FastAPI(title="Datarisk Credit Scoring API")

# ==========================================
# 2. CARREGAMENTO LOCAL DO MODELO
# ==========================================

MODEL_PATH = "credit_model_pipeline.pkl"

import __main__

try:
    print(f"Carregando modelo local de: {MODEL_PATH}...")

    # Inject class into __main__ (fix pickle issue)
    from feature_engineering import CreditFeatureEngineering
    __main__.CreditFeatureEngineering = CreditFeatureEngineering

    model = joblib.load(MODEL_PATH)

    print("✅ Modelo carregado com sucesso!")

except Exception as e:
    print(f"❌ Erro crítico ao carregar modelo: {e}")
    model = None
# ==========================================
# 3. SCHEMA DE ENTRADA (FEATURES)
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

# ==========================================
# 4. ENDPOINTS
# ==========================================

@app.post("/predict")
async def predict(application: CreditApplication):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor local.")
    
    try:
        # 1. Converter entrada para DataFrame
        input_df = pd.DataFrame([application.model_dump()])
        
        # 2. Inferência
        probability = model.predict_proba(input_df)[0, 1]
        
        # 3. Política de Crédito baseada nos Thresholds definidos no projeto
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