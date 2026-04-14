import importlib.util
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _load_prod_module():
    module_path = Path(__file__).resolve().parents[1] / "prod" / "lgbm_prod.py"
    spec = importlib.util.spec_from_file_location("lgbm_prod", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def api_module():
    return _load_prod_module()


@pytest.fixture()
def valid_payload():
    return {
        "id_cliente": "C001",
        "id_contrato": "K001",
        "tipo_contrato": "Novo",
        "status_contrato": "Ativo",
        "tipo_pagamento": "Boleto",
        "finalidade_emprestimo": "Pessoal",
        "tipo_cliente": "PF",
        "tipo_portfolio": "Retail",
        "tipo_produto": "Credito",
        "categoria_bem": "Sem garantia",
        "setor_vendedor": "Digital",
        "canal_venda": "Online",
        "faixa_rendimento": "2-5k",
        "combinacao_produto": "A",
        "area_venda": "Sul",
        "dia_semana_solicitacao": "Segunda",
        "data_nascimento": "1990-01-01",
        "data_decisao": "2025-01-10",
        "data_liberacao": "2025-01-11",
        "data_primeiro_vencimento": "2025-02-10",
        "data_ultimo_vencimento_original": "2025-12-10",
        "data_ultimo_vencimento": "2025-12-10",
        "data_encerramento": None,
        "valor_solicitado": 10000.0,
        "valor_credito": 9500.0,
        "valor_bem": 12000.0,
        "valor_parcela": 500.0,
        "valor_entrada": 1000.0,
        "percentual_entrada": 0.1,
        "qtd_parcelas_planejadas": 24,
        "taxa_juros_padrao": 0.03,
        "taxa_juros_promocional": 0.02,
        "hora_solicitacao": 10,
        "flag_ultima_solicitacao_contrato": 0,
        "flag_ultima_solicitacao_dia": 1,
        "acompanhantes_cliente": 0,
        "flag_seguro_contratado": 1,
        "motivo_recusa": None,
        "renda_anual": 72000.0,
        "qtd_membros_familia": 3,
        "possui_carro": "Y",
        "possui_imovel": "N",
    }


class DummyModel:
    def __init__(self, probability):
        self._probability = probability

    def predict_proba(self, input_df):
        assert len(input_df) == 1
        return np.array([[1 - self._probability, self._probability]])


def test_health_reports_model_loaded_flag(api_module):
    api_module.model = DummyModel(0.2)
    client = TestClient(api_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "online", "model_loaded": True}


def test_predict_returns_aprovado(api_module, valid_payload):
    api_module.model = DummyModel(0.2)
    client = TestClient(api_module.app)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["threshold_decision"] == "Aprovado"
    assert response.json()["probability"] == 0.2


def test_predict_returns_revisao_manual(api_module, valid_payload):
    api_module.model = DummyModel(0.4)
    client = TestClient(api_module.app)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["threshold_decision"] == "Revisão Manual"
    assert response.json()["probability"] == 0.4


def test_predict_returns_negado(api_module, valid_payload):
    api_module.model = DummyModel(0.8)
    client = TestClient(api_module.app)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["threshold_decision"] == "Negado"
    assert response.json()["probability"] == 0.8


def test_predict_returns_500_when_model_is_not_loaded(api_module, valid_payload):
    api_module.model = None
    client = TestClient(api_module.app)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Modelo não carregado no servidor."


def test_predict_returns_400_when_model_raises(api_module, valid_payload):
    class FailingModel:
        def predict_proba(self, _input_df):
            raise RuntimeError("bad payload")

    api_module.model = FailingModel()
    client = TestClient(api_module.app)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "bad payload"
