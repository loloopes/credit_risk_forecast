import csv
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from confluent_kafka import Producer


DATA_DICTIONARY_PATH = Path(
    os.getenv(
        "DATA_DICTIONARY_PATH",
        "/mnt/c/Users/guslc/project/credit_risk_forecast/data/dicionario_dados.csv",
    )
)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "predict")


def _parse_cli_kv() -> dict[str, str]:
    out: dict[str, str] = {}
    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def _read_dictionary_columns() -> set[str]:
    if not DATA_DICTIONARY_PATH.exists():
        raise FileNotFoundError(f"Dictionary not found: {DATA_DICTIONARY_PATH}")

    columns: set[str] = set()
    with DATA_DICTIONARY_PATH.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp, delimiter=";")
        for row in reader:
            col = (row.get("coluna") or "").strip()
            if col:
                columns.add(col)
    return columns


def _random_payload(allowed_columns: set[str], idx: int) -> dict[str, Any]:
    decision_date = date.today() - timedelta(days=random.randint(0, 30))
    birth_date = date.today() - timedelta(days=random.randint(20 * 365, 70 * 365))
    payload: dict[str, Any] = {
        "id_cliente": f"{100000 + idx}",
        "id_contrato": str(uuid4()),
        "tipo_contrato": random.choice(["Cash loans", "Revolving loans"]),
        "status_contrato": random.choice(["Approved", "Refused", "Canceled"]),
        "tipo_pagamento": random.choice(
            ["Cash through a bank", "XNA", "Non-cash from your account"]
        ),
        "finalidade_emprestimo": random.choice(
            ["XAP", "Repairs", "Used car", "Furniture", "Education"]
        ),
        "tipo_cliente": random.choice(["New", "Repeater"]),
        "tipo_portfolio": random.choice(["POS", "Cash", "Cards"]),
        "tipo_produto": random.choice(["XNA", "walk-in", "x-sell"]),
        "categoria_bem": random.choice(["Mobile", "Consumer Electronics", "Auto"]),
        "setor_vendedor": random.choice(["Connectivity", "Industry", "Trade"]),
        "canal_venda": random.choice(["Country-wide", "Regional", "Online"]),
        "faixa_rendimento": random.choice(["low", "medium", "high", None]),
        "combinacao_produto": random.choice(["Cash", "POS", "Cards", None]),
        "area_venda": random.choice(["urban", "rural", None]),
        "dia_semana_solicitacao": random.choice(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        ),
        "data_nascimento": birth_date.isoformat(),
        "data_decisao": decision_date.isoformat(),
        "data_liberacao": (decision_date + timedelta(days=1)).isoformat(),
        "data_primeiro_vencimento": (decision_date + timedelta(days=30)).isoformat(),
        "data_ultimo_vencimento_original": (
            decision_date + timedelta(days=360)
        ).isoformat(),
        "data_ultimo_vencimento": (decision_date + timedelta(days=360)).isoformat(),
        "data_encerramento": None,
        "valor_solicitado": round(random.uniform(1000, 50000), 2),
        "valor_credito": round(random.uniform(1000, 50000), 2),
        "valor_bem": round(random.uniform(800, 45000), 2),
        "valor_parcela": round(random.uniform(100, 4000), 2),
        "valor_entrada": round(random.uniform(0, 10000), 2),
        "percentual_entrada": round(random.uniform(0, 0.5), 4),
        "qtd_parcelas_planejadas": random.choice([6, 12, 18, 24, 36]),
        "taxa_juros_padrao": round(random.uniform(0.01, 0.08), 4),
        "taxa_juros_promocional": round(random.uniform(0.005, 0.06), 4),
        "hora_solicitacao": random.randint(0, 23),
        "flag_ultima_solicitacao_contrato": random.choice([0, 1]),
        "flag_ultima_solicitacao_dia": random.choice([0, 1]),
        "acompanhantes_cliente": random.randint(0, 5),
        "flag_seguro_contratado": random.choice([0, 1]),
        "motivo_recusa": random.choice([None, "score_baixo", "renda_insuficiente"]),
        "renda_anual": round(random.uniform(12000, 240000), 2),
        "qtd_membros_familia": random.randint(1, 8),
        "possui_carro": random.choice(["Y", "N", None]),
        "possui_imovel": random.choice(["Y", "N", None]),
    }
    required_fields = {
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
    }
    keep = required_fields.union(allowed_columns)
    return {k: v for k, v in payload.items() if k in keep}


def main() -> None:
    args = _parse_cli_kv()
    total_requests = int(args.get("requests", "100"))
    workers = max(int(args.get("workers", "2")), 1)
    topic = args.get("topic", KAFKA_TOPIC)
    bootstrap = args.get("bootstrap", KAFKA_BOOTSTRAP_SERVERS)

    allowed_columns = _read_dictionary_columns()
    producer = Producer({"bootstrap.servers": bootstrap})
    lock = threading.Lock()
    delivered = {"ok": 0, "err": 0}

    def _delivery_report(err, _msg) -> None:
        with lock:
            if err is None:
                delivered["ok"] += 1
            else:
                delivered["err"] += 1

    def _send_one(i: int) -> None:
        request_id = str(uuid4())
        payload = _random_payload(allowed_columns, i)
        message = {
            "request_id": request_id,
            "payload": payload,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        producer.produce(
            topic=topic,
            key=request_id.encode("utf-8"),
            value=json.dumps(message, ensure_ascii=False).encode("utf-8"),
            callback=_delivery_report,
        )
        producer.poll(0)

    print(
        f"Producing {total_requests} requests to topic={topic} "
        f"bootstrap={bootstrap} workers={workers}"
    )
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_send_one, range(total_requests)))

    producer.flush(30)
    print(
        f"Done. delivered_ok={delivered['ok']} delivered_err={delivered['err']} "
        f"requested={total_requests}"
    )


if __name__ == "__main__":
    main()

