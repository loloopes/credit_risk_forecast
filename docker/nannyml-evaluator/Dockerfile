FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    nannyml \
    requests \
    boto3 \
    scikit-learn \
    pyarrow

COPY evaluate.py /app/evaluate.py

CMD ["python", "/app/evaluate.py"]
