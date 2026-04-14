FROM apache/airflow:3.1.0

USER root

# LightGBM needs libgomp.so.1 at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY docker/airflow/requirements-af.txt /opt/airflow/requirements-af.txt

# apache/airflow forbids pip as root; install deps as the same user runtime uses.
USER airflow
RUN pip install --no-cache-dir -r /opt/airflow/requirements-af.txt
