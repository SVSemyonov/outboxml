FROM ghcr.io/mlflow/mlflow:v2.7.0
user root

RUN mkdir -p /root/mlflow
RUN mkdir -p /root/mlflow/artifacts
RUN chmod 777 /root/mlflow
RUN chmod 777 /root/mlflow/artifacts

RUN pip install psycopg2-binary
RUN pip install mlflow
