FROM mlflow:v2.7.0
user root

RUN mkdir -p /root/mlflow
RUN mkdir -p /root/mlflow/artifacts
chmod 777 /root/mlflow
chmod 777 /root/mlflow/artifacts

RUN pip install psycopg2-binary
RUN pip install mlflow
