# Базовый образ
FROM mlflow:v2.7.0
user root
#COPY pip.conf /etc/pip.conf
RUN mkdir -p /root/mlflow
RUN mkdir -p /root/mlflow/artifacts
chmod 777 /root/mlflow
chmod 777 /root/mlflow/artifacts
# Установка psycopg2
RUN pip install psycopg2-binary
RUN pip install mlflow
