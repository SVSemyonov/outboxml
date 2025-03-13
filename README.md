# README
OutBoxML is an open-source library designed to improve the process of automating machine learning pipelines from model training to deployment. This toolkit integrates several key components including Python for model development, Grafana for monitoring, FastAPI for serving models, and MLFlow for experiment tracking and management. Our aim is to provide a robust and user-friendly platform for ML practitioners to efficiently build, deploy, and monitor their ML solutions with ease. 

The key components include:
**AutoML**: Use AutoML algorithm with boosting or implement your custom models using low-code solution 
**MLFlow**: Track experiments, parameters, and outputs with MLFlow .
Grafana Monitoring: Utilize Grafana dashboards to monitor your ML models' performance and resource usage in real-time, ensuring that you can quickly identify and react to any issues that may arise.
FastAPI for Model Serving: Serve your trained models with a high-performance, easy-to-use web framework built on FastAPI that allows for quick deployment and testing of your ML models via RESTful APIs.

Scalable and Modular: Designed to be both scalable to handle large datasets and models, and modular to allow for easy customization and extension of the library to fit your specific needs.
Comprehensive Documentation: Detailed documentation is available to guide you through the setup, usage, and customization of the library, enabling you to get started quickly and understand the full capabilities of the toolset.

## Требования к ресурсам
Для корректной работы всех сервисов рекомендуется:
- **Процессор**: Минимум 4 ядра (рекомендуется 8 ядер для интенсивной работы).
- **Оперативная память**: 16 ГБ или больше (минимум 8 ГБ).
- **Диск**: Не менее 50 ГБ свободного места для баз данных, артефактов и контейнеров.
- **ОС**: Linux, macOS, Windows с установленным Docker и Docker Compose.

## Связь между контейнерами
Все контейнеры находятся в одной сети Docker по умолчанию (`<project>_default`):
- **MLflow** взаимодействует с PostgreSQL через `postgre`.
- **Prometheus** собирает метрики с `node-exporter`.
- **FastAPI** может отправлять метрики в MLflow через REST API.

Контейнеры используют DNS-имена сервисов, определённые в `docker-compose.yml` (например, `mlflow`, `postgre`, `fastapi`).

## Порты
Порты проброшены для доступа к сервисам с хоста:
- **MLflow**: `5000:5000`
- **Grafana**: `3000:3000`
- **Prometheus**: `9090:9090`
- **Node Exporter**: `9100:9100`
- **Jupyter Notebook**: `8888:8888`
- **FastAPI**: `8000:8000`
- **Minio**: `9001:9001`

Убедитесь, что эти порты свободны. При необходимости их можно изменить в секции `ports` файла `docker-compose.yml`.

## Настройка Minio (Обязательный пункт)
- Зайти на http://localhost:9001 (логин: minio, пароль: Strong#Pass#2022)
- Нажать "Create Bucket" и назвать mlflow
- Затем зайти в bucket
- Нажать редактировать "Access Policy:"
- Выставить там Public и нажать set
![Minio-mlflow-settings](outboxml/image.png)

## Сетевые ограничения
- Вся связь между контейнерами осуществляется через сеть Docker (bridge-сеть).
- Контейнеры изолированы от внешнего мира, за исключением проброшенных портов.
- Для дополнительной безопасности можно ограничить доступ к портам, используя брандмауэр или настройку Docker-сети.

## Дополнительные моменты
1. **Jupyter Notebook**
   - Открыт доступ без токена. Для сетевого окружения настройте пароль.

2. **FastAPI**
   - Предоставляет API для интеграции с MLflow.

3. **Prometheus и Grafana**
   - Grafana требует ручного подключения источника данных (Prometheus).

## Как проверить
1. Запустите проект:
   ```bash
   docker compose up
   ```
   Для запуска в фоне (что бы можно было закрыть консоль)
   ```bash
   docker compose up -d
   ```

2. Проверьте доступ к сервисам через браузер:
   - MLflow: [http://localhost:5000](http://localhost:5000)
   - Grafana: [http://localhost:3000](http://localhost:3000) (по умолчанию логин/пароль: `admin/admin`)
   - Prometheus: [http://localhost:9090](http://localhost:9090)
   - Jupyter Notebook: [http://localhost:8888](http://localhost:8888)
   - FastAPI: [http://localhost:8000](http://localhost:8000)

3. Убедитесь, что все контейнеры работают:
   ```bash
   docker ps
   ```

4. Для тестирования FastAPI используйте Swagger-документацию: [http://localhost:8000/docs](http://localhost:8000/docs).

## Возможные проблемы и решения
1. **Контейнеры не запускаются**:
   - Проверьте логи контейнеров:
     ```bash
     docker logs <container_name>
     ```
   - Убедитесь, что все зависимости загружены.

2. **Порты заняты**:
   - Найдите процессы, использующие порты, и освободите их:
     ```bash
     sudo lsof -i:<порт>
     ```
   - Либо измените порты в `docker-compose.yml`.

3. **Контейнеры не видят друг друга**:
   - Проверьте сеть Docker:
     ```bash
     docker network inspect <project>_default
     ```
   - Убедитесь, что имена сервисов совпадают.

4. **Jupyter или MLflow недоступны**:
   - Убедитесь, что контейнеры запущены и порты проброшены.

5. **FastAPI не работает с MLflow**:
   - Проверьте связь с MLflow API:
     ```bash
     curl http://mlflow:5000/api/2.0/mlflow/experiments/list
     ```

## Полезные команды
- Перезапустить проект:
  ```bash
  docker compose down && docker compose up --build
  ```
- Остановить все контейнеры:
  ```bash
  docker compose down
  ```
- Проверить статус контейнеров:
  ```bash
  docker ps
  
