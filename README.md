# README
OutBoxML is an open-source framework designed to improve the process of automating machine learning pipelines from model training to deployment. This toolkit integrates several key components including Python for model development, Grafana for monitoring, FastAPI for serving models, and MLFlow for experiment tracking and management. Our aim is to provide a robust and user-friendly platform for ML practitioners to efficiently build, deploy, and monitor their ML solutions with ease. 

The key components include:
- **AutoML**: Use AutoML algorithm with boosting or implement your custom models using low-code solution 
- **MLFlow**: Track experiments, parameters, and outputs with MLFlow .
- **Grafana Monitoring**: Utilize Grafana dashboards to monitor ML models performance in real-time
- **FastAPI**: Host the models with FastAPI that allows for quick deployment and testing of ML models via RESTful APIs.
- **PostgreSQL**: Use open source database to store and update data for AutoML proceses

The main connections between components are made with Docker, the framework requires OS with Docker и Docker Compose installed.

 
## Communications between the containers
All containers use one Docker network, by default (`<project>_default`):
- **MLflow** Communicates with PostgreSQL using `postgre`.
- **Prometheus** collect metrics from `node-exporter`.
- **FastAPI** Sends metrics to MLflow with REST API.
 

## Ports
By default containers map to the following ports:
- **MLflow**: `5000:5000`
- **Grafana**: `3000:3000`
- **Prometheus**: `9090:9090`
- **Node Exporter**: `9100:9100`
- **Jupyter Notebook**: `8889:8888`
- **FastAPI**: `8000:8000`
- **Minio**: `9001:9001`
  
## Начало работы
- Обязательно запустить файл create-folder.bat перед началом всех действий

## Настройка Minio (Обязательный пункт)
- Зайти на http://localhost:9001 (логин: minio, пароль: Strong#Pass#2022)
- Нажать "Create Bucket" и назвать mlflow
- Затем зайти в bucket
- Нажать редактировать "Access Policy:"
- Выставить там Public и нажать set
![Minio-mlflow-settings](outboxml/image.png)

## Network restrictions and security concerts
- The containers are isolated 
- Use firewall on the host machine for extra security 

1. **Jupyter Notebook**
   - By default open without password or token

2. **Prometheus и Grafana**
   - Grafana требует ручного подключения источника данных (Prometheus).

## Getting Started
1. To start the project
   ```bash
   docker compose up
   ```
   or for backround lunch
   ```bash
   docker compose up -d
   ```

- To restart:
  ```bash
  docker compose down && docker compose up --build
  ```
- To stop the project:
  ```bash
  docker compose down
  ```

2. Check availablity
   - MLflow: [http://localhost:5000](http://localhost:5000)
   - Grafana: [http://localhost:3000](http://localhost:3000) (default login/password: `admin/admin`)
   - Prometheus: [http://localhost:9090](http://localhost:9090)
   - Jupyter Notebook: [http://localhost:8889](http://localhost:8889)
   - FastAPI: [http://localhost:8000](http://localhost:8000)

3. Ensure that all containters are up
   ```bash
   docker ps
   ```
   
4.For testing of FastAPI use Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs).

## Possible issues and solutions 
1. **The ports are in sue**:
   - Find and free the neccesary ports:
     ```bash
     sudo lsof -i:<порт>
     ```
   - Alternatively change the ports in `docker-compose.yml`.

2. **No connection between containers**:
   - Check names of Docker network:
     ```bash
     docker network inspect <project>_default
     ```

3. **No connections between FastAPI and MLflow**:
   - Check connections MLflow API:
     ```bash
     curl http://mlflow:5000/api/2.0/mlflow/experiments/list
     ```
## Contributing
We welcome contributions from the community! If you'd like to contribute, please follow the contributing guidelines outlined in CONTRIBUTING.md.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Support
For support, please open an issue on GitHub or contact the maintainers directly.

## Acknowledgements
We would like to thank VSK, whose support and environment for innovation have been pivotal to the success of the project.
Special thanks to Nikulin Vladimir, who not only supervised the business aspects of the 
project but also provided invaluable insights and guidance on integrating it with business workflows. 
We appreciate help of our data science department for integration of the framework in the ML processes and MLOps team specifically(Makeev Aleksey and Zotov Dmitry) for testing and DevOps integrations.   

## Current contributors
- Semyon Semyonov-  original codebase developement, system design and product management 
- Suvorov Vladimir - core code development, software architecture
- Bochkarev Dmitry - code development, development of data science models
- Matcera Maxim -  specific modules development

 

   
   

