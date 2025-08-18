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
  
## Getting Started
- Change the directory to outboxml/app
- Run the create-folder.bat(on Windows) or create-folder.sh(on Linux) before starting any other actions.

1. To start the project change the directory to outboxml/app
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

4. For testing of FastAPI use Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs).

5. Minio setup 
- Open http://localhost:9001 (login: minio, password: Strong#Pass#2022)
- Click "Create Bucket" with name "mlflow"
- Open the bucket and edit "Access Policy:"
- Set Access Policy to Public and click set

## Network restrictions and security concerts
- The containers are isolated 
- Use firewall on the host machine for extra security 

1. **Jupyter Notebook**
   - By default open without password or token
2. **Prometheus и Grafana**
   - Manually connect Prometheus to Grafana.

## Possible issues and solutions 
1. **The ports are in use**:
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
4. Instructions if packages won't install via pip install
```
1. Log in as a user with sudo privileges.
2. Open the file /etc/default/docker:
    $ sudo nano /etc/default/docker
3. Find and uncomment or add the following line:
    DOCKER_OPTS="--dns 8.8.8.8"
4. Save and close the file.
5. Restart the Docker daemon service:
    $ sudo systemctl restart docker
```

## Contributing
We welcome contributions from the community! If you'd like to contribute, please follow the contributing guidelines outlined in docs/CONTRIBUTING.md.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Support
For support, please open an issue on GitHub or contact the maintainers directly.

## Acknowledgements
We would like to thank VSK, whose support has been pivotal to the success of the project.
Special thanks to Vladimir Nikulin, who not only supervised the business aspects of the project but also provided invaluable insights and guidance on its integration with business workflows.
We appreciate the support of our Data Science department for integrating the framework into ML processes, and extend special thanks to the MLOps team, especially Aleksey Makeev and Dmitry Zotov, or their contributions to testing and DevOps integration.

## Current contributors
- Semyon Semyonov - Original codebase development, system design and product management
- Vladimir Suvorov - Core code development and software architecture
- Dmitry Bochkarev - Code development and data science model implementation
- Maxim Matcera - Development of specific modules
   
   

