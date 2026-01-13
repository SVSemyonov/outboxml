Installation
============

Prerequisites
-------------

OutBoxML requires:

* Python 3.11 or higher
* Docker and Docker Compose
* Operating system with Docker support

Installing OutBoxML
-------------------

Clone the repository:

.. code-block:: bash

    git clone <repository-url>
    cd outboxml

Install dependencies:

.. code-block:: bash

    pip install -r requirements.txt

Docker Setup
------------

1. Navigate to the app directory:

.. code-block:: bash

    cd outboxml/app

2. Create necessary folders (Windows):

.. code-block:: bash

    create-folder.bat

Or on Linux/Mac:

.. code-block:: bash

    ./create-folder.sh

3. Start the Docker containers:

.. code-block:: bash

    docker compose up

Or for background launch:

.. code-block:: bash

    docker compose up -d

4. Verify all containers are running:

.. code-block:: bash

    docker ps

Accessing Services
-----------------

After starting the containers, you can access:

* **MLflow**: http://localhost:5000
* **Grafana**: http://localhost:3000 (default login/password: ``admin/admin``)
* **Prometheus**: http://localhost:9090
* **Jupyter Notebook**: http://localhost:8889
* **FastAPI**: http://localhost:8000
* **MinIO**: http://localhost:9001 (login: ``minio``, password: ``Strong#Pass#2022``)

MinIO Setup
-----------

1. Open http://localhost:9001
2. Click "Create Bucket" with name "mlflow"
3. Open the bucket and edit "Access Policy:"
4. Set Access Policy to Public and click set

Configuration
-------------

Create a ``.env`` file in the project root with the following variables:

.. code-block:: bash

    # Email configuration
    email_smtp_server=smtp.gmail.com
    email_port=587
    email_sender=your_email@example.com
    email_login=your_email@example.com
    email_pass=your_password
    email_receivers=recipient1@example.com,recipient2@example.com

    # MLflow configuration
    mlflow_tracking_uri=http://localhost:5000
    mlflow_experiment=FrameworkTest

    # Database configuration
    connection=postgresql+psycopg2://mlflow:mlflowpassword@127.0.0.1:5433/mlflow

    # Paths
    results_path=./results
    prod_models_folder=prod_models_from_mlflow
