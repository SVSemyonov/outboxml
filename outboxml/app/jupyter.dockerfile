FROM jupyter/base-notebook:python-3.11

COPY requirements-3-11.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
