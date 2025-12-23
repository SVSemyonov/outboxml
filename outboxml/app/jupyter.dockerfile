FROM jupyter/base-notebook:python-3.13

COPY requirements-3-13.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
