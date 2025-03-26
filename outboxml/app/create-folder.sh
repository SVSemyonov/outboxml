#!/bin/bash

base_path=$(dirname "$0")/

mkdir -p "${base_path}artifacts"
chmod 777 "${base_path}artifacts"

mkdir -p "${base_path}grafana-data"
chmod 777 "${base_path}grafana-data"

mkdir -p "${base_path}fastapi"
chmod 777 "${base_path}fastapi"

mkdir -p "${base_path}jupyter-workspace"
chmod 777 "${base_path}jupyter-workspace"