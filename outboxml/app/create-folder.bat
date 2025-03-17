@echo off
set "base_path=%~dp0"

mkdir "%base_path%artifacts"
mkdir "%base_path%grafana-data"
mkdir "%base_path%fastapi"
mkdir "%base_path%jupyter-workspace"