@echo off
cd /d E:\ANN Project

REM ANN model directory
mkdir ann_model\saved_model

REM LLM Report generator
mkdir llm_report\templates

REM FastAPI App
mkdir app

REM Frontend UI
mkdir ui

REM ArgoCD and Kubernetes
mkdir deployments\argo
mkdir deployments\k8s

REM Monitoring and observability
mkdir monitoring

REM Create empty essential files
type nul > ann_model\train_ann.py
type nul > ann_model\predict_ann.py
type nul > llm_report\report_generator.py
type nul > app\main.py
type nul > app\schemas.py
type nul > app\utils.py
type nul > ui\dashboard.py
type nul > monitoring\drift_check.py
type nul > monitoring\metrics_exporter.py
type nul > deployments\argo\ann-pipeline.yaml
type nul > deployments\argo\llm-report-pipeline.yaml
type nul > deployments\k8s\deployment.yaml
type nul > deployments\k8s\service.yaml
type nul > Dockerfile
type nul > requirements.txt
type nul > .gitignore
type nul > README.md

echo.
echo ✅ Project structure created successfully under E:\ANN Project!
pause
