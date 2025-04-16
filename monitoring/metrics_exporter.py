from fastapi import APIRouter
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from monitoring.drift_check import detect_drift

# Prometheus metrics
drift_metric = Gauge("feature_drift", "Z-score of feature drift", ["feature"])

router = APIRouter()

@router.get("/metrics")
def metrics_endpoint():
    try:
        drift_report = detect_drift("data/uploaded_input.csv")

        for feature, status in drift_report.items():
            if "z-score=" in status:
                z = float(status.split("z-score=")[-1].replace(")", "").strip())
                drift_metric.labels(feature=feature).set(z)

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        return Response(content=f"# Error generating metrics\n# {str(e)}", media_type="text/plain")
