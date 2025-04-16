from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import traceback
from PIL import Image

from ann_model.predict_ann import predict
from ann_model.image_feature_extractor import extract_features_from_image
from llm_report.report_generator import generate_report
from monitoring.drift_check import detect_drift

# Prometheus metrics route
from monitoring import metrics_exporter

app = FastAPI(title="The ANN Project - Breast Cancer Severity Predictor")
app.include_router(metrics_exporter.router)

templates = Jinja2Templates(directory="ui/templates")

@app.get("/")
def root():
    return {"message": "Welcome to The ANN Project API"}

@app.post("/predict/")
async def predict_from_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        file_path = "./data/uploaded_input.csv"
        df.to_csv(file_path, index=False)

        results = predict(file_path)

        response = []
        for i, (prob, label) in enumerate(results):
            report = generate_report(prob, label)
            response.append({
                "row": i + 1,
                "prediction": "Malignant" if int(label) == 1 else "Benign",
                "confidence": round(prob * 100, 2),
                "report": report
            })

        return {"status": "success", "results": response}

    except Exception as e:
        print("❌ ERROR in /predict endpoint:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Failed to process CSV file.", "details": str(e)},
        )

@app.post("/predict-image/")
async def predict_from_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        try:
            Image.open(io.BytesIO(contents)).verify()
        except Exception as img_err:
            raise ValueError(f"Uploaded file is not a valid image: {img_err}")

        df = extract_features_from_image(contents)
        image_csv_path = "./data/image_input_converted.csv"
        df.to_csv(image_csv_path, index=False)

        results = predict(image_csv_path)

        response = []
        for i, (prob, label) in enumerate(results):
            report = generate_report(prob, label)
            response.append({
                "row": i + 1,
                "prediction": "Malignant" if int(label) == 1 else "Benign",
                "confidence": round(prob * 100, 2),
                "report": report
            })

        return {"status": "success", "source": file.filename, "results": response}

    except Exception as e:
        print("❌ ERROR in /predict-image endpoint:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Failed to process image", "details": str(e)},
        )

@app.get("/drift-ui")
def render_drift_dashboard(request: Request):
    try:
        drift = detect_drift("data/uploaded_input.csv")
        return templates.TemplateResponse("drift.html", {"request": request, "drift": drift})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Could not generate drift report: {str(e)}"},
        )
