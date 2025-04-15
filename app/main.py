from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
import traceback
from PIL import Image
from ann_model.predict_ann import predict
from llm_report.report_generator import generate_report
from ann_model.image_feature_extractor import extract_features_from_image

app = FastAPI(title="The ANN Project - Brest Cancer Severity Predictor")

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
        return {
            "status": "error",
            "message": "Failed to process CSV file.",
            "details": str(e)
        }

@app.post("/predict-image/")
async def predict_from_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Validate image
        try:
            Image.open(io.BytesIO(contents)).verify()
        except Exception as img_err:
            raise ValueError(f"Uploaded file is not a valid image: {img_err}")

        # Extract features and save to CSV
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

        return {
            "status": "success",
            "source": file.filename,
            "results": response
        }

    except Exception as e:
        print("❌ ERROR in /predict-image endpoint:")
        traceback.print_exc()
        return {
            "status": "error",
            "message": "Failed to process image",
            "details": str(e)
        }
