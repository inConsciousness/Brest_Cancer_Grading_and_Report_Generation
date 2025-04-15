import requests
import json

def generate_report(probability, label):
    """
    Generates a medical-style report using the prediction via Ollama + Phi.

    Args:
        probability (float): Confidence score from ANN (e.g., 0.91)
        label (int): 1 for malignant, 0 for benign

    Returns:
        str: Natural language medical report
    """
    condition = "malignant" if label == 1 else "benign"
    confidence_percent = round(probability * 100, 2)

    prompt = (
        f"A patient's tumor is predicted to be {condition} with a confidence of {confidence_percent}%. "
        f"Write a short, clear medical report for the doctor. Include: "
        f"1. Tumor condition\n"
        f"2. Possible medical implications\n"
        f"3. Recommended next steps\n"
        f"Keep the tone professional and suitable for a clinical setting."
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )

        # 🧪 Debug: print raw response safely
        print("🧪 Ollama raw content:", response.content.decode("utf-8", errors="ignore"))

        # Handle case where response contains multiple JSON objects
        lines = response.content.decode("utf-8", errors="ignore").splitlines()
        combined = "".join(lines)
        first_json = json.loads(combined)

        return first_json.get("response", "").strip()

    except Exception as e:
        return f"⚠️ Report generation failed: {str(e)}"
