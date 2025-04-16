import requests
import hashlib
import json

# ✅ In-memory cache to avoid redundant generation
report_cache = {}

def generate_report(probability, label):
    """
    Generates a medical-style report using prediction results via Ollama (Phi model).
    
    Uses a hash-based cache to avoid regenerating the same prompt.

    Args:
        probability (float): Model confidence score (0 to 1)
        label (int): Prediction label (1 for malignant, 0 for benign)

    Returns:
        str: Generated clinical-style report or fallback error message
    """
    condition = "malignant" if label == 1 else "benign"
    confidence_percent = round(probability * 100, 3)

    prompt = (
        f"A patient's tumor is predicted to be {condition} with a confidence of {confidence_percent}%. "
        f"Write a short, clear medical report for the doctor. Include:\n"
        f"1. Tumor condition\n"
        f"2. Possible medical implications\n"
        f"3. Recommended next steps\n"
        f"Keep the tone professional and suitable for a clinical setting."
    )

    # 🔐 Generate unique hash key for this prompt
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

    # ✅ Return cached result if exists
    if prompt_hash in report_cache:
        return report_cache[prompt_hash]

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

        # 🧪 Debug log: view raw response
        raw_text = response.content.decode("utf-8", errors="ignore")
        print("🧪 Ollama raw content:", raw_text)

        # 🔍 Load and extract from JSON
        parsed = json.loads(raw_text)
        report = parsed.get("response", "").strip()

        # 🧠 Store in cache
        report_cache[prompt_hash] = report

        return report

    except Exception as e:
        return f"⚠️ Report generation failed: {str(e)}"
