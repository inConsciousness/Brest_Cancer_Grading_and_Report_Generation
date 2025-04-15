import requests

OLLAMA_BASE_URL = "http://localhost:11434"

def is_ollama_model_available(model_name="phi"):
    try:
        res = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        res.raise_for_status()
        models = res.json().get("models", [])
        loaded_models = [m["name"] for m in models]
        return model_name in loaded_models
    except Exception as e:
        print(f"⚠️ Ollama model check failed: {str(e)}")
        return False
