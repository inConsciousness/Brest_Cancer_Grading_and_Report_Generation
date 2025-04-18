import os
import pandas as pd
import numpy as np
import keras
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ✅ Suppress TensorFlow logs (INFO & WARNINGS)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ✅ Backward-compatible .h5 model loading
MODEL_PATH = './ann_model/saved_model/model_ann.h5'
with keras.utils.custom_object_scope({}):
    model = load_model(MODEL_PATH, compile=False)

# ✅ Scaler (NOTE: Use saved scaler.pkl in prod)
scaler = StandardScaler()

def predict(input_csv_path):
    """
    Predicts tumor severity using a pre-trained ANN model.

    Args:
        input_csv_path (str): Path to the input CSV file.

    Returns:
        List[Tuple[float, int]]: (confidence, label) per sample.
    """
    # Load CSV, drop irrelevant columns
    df = pd.read_csv(input_csv_path)
    features = df.drop(['id', 'diagnosis'], axis=1, errors='ignore')

    # Standardize features
    X_scaled = scaler.fit_transform(features)

    # Predict probabilities
    probs = model.predict(X_scaled)

    # Convert to (confidence, binary label)
    return [(float(p[0]), int(p[0] > 0.5)) for p in probs]
