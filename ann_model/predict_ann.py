import os
import pandas as pd
import numpy as np
from keras.saving import load_model  # ✅ For Keras 3.x compatibility
from sklearn.preprocessing import StandardScaler

# ✅ Suppress TensorFlow logs (INFO & WARNINGS)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ✅ Model path
MODEL_PATH = './ann_model/saved_model/model_ann.h5'

# ✅ Load pre-trained model (Keras 3 compatible)
model = load_model(MODEL_PATH, compile=False)

# ✅ Initialize standard scaler (NOTE: In production, load a saved scaler instead)
scaler = StandardScaler()

def predict(input_csv_path):
    """
    Predicts tumor severity using a pre-trained ANN model.

    Args:
        input_csv_path (str): Path to the input CSV file.

    Returns:
        List[Tuple[float, int]]: (confidence, label) per sample.
    """
    # ✅ Load CSV and remove irrelevant columns
    df = pd.read_csv(input_csv_path)
    features = df.drop(['id', 'diagnosis'], axis=1, errors='ignore')

    # ✅ Standardize features
    X_scaled = scaler.fit_transform(features)

    # ✅ Predict probabilities using ANN
    probs = model.predict(X_scaled)

    # ✅ Format predictions into (confidence, binary label)
    return [(float(p[0]), int(p[0] > 0.5)) for p in probs]
