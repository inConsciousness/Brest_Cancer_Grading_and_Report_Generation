import os
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import keras

# ✅ Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ✅ Load model in backwards-compatible way for .h5 format
MODEL_PATH = './ann_model/saved_model/model_ann.h5'
with keras.utils.custom_object_scope({}):
    model = load_model(MODEL_PATH, compile=False)

# ✅ Initialize scaler (NOTE: In real deployment, load a saved scaler for consistency)
scaler = StandardScaler()

def predict(input_csv_path):
    """
    Predicts tumor severity using a pre-trained ANN model.

    Args:
        input_csv_path (str): Path to the input CSV file.

    Returns:
        List[Tuple[float, int]]: List of (confidence, binary label) predictions.
    """
    # Load and preprocess input
    input_df = pd.read_csv(input_csv_path)
    input_features = input_df.drop(['id', 'diagnosis'], axis=1, errors='ignore')

    # Scale features
    X_scaled = scaler.fit_transform(input_features)

    # Predict using ANN model
    predictions = model.predict(X_scaled)

    # Return (confidence, label) for each sample
    results = [(float(prob[0]), int(prob[0] > 0.5)) for prob in predictions]
    return results
