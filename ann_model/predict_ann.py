import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ✅ Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ✅ Load model once (not repeatedly)
MODEL_PATH = './ann_model/saved_model/model_ann.h5'
model = load_model(MODEL_PATH, compile=False)

# ✅ Use a reusable scaler – normally, this should be saved during training for consistency
scaler = StandardScaler()

def predict(input_csv_path):
    """
    Predicts tumor severity using a pre-trained ANN model.

    Args:
        input_csv_path (str): Path to the input CSV file.

    Returns:
        List of tuples: (probability, binary label)
    """
    input_df = pd.read_csv(input_csv_path)

    # Drop irrelevant columns if present
    input_features = input_df.drop(['id', 'diagnosis'], axis=1, errors='ignore')

    # Standardize input features (NOTE: in real-world, use saved scaler)
    X_scaled = scaler.fit_transform(input_features)

    # Predict probabilities and apply threshold
    predictions = model.predict(X_scaled)
    results = [(float(prob[0]), int(prob[0] > 0.5)) for prob in predictions]

    return results
