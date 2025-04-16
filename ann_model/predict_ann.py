import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ✅ Suppress TensorFlow log verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ✅ Load model once globally to avoid reloading on each request
MODEL_PATH = './ann_model/saved_model/model_ann.h5'
model = load_model(MODEL_PATH, compile=False)

# ✅ Initialize scaler (NOTE: In real deployment, load a saved scaler to maintain consistency)
scaler = StandardScaler()

def predict(input_csv_path):
    """
    Predicts tumor severity using a pre-trained ANN model.

    Args:
        input_csv_path (str): Path to the input CSV file.

    Returns:
        List[Tuple[float, int]]: List of (confidence, binary label) predictions.
    """
    # Read CSV and drop unnecessary columns if they exist
    input_df = pd.read_csv(input_csv_path)
    input_features = input_df.drop(['id', 'diagnosis'], axis=1, errors='ignore')

    # Scale the input features
    X_scaled = scaler.fit_transform(input_features)

    # Perform model prediction
    predictions = model.predict(X_scaled)

    # Return list of (probability, label)
    results = [(float(prob[0]), int(prob[0] > 0.5)) for prob in predictions]
    return results
