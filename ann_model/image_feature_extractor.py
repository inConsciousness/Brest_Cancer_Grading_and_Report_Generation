import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

# Breast cancer feature columns (excluding 'id', 'diagnosis', 'Unnamed: 32')
COLUMNS = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
    'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
    'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
]

def extract_features_from_image(image_bytes):
    """
    Converts image to grayscale, extracts intensity-based patterns
    (Currently mocked to produce a vector of 30 synthetic features)
    """

    image = Image.open(BytesIO(image_bytes)).convert("L")  # grayscale
    image = image.resize((64, 64))  # resize for consistency

    pixel_data = np.array(image)

    # Simulated features from image stats (mock — replace later)
    mean_val = np.mean(pixel_data)
    std_val = np.std(pixel_data)
    min_val = np.min(pixel_data)
    max_val = np.max(pixel_data)

    # Create 30 float features based on pixel intensity variations
    features = np.linspace(mean_val + std_val, min_val + max_val, num=30)

    # Return as 1-row DataFrame matching ANN input
    return pd.DataFrame([features], columns=COLUMNS)
