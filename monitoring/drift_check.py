import pandas as pd
import json
import numpy as np

DRIFT_THRESHOLD = 3.0  # Standard deviations

def load_training_stats(stats_path="monitoring/train_stats.json"):
    with open(stats_path, "r") as f:
        return json.load(f)

def detect_drift(input_csv_path, stats_path="monitoring/train_stats.json"):
    stats = load_training_stats(stats_path)
    df = pd.read_csv(input_csv_path)
    
    # Drop non-feature columns
    df = df.drop(['id', 'diagnosis'], axis=1, errors='ignore')
    
    drift_report = {}

    for col in df.columns:
        if col not in stats:
            drift_report[col] = "⚠️ Not found in training stats"
            continue

        test_mean = df[col].mean()
        train_mean = stats[col]['mean']
        train_std = stats[col]['std']

        if train_std == 0:
            drift_report[col] = "⚠️ Zero std in training data"
            continue

        z_score = abs(test_mean - train_mean) / train_std
        if z_score > DRIFT_THRESHOLD:
            drift_report[col] = f"🚨 Drift detected (z-score={z_score:.2f})"
        else:
            drift_report[col] = f"✅ No drift (z-score={z_score:.2f})"

    return drift_report

# For standalone testing
if __name__ == "__main__":
    report = detect_drift("data/uploaded_input.csv")
    for feature, status in report.items():
        print(f"{feature}: {status}")
