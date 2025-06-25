import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from src.models.autoencoder import Autoencoder

def predict_anomalies(input_path, model_path, threshold_path, output_path = None):
    # Load in data
    df = pd.read_csv(input_path)
    X = torch.tensor(df.values, dtype = torch.float32)

    input_dim = X.shape[1]

    # Load model
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(model_path, weights_only = True))
    model.eval()

    with torch.no_grad():
        outputs = model(X)
        mse = F.mse_loss(outputs, X, reduction='none').mean(dim=1)
        anomaly_scores = mse.numpy()

    # Load threshold
    with open(threshold_path, 'r') as f:
        threshold = float(f.read().strip())
    
    # Normalize anomaly scores for confidence
    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    if max_score - min_score > 0:
        confidence_scores = (anomaly_scores - min_score) / (max_score - min_score)
    else:
        confidence_scores = np.zeros_like(anomaly_scores)
    
    # Predict labels
    predictions = (anomaly_scores > threshold).astype(int)

    print(f"Prediction complete. {predictions.sum()} anomalies found out of {len(predictions)} samples.")

    # Save results (optional)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df = df.copy()
        result_df['anomaly_score'] = anomaly_scores
        result_df['confidence_score'] = confidence_scores
        result_df['is_anomaly'] = predictions
        result_df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")

    return predictions, anomaly_scores    