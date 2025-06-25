import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.models.autoencoder import Autoencoder
from src.data.dataloader import load_data

def evaluate_model(model_path, threshold_save_path, plot_dir = '../plots', data_dir = '../data/splits.pkl'):
    _, _, x_test, y_test = load_data(data_dir = data_dir, as_tensors = True)

    # Load Model
    input_dim = x_test.shape[1]
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    test_loader = torch.utils.data.DataLoader(x_test, batch_size=64, shuffle=False)

    anomaly_scores = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch)
            mse = torch.mean((outputs - batch)**2, dim = 1)
            anomaly_scores.append(mse.numpy())
        
    anomaly_scores = np.concatenate(anomaly_scores)

    y_test = np.array(y_test)

    thresholds = np.linspace(min(anomaly_scores), max(anomaly_scores), 100)

    best_f1 = 0
    best_threshold = thresholds[0]

    for thresh in thresholds:
        y_pred = (anomaly_scores > thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division = 0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
        
    os.makedirs(os.path.dirname(threshold_save_path), exist_ok=True)
    with open(threshold_save_path, 'w') as f:
        f.write(f"{best_threshold:.10f}\n")
    
    precision = precision_score(y_test, anomaly_scores > best_threshold, zero_division=0)
    recall = recall_score(y_test, anomaly_scores > best_threshold, zero_division=0)
    roc_auc = roc_auc_score(y_test, anomaly_scores)

    # Split scores by true label
    normal_scores = anomaly_scores[y_test == 0]
    anomalous_scores = anomaly_scores[y_test == 1]

    # Plot histogram
    plt.figure(figsize = (6, 4.5), dpi = 150)
    #plt.hist(normal_scores, bins = 50, alpha = 0.6, label = 'Normal', color = 'green')
    plt.hist(anomalous_scores, bins = 50, alpha = 0.6, label = 'Anomaly', color = 'red')
    #plt.axvline(x = best_threshold, color = 'blue', linestyle = '--', label = 'Threshold')

    plt.autoscale(enable=True, axis='x', tight=True)

    plt.xlabel('Anomaly Score (MSE)', fontsize = 12)
    plt.ylabel('Frequency', fontsize = 12)
    plt.title('Anomaly Score Distribution', fontsize = 13, weight = 'bold')
    plt.legend()
    plt.grid(True, linestyle = '--', alpha = 0.6)
    plt.tight_layout()

    # Save histogram
    hist_path = f'{plot_dir}/best_threshold_score_hist.png'
    plt.savefig(hist_path, dpi = 200, bbox_inches = 'tight')
    plt.close()

    print("==== Best Threshold Results ====")
    print(f"Best threshold    : {best_threshold:.5f}")
    print(f"F1 Score          : {best_f1:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"ROC AUC           : {roc_auc:.4f}")
    print(f"Histogram saved to: {hist_path}")

    return best_threshold, anomaly_scores, y_test