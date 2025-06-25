# Credit Card Anomaly Detector

This project implements an **autoencoder-based anomaly detection system** for detecting fraudulent credit card transactions.  
Using unsupervised deep learning, the model learns to reconstruct normal transaction patterns and flags deviations as anomalies.

---

## Project Overview

- Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Model: Fully connected Autoencoder  
- Training: On normal (non-fraud) transactions only  
- Evaluation: Reconstruction error thresholding for anomaly detection
- Fully modular training, evaluation, and prediction scripts
- Threshold tuning using F1 score maximization
- Precision-Recall and Anomaly Score visualizations
- Confidence scoring for anomaly predictions
- Clean CLI-style driver scripts (train/eval/predict)
- Tested on CPU-friendly environments (no CUDA required)

---

## Project Structure

```bash
anomaly-detection/
├── data/                    # input CSVs, train/test splits
├── models/                  # saved PyTorch model + threshold
├── results/                 # output predictions
├── notebooks/               # development notebooks
├── scripts/                 # driver scripts
│   ├── train_model_driver.py
│   ├── evaluate_model_driver.py
│   └── predict_driver.py
├── src/
│   ├── train.py             # training logic
│   ├── evaluate.py          # evaluation logic
│   ├── predict.py           # prediction logic
│   ├── models/
│   │   └── autoencoder.py   # model definition
│   └── data/
│       └── dataloader.py    # data I/O utilities
├── environment.yml
└── README.md
```

## How to Run
1. Set up environment
```bash
conda env create -f environment.yml
conda activate fraud-detection
```
2. Train Model
```bash
python scripts/train_model.py
```
3. Evaluate and Find Threshold
```bash
python scripts/evaluate_model.py
```
4. Predict on New Data
```bash
python scripts/predict_driver.py \
  --input_path data/new_samples.csv \
  --output_path results/predictions.csv
```