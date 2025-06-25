import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.predict import predict_anomalies

def main(args):
    predict_anomalies(
        input_path= args.input_path,
        model_path= args.model_path,
        threshold_path= args.threshold_path,
        output_path= args.output_path
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection on new data")
    parser.add_argument('--input_path', type=str, required=True, help='Path to CSV file with new data (no labels)')
    parser.add_argument('--model_path', type=str, default='models/autoencoder_model.pth', help='Path to trained model')
    parser.add_argument('--threshold_path', type=str, default='models/best_threshold.txt', help='Path to threshold file')
    parser.add_argument('--output_path', type=str, default='results/predictions.csv', help='Path to save output predictions')

    args = parser.parse_args()
    main(args)