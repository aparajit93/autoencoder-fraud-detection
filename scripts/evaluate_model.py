import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.evaluate import evaluate_model

def main(args):
    evaluate_model(
        model_path = args.model_path,
        threshold_save_path = args.threshold_path,
        plot_dir = args.plot_dir,
        data_dir = args.data_dir
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Evaluate trained autoencoder and find best threshold')
    parser.add_argument('--model_path', type = str, default = 'models/autoencoder_model.pth', help = 'Path to model')
    parser.add_argument('--threshold_path', type=str, default='models/best_threshold.txt', help='Path to save threshold')
    parser.add_argument('--plot_dir', type = str, default = 'plots', help = 'Directory to save plots')
    parser.add_argument('--data_dir', type = str, default = 'data/splits.pkl', help = 'Path to data')

    args = parser.parse_args()
    main(args)