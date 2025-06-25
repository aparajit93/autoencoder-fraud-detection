import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.data.dataloader import load_data
from src.train import train_model

def main(args):
    X_train, X_val, _, _ = load_data(data_dir = args.data_dir, as_tensors=True)

    input_dim = X_train.shape[1]

    train_model(
        x_train = X_train,
        x_val = X_val,
        input_dim = input_dim,
        epochs = args.epochs,
        batch_size = args.batch_size,
        lr = args.learning_rate,
        model_dir = args.model_dir,
        filename = args.model_name
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train Autoencoder for Anomaly Detection')
    parser.add_argument('--data_dir', type = str, default = 'data/splits.pkl', help = 'Path to data')
    parser.add_argument('--epochs', type = int, default = 20, help = 'Number of training epochs')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'Training batch size')
    parser.add_argument('--learning_rate', type = float, default = 1e-3, help = 'Learning Rate')
    parser.add_argument('--model_dir', type = str, default = 'models', help = 'Path to save model')
    parser.add_argument('--model_name', type = str, default = 'autoencoder_model.pth', help = 'Model name to save as')

    args = parser.parse_args()
    main(args)