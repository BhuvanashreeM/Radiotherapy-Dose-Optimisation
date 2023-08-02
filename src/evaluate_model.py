import argparse
import os
import logging
import datetime

import tensorflow as tf

from data.data_generator import DataGenerator
from data.data_utils import make_dataset_splits
from src.config import LOG_DIR

# Parse command-line arguments
parser = argparse.ArgumentParser(description='test the model')
parser.add_argument('--dim', type=int, nargs=3, default=[128,128,128])
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_channels_input', type=int, default=8)
parser.add_argument('--n_channels_output', type=int, default=1)
parser.add_argument('--shuffle', type=bool, default=False)
parser.add_argument('--model_path',type=str, required=True)
args = parser.parse_args()

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.log')
logging.basicConfig(filename=log_file, level=logging.INFO)


# Testing mode
# Load trained model
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"The file '{args.model_path}' does not exist.")
model = tf.keras.models.load_model(args.model_path)

# Parameters for testing
params = {'dim': tuple(args.dim),
            'batch_size': args.batch_size,
            'n_channels_input': args.n_channels_input,
            'n_channels_output': args.n_channels_output,
            'shuffle': args.shuffle}

partition = make_dataset_splits()

# Generators
testing_generator = DataGenerator(partition['test'], **params)

# Evaluate the model
scores = model.evaluate(testing_generator, verbose=0)
logging.info(f'Test score: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
logging.info('Testing complete.')