import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import datetime
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

from data.data_utils import make_dataset_splits
from data.data_generator import DataGenerator
from models.de_convgraph_unet3d import DE_ConvGraph_Unet3D
from src.config import LOG_DIR

# Parse command-line arguments
parser = argparse.ArgumentParser(description='train the model')
parser.add_argument('--dim', type=int, nargs=3, default=[128,128,128])
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_channels_input', type=int, default=8)
parser.add_argument('--n_channels_output', type=int, default=1)
parser.add_argument('--shuffle', type=bool, default=False)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--model_name', type=str, choices=['de_convgraph_unet3d', 'unet3d'], required=True)
args = parser.parse_args()

# Set up logging
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.log')
logging.basicConfig(filename=log_file, level=logging.INFO)

# Metrics
metrics = [MeanAbsoluteError(), MeanAbsolutePercentageError()]

# Train-Valid-Test Split
partition = make_dataset_splits()

# Training mode
# Parameters
params = {'dim': tuple(args.dim),
            'batch_size': args.batch_size,
            'n_channels_input': args.n_channels_input,
            'n_channels_output': args.n_channels_output,
            'shuffle': args.shuffle}


# K-Fold Cross Validation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)

fold_no = 1
for train, test in kfold.split(partition['train'], partition['validation']):
    # Generators
    training_generator = DataGenerator(train, **params)
    validation_generator = DataGenerator(test, **params)

    # Instantiating the model
    model = DE_ConvGraph_Unet3D()

    # Model Compilation with optimizer, loss function, and metrics
    optimizer = Adam(learning_rate=args.learning_rate)
    loss_fn = MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        os.path.join(LOG_DIR, f'model_checkpoint_fold_{fold_no}.h5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        period=5
    )

    # Training the model
    history = model.fit(training_generator, 
                        validation_data=validation_generator, 
                        epochs=10, 
                        callbacks=[checkpoint_callback]
    )

    # Generalization metrics
    scores = model.evaluate(validation_generator, verbose=0)
    logging.info(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

    fold_no = fold_no + 1

    # Save training & validation loss values for plotting later
    np.savez(os.path.join(log_dir, f'history_fold_{fold_no}.npz'), loss=history.history['loss'], val_loss=history.history['val_loss'])

logging.info('Training complete.')
