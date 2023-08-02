# Radiotherapy Dose Opitmisation

This repository contains code for training and testing deep learning approaches to radiotherapy treatment plan optimisation in oropharyngeal cancer.

## Obtaining the OpenKBP-Opt Dataset
The OpenKBP-Opt Dataset is the benchmark used to train the models and is available for download on the [official OpenKBP-Opt Github Repository]( https://github.com/ababier/open-kbp-opt/tree/master). Follow these steps to download and extract the dataset:

The OpenKBP-Opt dataset contains the following subdirectories:

- `reference-plans`: All the reference patient data (e.g., reference dose, contours) are stored here
- `paper-predictions`: The predictions that were generated during the OpenKBP Grand Challenge are stored here.

Assign the path to the root folder containing the above two folders to `DATA_PATH` variable in the `src/configs.py` file

## Repository Structure

This repository is organized as follows:

- `data/`: This directory contains the code for pre-processing the data.
    - `data_generator.py`: Custom keras DataGenerator for loading 3D volumetric data and corresponding output labels for training a neural network.
    - `data_utils.py`: Contains utility functions used in the DataGenerator class
- `modesl/`: This directory contains the different model definitions
- `src/`: This directory contains the scirpts for training and evaluating the models
    - `config.py`:  Used to setup path variables used throughout the project
    - `tran_model.py`: Script for training a model
    - `evaluate_model.py`: Script for evaluating a trained model

## Requirements

- Python 3.6 or above
- TensorFlow 2.12.0
- Keras 2.12.0
- Spektral 1.3.0
- Numpy 1.22.4

## Usage

### Training the Model

To train the model, run the `train_model.py` script. The script accepts several command-line arguments:

- `--dim`: The dimensions of the input data. Defaults to `<128,128,128>`.
- `--batch_size`: The batch size for training. Defaults to `1`.
- `--n_channels_input`: The number of input channels. Defaults to `8`.
- `--n_channels_output`: The number of output channels. Defaults to `1`.
- `--shuffle`: Whether or not to shuffle the data. Defaults to `False`.
- `--learning_rate`: The learning rate for the Adam optimizer. Defaults to `1e-4`.
- `--model_name`: The name of the model to use. Must be either `de_convgraph_unet3d` or `unet3d`.

Example:

```bash
python train_model.py --dim 128 128 128 --batch_size 1 --n_channels_input 8 --n_channels_output 1 --shuffle False --learning_rate 0.0001 --model_name de_convgraph_unet3d
```
The script logs the training process and saves the model weights and training history after each fold of cross-validation.

### Testing the Model

To test the model, run the `evaluate_model.py` script. The script accepts several command-line arguments:

- `--dim`: The dimensions of the input data. Defaults to `<128,128,128>`.
- `--batch_size`: The batch size for testing. Defaults to `1`.
- `--n_channels_input`: The number of input channels. Defaults to `8`.
- `--n_channels_output`: The number of output channels. Defaults to `1`.
- `--shuffle`: Whether or not to shuffle the data. Defaults to `False`.
- `--model_path`: The path to the trained model. Required.

Example:

```bash
python evaluate_model.py --dim 128 128 128 --batch_size 1 --n_channels_input 8 --n_channels_output 1 --shuffle False --model_path /path/to/model_checkpoint.h5
```
The script logs the testing process and outputs the model's performance on the test data.

## Note
Make sure to adjust the command-line arguments as per your data and hardware capabilities (e.g., GPU memory for batch size).