import os
import numpy as np

def make_dataset_splits():
    """
    Splits the list of patient IDs into train, validation, and test sets.

    Returns:
        dict: A dictionary containing the partitioned patient IDs, with keys 'train', 'validation', and 'test'.
              Each key maps to a list of patient IDs for the corresponding split.
    """
    patient_ids = list(range(241, 341))

    num_patients = len(patient_ids)
    split1_size = int(num_patients * 0.8) # train
    split2_size = int(num_patients * 0.1) # test
    # rest for validation

    train_IDs = patient_ids[:split1_size]
    validation_IDs = patient_ids[split1_size:(split1_size+split2_size)]
    test_IDs = patient_ids[(split1_size+split2_size):]

    partition = {'train': train_IDs, 'validation': validation_IDs, 'test': test_IDs}

    return partition

def populate_3d_volume(volume, volume_shape, file_path, label):
    """
    Populates a 3D volume with a specific label based on the information provided in a CSV file. The CSV file contains voxel IDs and their corresponding labels. The function reads the CSV file and assigns the specified label to the corresponding voxel in the 3D volume.

    Parameters:
        volume (ndarray): The 3D volume to populate with labels.
        volume_shape (tuple): The dimensions of the 3D volume in the format (depth, height, width).
        file_path (str): The path to the CSV file containing voxel IDs and labels.
        label (int or float): The label to assign to the voxels in the volume.
    """

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            next(f)
            for line in f:
                voxel_id = line.split(',')[0]
                voxel_coord = np.unravel_index(int(voxel_id), volume_shape)
                volume[voxel_coord] = label