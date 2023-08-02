from os import environb
import os
import numpy as np
import tensorflow as tf

from src.config import DATA_PATH
from data.data_utils import populate_3d_volume

class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for loading 3D volumetric data and corresponding output labels for training a neural network.This generator is designed to work with a list of patient IDs and loads the necessary input and output volumes for each patient in batches.

    Parameters:
        list_IDs (list): List of patient IDs to generate data for.
        batch_size (int): Number of samples in each batch (default is 1).
        dim (tuple): Dimensions of the input 3D volumes (default is (128, 128, 128)).
        n_channels_input (int): Number of input channels in the input 3D volumes (default is 8).
        n_channels_output (int): Number of output channels in the output 3D volumes (default is 1).
        shuffle (bool): If True, shuffle the data after each epoch (default is False).
    """

    def __init__(self, list_IDs, batch_size=1, dim=(128,128,128),
                 n_channels_input=8, n_channels_output=1, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels_input = n_channels_input
        self.n_channels_output = n_channels_output
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        print(indexes)

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        print(list_IDs_temp)

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        volume_shape = self.dim

        # Initialization
        X_graph = np.empty((self.batch_size, 2097152, self.n_channels_input))
        X_adj = [None]*self.batch_size
        y = np.empty((self.batch_size, *self.dim, self.n_channels_output))

        roi_to_label = {'ptv56' : 1, 
                          'ptv63' : 2, 
                          'ptv70' : 3, 
                          'brainstem' : 4, 
                          'leftparotid' : 5, 
                          'rightparotid' : 6, 
                          'spinalcord' : 7}


        # Generate X data
        for i, ID in enumerate(list_IDs_temp):
          
          #Predicted Dose
          patient_volume = np.zeros(volume_shape)
          patient_dose_path = f'{DATA_PATH}/paper-predictions/set_1/pt_{ID}.csv'
          if os.path.exists(patient_dose_path):
            with open(patient_dose_path, 'r') as f:
                next(f)
                for line in f:
                    voxel_id, dose_value = line.split(',')
                    voxel_coord = np.unravel_index(int(voxel_id), volume_shape)
                    patient_volume[voxel_coord] = float(dose_value)

          #PTV56
          ptv56 = np.zeros(volume_shape)
          ptv56_path = f'{DATA_PATH}/reference-plans/pt_{ID}/PTV56.csv'
          populate_3d_volume(ptv56, volume_shape, ptv56_path, roi_to_label['ptv56'])

          #PTV63
          ptv63 = np.zeros(volume_shape)
          ptv63_path = f'{DATA_PATH}/reference-plans/pt_{ID}/PTV63.csv'
          populate_3d_volume(ptv63, volume_shape, ptv63_path, roi_to_label['ptv63'])

          #PTV70
          ptv70 = np.zeros(volume_shape)
          ptv70_path = f'{DATA_PATH}/reference-plans/pt_{ID}/PTV70.csv'
          populate_3d_volume(ptv70, volume_shape, ptv70_path, roi_to_label['ptv70'])

          #Brainstem
          brainstem = np.zeros(volume_shape)
          brainstem_path = f'{DATA_PATH}/reference-plans/pt_{ID}/Brainstem.csv'
          populate_3d_volume(brainstem, volume_shape, brainstem_path, roi_to_label['brainstem'])

          #LeftParotid
          leftparotid = np.zeros(volume_shape)
          leftparotid_path = f'{DATA_PATH}/reference-plans/pt_{ID}/LeftParotid.csv'
          populate_3d_volume(leftparotid, volume_shape, leftparotid_path, roi_to_label['leftparotid'])

          #RightParotid
          rightparotid = np.zeros(volume_shape)
          rightparotid_path = f'{DATA_PATH}/reference-plans/pt_{ID}/RightParotid.csv'
          populate_3d_volume(rightparotid, volume_shape, rightparotid_path, roi_to_label['rightparotid'])

          #Spinal Cord
          spinalcord = np.zeros(volume_shape)
          spinalcord_path = f'{DATA_PATH}/reference-plans/pt_{ID}/SpinalCord.csv'
          populate_3d_volume(spinalcord, volume_shape, spinalcord_path, roi_to_label['spinalcord'])

          #Constructing Graph Representation
          input_volume_graph = np.zeros((128*128*128, 8))

          # Iterate over the indices of the input volumes
          for p in range(128):
              for q in range(128):
                  for r in range(128):
                      input_volume_graph[p*128*128 + q*128 + r, 0] = patient_volume[p, q, r]
                      input_volume_graph[p*128*128 + q*128 + r, 1] = ptv56[p, q, r]
                      input_volume_graph[p*128*128 + q*128 + r, 2] = ptv63[p, q, r]
                      input_volume_graph[p*128*128 + q*128 + r, 3] = ptv70[p, q, r]
                      input_volume_graph[p*128*128 + q*128 + r, 4] = brainstem[p, q, r]
                      input_volume_graph[p*128*128 + q*128 + r, 5] = leftparotid[p, q, r]
                      input_volume_graph[p*128*128 + q*128 + r, 6] = rightparotid[p, q, r]
                      input_volume_graph[p*128*128 + q*128 + r, 7] = spinalcord[p, q, r]

          #Adjacency Matrix
          indices = np.load(f'{DATA_PATH}/adj_mat_folder/indices.npy')
          values = np.load(f'{DATA_PATH}/adj_mat_folder/values.npy')
          dense_shape = np.load(f'{DATA_PATH}/dense_shape.npy')

          adjacency_matrix_sparse = tf.sparse.SparseTensor(indices, values, dense_shape)

          # Optimised Dose
          output_volume = np.zeros(volume_shape)
          patient_ouput_dose_path = f'{DATA_PATH}/reference-plans/pt_{ID}/dose.csv'
          with open(patient_ouput_dose_path, 'r') as f:
              #skip column names row
              next(f)
              for line in f:
                  voxel_id, dose_value = line.split(',')
                  voxel_coord = np.unravel_index(int(voxel_id), volume_shape)
                  output_volume[voxel_coord] = float(dose_value)

          output_volume = np.reshape(output_volume, (128, 128, 128, -1))

          X_graph[i] = input_volume_graph
          X_adj[i] = adjacency_matrix_sparse
          y[i] = output_volume

        return [X_graph, X_adj], y