
import tensorflow as tf
from tensorflow.keras.layers import UpSampling3D, Conv3D, Activation, Concatenate, Reshape, Permute, MaxPooling3D
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from spektral.layers import GCNConv

class FusionLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer that performs fusion of two input volumes.

    This layer takes two input volumes (vol1 and vol2) and applies element-wise fusion with a learnable weight
    'kernel'. The output fused volume is computed as kernel * vol1 + (1 - kernel) * vol2.

    Attributes:
        kernel (tf.Variable): Trainable weight variable for this layer representing the fusion weight.

    Methods:
        build: Builds the layer by adding the 'kernel' weight variable.
        call: Performs the fusion operation on the input volumes.
    """
    def __init__(self, **kwargs):
        super(FusionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
      self.kernel = self.add_weight(name='kernel',
                                    shape=(1,) + input_shape[0][1:],
                                    initializer='uniform',
                                    trainable=True)
      super(FusionLayer, self).build(input_shape)

    def call(self, x):
        vol1, vol2 = x
        fused_vol = self.kernel * vol1 + (1. - self.kernel) * vol2
        return fused_vol
    

def DE_ConvGraph_Unet3D():
    """
    Construct a 3D Convolutional Graph U-Net model for volumetric data.

    Returns:
        tf.keras.Model: The constructed 3D Convolutional Graph U-Net model.
    """
    # Graph Input
    volume_input = Input(shape=(128*128*128, 8))
    adjacency_input = Input(shape=(128*128*128, 128*128*128), sparse=True)

    # Getting 3D Volume
    conv1_transposed = tf.transpose(volume_input, perm=[0, 2, 1])
    conv1_reshaped = Reshape((8, 128, 128, 128))(conv1_transposed)
    conv1_permuted = Permute((2,3,4,1))(conv1_reshaped)
    three_dim_input = conv1_permuted

    # Encoder 1 Layers:
    conv1 = GCNConv(32)([volume_input, adjacency_input])
    conv1 = GCNConv(32)([conv1, adjacency_input])

    # Graph Encoder Output Transformation
    conv3_transposed = tf.transpose(conv1, perm=[0, 2, 1])
    conv3_reshaped = Reshape((32, 128, 128, 128))(conv3_transposed)
    conv3_permuted = Permute((2,3,4,1))(conv3_reshaped)
    gcn_output = conv3_permuted

    # Level 1
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(gcn_output)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    # Level 2
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    # Encoder 2 Layers:
    # Level 1
    conv1_enc2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(three_dim_input)
    pool1_enc2 = MaxPooling3D(pool_size=(2, 2, 2))(conv1_enc2)

    # Level 2
    conv2_enc2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1_enc2)
    pool2_enc2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2_enc2)

    # Level 3
    # Fusion Encoder 1 and Encoder 2
    fused = FusionLayer()([pool2, pool2_enc2])

    # Level 2
    up1 = Concatenate(axis=-1)([UpSampling3D(size=(2, 2, 2))(fused), conv2])
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up1)

    # Level 1
    up2 = Concatenate(axis=-1)([UpSampling3D(size=(2, 2, 2))(conv4), conv1])
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up2)

    outputs = Conv3D(1, (1, 1, 1), activation='linear')(conv5)

    model = Model(inputs=[volume_input, adjacency_input], outputs=outputs)
    return model