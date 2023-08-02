from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Concatenate
from tensorflow.keras.models import Model

def UNet(input_channels=8, output_channels=1):
    """
    Construct a 3D U-Net model for voxel wise regression.

    The U-Net architecture consists of an encoder-decoder structure that aims to perform voxel wise regression. The model takes a 5D input tensor representing the volumetric patient data (OAR and predicted dose) and produces a 5D output tensor representing the optimised dose.

    Parameters:
        input_channels (int): Number of input channels for the input 3D volumes.
        output_channels (int): Number of output channels for the segmented 3D volumes.

    Returns:
        tf.keras.Model: The constructed 3D U-Net model.
    """

    # Input layer
    inputs = Input(shape=(128, 128, 128, input_channels))

    # Define the contracting path (encoder)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    # Bottleneck layer
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

    # Expanding path (decoder)
    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    up6 = Conv3D(256, (2, 2, 2), activation='relu', padding='same')(up6)
    merge6 = Concatenate(axis=-1)([conv4, up6])
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    up7 = Conv3D(128, (2, 2, 2), activation='relu', padding='same')(up7)
    merge7 = Concatenate(axis=-1)([conv3, up7])
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = UpSampling3D(size=(2, 2, 2))(conv7)
    up8 = Conv3D(64, (2, 2, 2), activation='relu', padding='same')(up8)
    merge8 = Concatenate(axis=-1)([conv2, up8])
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = UpSampling3D(size=(2, 2, 2))(conv8)
    up9 = Conv3D(32, (2, 2, 2), activation='relu', padding='same')(up9)
    merge9 = Concatenate(axis=-1)([conv1, up9])
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(merge9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

    #Output Layer
    outputs = Conv3D(output_channels, (1, 1, 1), activation='linear', padding='same')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model