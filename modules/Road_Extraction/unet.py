import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, SpatialDropout2D, Concatenate, Conv2DTranspose, BatchNormalization, Activation

def conv_block(x, filters, dropout_rate=0.0):
    """
    Standard U-Net convolution block: Conv -> BN -> ReLU.
    Using SpatialDropout2D to prevent fragmenting thin line features.
    """
    # First Convolution
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second Convolution
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Apply Spatial Dropout if specified
    if dropout_rate > 0:
        x = SpatialDropout2D(dropout_rate)(x)

    return x

def unet(input_shape, output_layer=1, start_filters=32):
    """
    Optimized U-Net for thin-structure segmentation.
    Base filters increased to 32 for better low-level feature extraction.
    """
    inputs = Input(input_shape)

    # -----------------------------------------
    # ENCODER (Downsampling)
    # -----------------------------------------
    conv1 = conv_block(inputs, start_filters, dropout_rate=0.1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, start_filters * 2, dropout_rate=0.1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, start_filters * 4, dropout_rate=0.2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, start_filters * 8, dropout_rate=0.2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # -----------------------------------------
    # BOTTLENECK
    # -----------------------------------------
    conv5 = conv_block(pool4, start_filters * 16, dropout_rate=0.3)

    # -----------------------------------------
    # DECODER (Upsampling & Concatenation)
    # -----------------------------------------
    up6 = Conv2DTranspose(start_filters * 8, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = Concatenate()([conv4, up6])
    conv6 = conv_block(merge6, start_filters * 8, dropout_rate=0.2)

    up7 = Conv2DTranspose(start_filters * 4, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = Concatenate()([conv3, up7])
    conv7 = conv_block(merge7, start_filters * 4, dropout_rate=0.2)

    up8 = Conv2DTranspose(start_filters * 2, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = Concatenate()([conv2, up8])
    conv8 = conv_block(merge8, start_filters * 2, dropout_rate=0.1)

    up9 = Conv2DTranspose(start_filters, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = Concatenate()([conv1, up9])
    conv9 = conv_block(merge9, start_filters, dropout_rate=0.1)

    # -----------------------------------------
    # OUTPUT
    # -----------------------------------------
    output = Conv2D(output_layer, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=output)
    return model