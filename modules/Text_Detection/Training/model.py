# models.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.initializers import Constant
from config import *

def build_center_model():
    """
    Stage 1: Deep Vanilla U-Net.
    Optimized for high-frequency map textures and sharp center detection.
    """
    inputs = Input((PATCH_SIZE, PATCH_SIZE, 3))
    
    def conv_block(x, filters):
        x = Conv2D(filters, 3, padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, 3, padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    # --- ENCODER ---
    f1 = conv_block(inputs, 32)
    p1 = MaxPooling2D((2, 2))(f1) # 128
    
    f2 = conv_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(f2) # 64
    
    f3 = conv_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(f3) # 32
    
    f4 = conv_block(p3, 256)
    p4 = MaxPooling2D((2, 2))(f4) # 16
    
    # --- BOTTLENECK ---
    b = conv_block(p4, 512)
    
    # --- DECODER ---
    u4 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(b)
    u4 = Concatenate()([u4, f4])
    u4 = conv_block(u4, 256)
    
    u3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(u4)
    u3 = Concatenate()([u3, f3])
    u3 = conv_block(u3, 128)
    
    u2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(u3)
    u2 = Concatenate()([u2, f2])
    u2 = conv_block(u2, 64)
    
    u1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(u2)
    u1 = Concatenate()([u1, f1])
    u1 = conv_block(u1, 32)
    
    # --- HEAD ---
    # Constant bias -5.0 means the model starts by assuming 0.6% probability of text.
    # This prevents the blue "haze" by making the model earn every pixel of confidence.
    outputs = Conv2D(1, (1, 1), activation='sigmoid', 
                    bias_initializer=Constant(-5.0), name='center_heatmap')(u1)
    
    return Model(inputs, outputs, name="Stage1_DeepUNet")

def build_polygon_model():
    """Stage 2: Polygon Regressor remains ResNet50 as it handles specific crops well."""
    input_shape = (128, 128, 3) 
    inputs = Input(input_shape)
    base = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_RAYS, activation='sigmoid', name='ray_vector')(x)
    return Model(inputs, outputs, name="Stage2_PolygonRegressor")