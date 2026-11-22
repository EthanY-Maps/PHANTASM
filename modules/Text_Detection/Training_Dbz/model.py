import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import ResNet50

def conv_bn_relu(x, filters, kernel_size=3, strides=1, padding='same'):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_dbnet_plusplus_model(input_size=640):
    input_shape = (input_size, input_size, 3)
    inputs = Input(input_shape)

    # --- 1. BACKBONE (ResNet50) ---
    resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    
    c2 = resnet.get_layer('conv2_block3_out').output  # stride 4
    c3 = resnet.get_layer('conv3_block4_out').output  # stride 8
    c4 = resnet.get_layer('conv4_block6_out').output  # stride 16
    c5 = resnet.get_layer('conv5_block3_out').output  # stride 32

    # --- 2. NECK (FPN) ---
    p5 = layers.Conv2D(256, 1)(c5)
    p4 = layers.Add()([layers.Conv2D(256, 1)(c4), layers.UpSampling2D(2)(p5)])
    p3 = layers.Add()([layers.Conv2D(256, 1)(c3), layers.UpSampling2D(2)(p4)])
    p2 = layers.Add()([layers.Conv2D(256, 1)(c2), layers.UpSampling2D(2)(p3)])

    p5 = layers.Conv2D(64, 3, padding='same', use_bias=False)(p5)
    p4 = layers.Conv2D(64, 3, padding='same', use_bias=False)(p4)
    p3 = layers.Conv2D(64, 3, padding='same', use_bias=False)(p3)
    p2 = layers.Conv2D(64, 3, padding='same', use_bias=False)(p2)

    # Upsample all to the same resolution (stride 4)
    p3_up = layers.UpSampling2D(2)(p3)
    p4_up = layers.UpSampling2D(4)(p4)
    p5_up = layers.UpSampling2D(8)(p5)

    # --- 3. ADAPTIVE SCALE FUSION (ASF) ---
    # Concatenate features initially to learn spatial relationships
    concat_features = layers.Concatenate()([p2, p3_up, p4_up, p5_up])

    # Learn the attention weights
    attn = conv_bn_relu(concat_features, 256)
    # Output 4 channels (one weight map for each of our 4 scales), using sigmoid to get values between 0 and 1
    attn = layers.Conv2D(4, 1, padding='same', activation='sigmoid', name='asf_attention')(attn)

    # Split the 4-channel attention map into 4 individual maps
    w2 = layers.Lambda(lambda x: x[..., 0:1])(attn)
    w3 = layers.Lambda(lambda x: x[..., 1:2])(attn)
    w4 = layers.Lambda(lambda x: x[..., 2:3])(attn)
    w5 = layers.Lambda(lambda x: x[..., 3:4])(attn)

    # Multiply each scale by its learned spatial weight
    p2_weighted = layers.Multiply()([p2, w2])
    p3_weighted = layers.Multiply()([p3_up, w3])
    p4_weighted = layers.Multiply()([p4_up, w4])
    p5_weighted = layers.Multiply()([p5_up, w5])

    # Final fusion
    fuse = layers.Concatenate(name='asf_fuse')([p2_weighted, p3_weighted, p4_weighted, p5_weighted])

    # --- 4. HEADS ---
    # Probability Map (P)
    p = conv_bn_relu(fuse, 256)
    p = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(p)
    p = layers.Conv2DTranspose(1, 2, strides=2, padding='same', activation='sigmoid')(p)

    # Threshold Map (T)
    t = conv_bn_relu(fuse, 256)
    t = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(t)
    t = layers.Conv2DTranspose(1, 2, strides=2, padding='same', activation='sigmoid')(t)

    # Approximate Binary Map (B_hat)
    k = 50.0
    binary = layers.Lambda(lambda x: 1.0 / (1.0 + tf.exp(-k * (x[0] - x[1]))))([p, t])

    # --- 5. OUTPUT CONCATENATION ---
    outputs = layers.Concatenate(axis=-1, name='db_output')([p, t, binary])

    return models.Model(inputs, outputs, name='DBNet_PlusPlus')