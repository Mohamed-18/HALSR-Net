"""
Created on December 10, 2024

@author: Mohamed Fakhfakh
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Reshape, Dense, Multiply, UpSampling2D, Concatenate, Add

def attention_block_2d(x, g, inter_channel):
    """
    2D Attention Block.

    Args:
    - x: Input tensor.
    - g: Gating signal tensor.
    - inter_channel: Number of intermediate channels for attention.

    Returns:
    - att_x: Output tensor after applying the attention mechanism.
    """
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    theta_x_resized = tf.image.resize(theta_x, (phi_g.shape[1], phi_g.shape[2]))
    f = Activation('relu')(Add()([theta_x_resized, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = Activation('sigmoid')(psi_f)
    rate_resized = tf.image.resize(rate, (x.shape[1], x.shape[2]))
    att_x = Multiply()([x, rate_resized])
    return att_x

def attention_up_and_concate(down_layer, layer):
    """
    Attention mechanism with upsampling and concatenation.

    Args:
    - down_layer: Downsampled feature map tensor.
    - layer: Tensor to be concatenated after attention.

    Returns:
    - concate: Output tensor after attention and concatenation.
    """
    in_channel = down_layer.get_shape().as_list()[-1]
    up = UpSampling2D(size=(2, 2))(down_layer)
    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4)
    layer_resized = tf.image.resize(layer, (up.shape[1], up.shape[2]))
    concate = Concatenate(axis=-1)([up, layer_resized])
    return concate

def ASPP(input_tensor, num_filters=256):
    """
    Atrous Spatial Pyramid Pooling (ASPP).

    Args:
    - input_tensor: Input tensor to the ASPP module.
    - num_filters: Number of filters for each convolution layer.

    Returns:
    - output: Output tensor after applying ASPP.
    """
    dilation_rates = [1, 6, 12, 18]
    branches = []
    for rate in dilation_rates:
        b = Conv2D(num_filters, (3, 3), dilation_rate=rate, padding='same', use_bias=False)(input_tensor)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)
        branches.append(b)

    global_avg_pool = GlobalAveragePooling2D()(input_tensor)
    global_avg_pool = Reshape((1, 1, input_tensor.shape[-1]))(global_avg_pool)
    global_avg_pool = Conv2D(num_filters, (1, 1), padding='same', use_bias=False)(global_avg_pool)
    global_avg_pool = BatchNormalization()(global_avg_pool)
    global_avg_pool = Activation('relu')(global_avg_pool)
    global_avg_pool = UpSampling2D(size=input_tensor.shape[1:3])(global_avg_pool)

    output = Concatenate(axis=-1)(branches + [global_avg_pool])
    output = Conv2D(num_filters, (1, 1), padding='same', use_bias=False)(output)
    output = BatchNormalization()(output)
    return Activation('relu')(output)

def channel_attention(input_feature, ratio=8):
    """
    Channel Attention Mechanism.

    Args:
    - input_feature: Input tensor.
    - ratio: Reduction ratio for intermediate dense layers.

    Returns:
    - scale: Output tensor with applied channel attention.
    """
    channel = input_feature.shape[-1]
    shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_dense_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    avg_pool = Reshape((1, 1, channel))(avg_pool)
    scale = Multiply()([input_feature, avg_pool])
    return scale

def spatial_attention(input_tensor):
    """
    Spatial Attention Mechanism.

    Args:
    - input_tensor: Input tensor.

    Returns:
    - output_tensor: Tensor after applying spatial attention.
    """
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, -1))(avg_pool)
    attention = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(avg_pool)
    attention_map = Activation('sigmoid')(attention)
    output_tensor = Multiply()([input_tensor, attention_map])
    return output_tensor

def spatial_pyramid_pooling(input_tensor, levels=[1, 2, 4]):
    """
    Spatial Pyramid Pooling.

    Args:
    - input_tensor: Input tensor.
    - levels: Pyramid levels for pooling.

    Returns:
    - spp_pool: Tensor after spatial pyramid pooling.
    """
    spp_pools = []
    input_shape = tf.shape(input_tensor)
    h, w = input_shape[1], input_shape[2]

    for level in levels:
        pooled = tf.keras.layers.AveragePooling2D(pool_size=(h // level, w // level),
                                                  strides=(h // level, w // level), padding='valid')(input_tensor)
        resized = tf.image.resize(pooled, (h, w))
        spp_pools.append(resized)

    spp_pool = tf.keras.layers.Concatenate(axis=-1)(spp_pools)
    return spp_pool

def build_model_HAPM(input_feature_map, num_filters=64):
    """
    Hybrid Attention and Pooling Module (HAPM).

    Args:
    - input_feature_map: Input tensor to the HAPM.
    - num_filters: Number of filters for the ASPP module.

    Returns:
    - fused_attention_map: Tensor after applying HAPM.
    """
    aspp = ASPP(input_feature_map, num_filters)
    ca_aspp = channel_attention(aspp)
    sa_aspp = spatial_attention(aspp)
    concatenated_ASPP = Concatenate(axis=-1)([ca_aspp, sa_aspp])

    spp = spatial_pyramid_pooling(input_feature_map, levels=[1, 2, 4])
    ca_spp = channel_attention(spp)
    sa_spp = spatial_attention(spp)
    concatenated_SPP = Concatenate(axis=-1)([ca_spp, sa_spp])

    concatenated_A_S = Concatenate(axis=-1)([concatenated_ASPP, concatenated_SPP])
    nla_adjusted = Conv2D(128, (1, 1))(concatenated_A_S)
    input_feature_map = Conv2D(128, (1, 1))(input_feature_map)
    fused_attention_map = Add()([input_feature_map, nla_adjusted])
    return fused_attention_map

def build_reconstructed_image(inter_layer, num_classes, size):
    """
    Reconstruction Module with Attention.

    Args:
    - inter_layer: Intermediate feature tensor.
    - num_classes: Number of output classes.
    - size: Upsampling size.

    Returns:
    - seg_map: Segmentation map tensor.
    """
    attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(inter_layer)
    attended_inter_layer = Multiply()([inter_layer, attention])
    nlb = squeeze_excite_block(attended_inter_layer)
    upsampled_inter_layer = UpSampling2D(size=(size, size))(nlb)
    seg_map = Conv2D(num_classes, (1, 1), padding='same', activation='softmax')(upsampled_inter_layer)
    return seg_map

def squeeze_excite_block(input, ratio=16):
    """
    Squeeze-and-Excitation Block.

    Args:
    - input: Input tensor.
    - ratio: Reduction ratio for dense layers.

    Returns:
    - x: Tensor after applying SE block.
    """
    filters = input.shape[-1]
    se = GlobalAveragePooling2D()(input)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = Multiply()([input, se])
    return x
