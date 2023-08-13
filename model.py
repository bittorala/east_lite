import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D
import numpy as np
import os

TEXT_SCALE = 512
INVALID_MODEL_MSG = "Choose a valid base model: either 'resnet' or 'mobilenet'"


def unpool(inputs):
    return UpSampling2D(
        size=(2, 2), data_format="channels_last", interpolation="bilinear"
    )(inputs)


def find_out_layers(m, base_model):
    transfer_layers = []
    if base_model == "resnet":
        transfer_layers = [
            "pool1_pool",
            "conv3_block4_out",
            "conv4_block6_out",
            "conv5_block3_out",
        ]
    elif base_model == "mobilenet":
        transfer_layers = [
            "conv_pw_2_relu",
            "conv_pw_5_relu",
            "conv_pw_11_relu",
            "conv_pw_13_relu",
        ]
    else:
        raise (INVALID_MODEL_MSG)
    return [l for l in m.layers if l.name in transfer_layers]


def dice_coefficient(y_true_score, y_pred_score, training_mask):
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_score * y_pred_score * training_mask)
    # intersection = tf.reduce_sum(y_true_score * y_pred_score)
    union = tf.reduce_sum(y_true_score) + tf.reduce_sum(y_pred_score) + eps
    loss = 1.0 - (2 * intersection / union)
    tf.summary.scalar("classification_dice_loss", loss)
    return loss


def loss(y_true, y_pred):
    # Extract the training mask from the gt maps
    training_mask = y_true[:, :, :, -1:]
    y_true = y_true[:, :, :, :-1]
    classification_loss = 0.01 * dice_coefficient(
        y_true[:, :, :, -1:], y_pred[:, :, :, -1:], training_mask
    )

    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(
        value=y_true[:, :, :, :-1], num_or_size_splits=5, axis=3
    )
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(
        value=y_pred[:, :, :, :-1], num_or_size_splits=5, axis=3
    )
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_intersect = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_intersect = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_intersect * h_intersect
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.math.log((area_intersect + 1.0) / (area_union + 1.0))
    L_theta = 1 - tf.math.cos(theta_pred - theta_gt)
    # tf.summary.scalar("geometry_AABB", tf.reduce_mean(L_AABB * y_true[:, :, :, -1:]))
    # tf.summary.scalar("geometry_theta", tf.reduce_mean(L_theta * y_true[:, :, :, -1:]))
    L_g = L_AABB + 20 * L_theta

    return 300 * (
        tf.reduce_mean(L_g * y_true[:, :, :, -1:] * training_mask) + classification_loss
    )


# Return the model, based on either ResNet or MobileNet
# Args
# 1. freeze: Whether or not to freeze base model layers (make them non trainable)
# 2. base_model: One of either 'resnet' or 'mobilenet'
def model(freeze=True, base_model="mobilenet"):
    # RGB -> BGR conversion and substract the means of ImageNet dataset
    # Note that although cv2 loads as BGR, the generator has changed to RGB
    input = tf.keras.Input(shape=[None, None, 3], dtype=tf.float32)
    input = tf.keras.applications.resnet.preprocess_input(input)
    m = None
    if base_model == "resnet":
        m = tf.keras.applications.resnet50.ResNet50(
            input_tensor=input, include_top=False
        )
    elif base_model == "mobilenet":
        m = tf.keras.applications.MobileNet(input_tensor=input, include_top=False)
    else:
        raise (INVALID_MODEL_MSG)

    if freeze:
        for l in m.layers:
            l.trainable = False

    block_layer_indices = find_out_layers(m, base_model)
    block_layer_indices.reverse()
    f = [l.output for l in block_layer_indices]
    g = [None, None, None, None]
    h = [None, None, None, None]
    num_outputs = [None, 128, 64, 32]
    for i in range(4):
        if i == 0:
            h[i] = f[i]
        else:
            c1_1 = Conv2D(
                filters=num_outputs[i],
                kernel_size=1,
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            )(tf.concat([g[i - 1], f[i]], axis=-1))
            h[i] = Conv2D(
                filters=num_outputs[i],
                kernel_size=3,
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            )(c1_1)
        if i <= 2:
            g[i] = unpool(h[i])
        else:
            g[i] = Conv2D(
                filters=num_outputs[i],
                kernel_size=3,
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            )(h[i])

    F_score = Conv2D(
        filters=1,
        kernel_size=1,
        activation=tf.nn.sigmoid,
        padding="same",
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
    )(g[3])
    geo_map = (
        Conv2D(
            filters=4,
            kernel_size=1,
            activation=tf.nn.sigmoid,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
        )(g[3])
        * TEXT_SCALE
    )
    angle_map = (
        (
            Conv2D(
                filters=1,
                kernel_size=1,
                activation=tf.nn.sigmoid,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            )(g[3])
            - 0.5
        )
        * np.pi
        / 2
    )
    F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    output = tf.concat([F_geometry, F_score], axis=-1)

    model = tf.keras.Model(inputs=input, outputs=[output], name="east_tf_keras")

    return model


if __name__ == "__main__":
    model()
