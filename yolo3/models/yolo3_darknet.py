#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 Darknet Model Defined in Keras."""

from tensorflow._api.v1.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow._api.v1.keras.models import Model

from yolo3.models.layers import compose, DarknetConv2D_BN_Leaky, make_spp_last_layers, DarknetConv2D
from yolo3.models.layers import make_last_layers


def resblock_body(x, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet53_body(x):
    """Darknet53 body having 52 Convolution2D layers"""
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def yolo3_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet53_body(inputs))

    # f1: 13 x 13 x 1024
    f1 = darknet.output
    # f2: 26 x 26 x 512
    f2 = darknet.layers[152].output
    # f3: 52 x 52 x 256
    f3 = darknet.layers[92].output

    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    # feature map 1 head & output (19x19 for 608 input)
    x, y1 = make_last_layers(f1, f1_channel_num // 2, num_anchors * (num_classes + 5))

    # upsample fpn merge for feature map 1 & 2
    x = compose(
        DarknetConv2D_BN_Leaky(f2_channel_num // 2, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, f2])

    # feature map 2 head & output (38x38 for 608 input)
    x, y2 = make_last_layers(x, f2_channel_num // 2, num_anchors * (num_classes + 5))

    # upsample fpn merge for feature map 2 & 3
    x = compose(
        DarknetConv2D_BN_Leaky(f3_channel_num // 2, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    # feature map 3 head & output (76x76 for 608 input)
    x, y3 = make_last_layers(x, f3_channel_num // 2, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


def custom_yolo3_spp_body(inputs, num_anchors, num_classes):
    """Create a custom YOLO_v3 SPP model, use
       pre-trained weights from darknet and fit
       for our target classes."""
    # TODO: get darknet class number from class file
    num_classes_coco = 80
    base_model = yolo3_spp_body(inputs, num_anchors, num_classes_coco)

    # get conv output in original network
    y1 = base_model.get_layer('leaky_re_lu_58').output
    y2 = base_model.get_layer('leaky_re_lu_65').output
    y3 = base_model.get_layer('leaky_re_lu_72').output
    y1 = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='prediction_13')(y1)
    y2 = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='prediction_26')(y2)
    y3 = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='prediction_52')(y3)
    return Model(inputs, [y1, y2, y3])


def yolo3_spp_body(inputs, num_anchors, num_classes, weights_path=None):
    """Create YOLO_V3 SPP model CNN body in Keras."""
    darknet = Model(inputs, darknet53_body(inputs))
    if weights_path is not None:
        darknet.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))

    # f1: 13 x 13 x 1024
    f1 = darknet.output
    # f2: 26 x 26 x 512
    f2 = darknet.layers[152].output
    # f3: 52 x 52 x 256
    f3 = darknet.layers[92].output

    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    # feature map 1 head & output (19x19 for 608 input)
    x, y1 = make_spp_last_layers(f1, f1_channel_num // 2, num_anchors * (num_classes + 5))

    # upsample fpn merge for feature map 1 & 2
    x = compose(
        DarknetConv2D_BN_Leaky(f2_channel_num // 2, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, f2])

    # feature map 2 head & output (38x38 for 608 input)
    x, y2 = make_last_layers(x, f2_channel_num // 2, num_anchors * (num_classes + 5))

    # upsample fpn merge for feature map 2 & 3
    x = compose(
        DarknetConv2D_BN_Leaky(f3_channel_num // 2, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    # feature map 3 head & output (76x76 for 608 input)
    x, y3 = make_last_layers(x, f3_channel_num // 2, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])
