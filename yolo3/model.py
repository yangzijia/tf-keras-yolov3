#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.optimizers import Adam

from yolo3.loss import yolo3_loss
from common.model_utils import add_metrics

from yolo3.models.yolo3_darknet import yolo3_body, custom_yolo3_spp_body
from yolo3.postprocess import batched_yolo3_postprocess

yolo3_model_map = {
    'yolo3_darknet': [yolo3_body, 185],
    'yolo3_darknet_spp': [custom_yolo3_spp_body, 185],
}


def get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, input_tensor=None, input_shape=None):
    if input_shape:
        input_tensor = Input(shape=input_shape, name='image_input')

    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

    # YOLOv3 model has 9 anchors and 3 feature layers
    if num_feature_layers == 3:
        if model_type in yolo3_model_map:
            model_function = yolo3_model_map[model_type][0]
            backbone_len = yolo3_model_map[model_type][1]

            model_body = model_function(input_tensor, num_anchors // 3, num_classes)
        else:
            raise ValueError('This model type is not supported now')
    else:
        raise ValueError('model type mismatch anchors')

    return model_body, backbone_len


def get_yolo3_train_model(model_type, anchors, num_classes, input_shape, weights_path=None, freeze_level=1,
                          optimizer=Adam(lr=1e-3, decay=0), label_smoothing=0):
    """create the training model, for YOLOv3"""
    num_anchors = len(anchors)
    num_feature_layers = num_anchors // 3

    h, w = input_shape

    # y_true = [Input(shape=(None, None, 3, num_classes + 5), name='y_true_{}'.format(l)) for l in
    #           range(num_feature_layers)]
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], 3, num_classes + 5),
                    name='y_true_{}'.format(l)) for l in range(num_feature_layers)]

    model_body, backbone_len = get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes)
    print('Create {} model with {} anchors and {} classes.'.format(model_type, num_anchors, num_classes))
    print('model layer number:', len(model_body.layers))

    if weights_path:
        model_body.load_weights(weights_path, by_name=True)  # , skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    if freeze_level in [1, 2]:
        # Freeze the backbone part or freeze all but final feature map & input layers.
        num = (backbone_len, len(model_body.layers) - 3)[freeze_level - 1]
        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    elif freeze_level == 0:
        # Unfreeze all layers.
        for i in range(len(model_body.layers)):
            model_body.layers[i].trainable = True
        print('Unfreeze all of the layers.')

    use_focal_obj_loss = False
    use_focal_loss = False
    use_diou_loss = False
    use_softmax_loss = False

    model_loss, location_loss, confidence_loss, class_loss = Lambda(yolo3_loss, output_shape=(1,), name='yolo_loss',
                                                                    arguments={'anchors': anchors,
                                                                               'num_classes': num_classes,
                                                                               'ignore_thresh': 0.5,
                                                                               'label_smoothing': label_smoothing,
                                                                               'use_focal_obj_loss': use_focal_obj_loss, 
                                                                               'use_focal_loss': use_focal_loss, 
                                                                               'use_diou_loss': use_diou_loss,
                                                                               'use_softmax_loss': use_softmax_loss})(
                                                                                [*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    loss_dict = {'location_loss': location_loss, 'confidence_loss': confidence_loss, 'class_loss': class_loss}
    add_metrics(model, loss_dict)

    return model


def get_yolo3_inference_model(model_type, anchors, num_classes, weights_path=None, input_shape=None, confidence=0.1):
    '''create the inference model, for YOLOv3'''
    # K.clear_session() # get a new session
    num_anchors = len(anchors)
    # YOLOv3 model has 9 anchors and 3 feature layers but
    # Tiny YOLOv3 model has 6 anchors and 2 feature layers,
    # so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors // 3

    image_shape = Input(shape=(2,), dtype='int64', name='image_shape')

    model_body, _ = get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, input_shape=input_shape)
    print('Create {} YOLOv3 {} model with {} anchors and {} classes.'.format('Tiny' if num_feature_layers == 2 else '',
                                                                             model_type, num_anchors, num_classes))

    if weights_path:
        model_body.load_weights(weights_path, by_name=False)  # , skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    boxes, scores, classes = Lambda(batched_yolo3_postprocess, name='yolo3_postprocess',
                                    arguments={'anchors': anchors, 'num_classes': num_classes,
                                               'confidence': confidence})(
        [*model_body.output, image_shape])
    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model
