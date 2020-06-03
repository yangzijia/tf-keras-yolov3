# -*- coding: utf-8 -*-
import os
import json
import codecs
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from collections import OrderedDict
from yolo3.utils import letterbox_image
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body

from model_service.tfserving_model_service import TfServingBaseService

import time
from metric.metrics_manager import MetricsManager
import log
logger = log.getLogger(__name__)


class ObjectDetectionService(TfServingBaseService):
    def __init__(self, model_name, model_path):
        if self.is_tf_gpu_version() is True:
            print('use tf GPU version,', tf.__version__)
        else:
            print('use tf CPU version,', tf.__version__)

        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = os.path.join(os.path.dirname(__file__), 'trained_weights_final.h5')
        self.input_image_key = 'images'

        self.anchors_path = os.path.join(os.path.dirname(__file__), 'train_anchors.txt')
        self.classes_path = os.path.join(os.path.dirname(__file__), 'train_classes.txt')
        self.score = 0.3
        self.iou = 0.45
        self.model_image_size = (416, 416)

        self.label_map = parse_classify_rule(os.path.join(os.path.dirname(__file__), 'classify_rule.json'))
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.K_learning_phase = K.learning_phase()
                self.boxes, self.scores, self.classes = self.generate()
        print('load weights file success')

    def is_tf_gpu_version(self):
        from tensorflow.python.client import device_lib
        is_gpu_version = False
        devices_info = device_lib.list_local_devices()
        for device in devices_info:
            if 'GPU' == str(device.device_type):
                is_gpu_version = True
                break

        return is_gpu_version

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with codecs.open(classes_path, 'r', 'utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting

        self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
            if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
        self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match

        print('{} model, anchors, and classes loaded.'.format(model_path))

        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content)
                preprocessed_data[k] = image
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        image = data[self.input_image_key]

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                self.K_learning_phase: 0
            })

        result = OrderedDict()
        if out_boxes is not None:
            detection_class_names = []
            for class_id in out_classes:
                class_name = self.class_names[int(class_id)]
                class_name = self.label_map[class_name] + '/' + class_name
                detection_class_names.append(class_name)
            out_boxes_list = []
            for box in out_boxes:
                out_boxes_list.append([round(float(v), 1) for v in box])  # v是np.float32类型，会导致无法json序列化，因此使用float(v)转为python内置float类型
            result['detection_classes'] = detection_class_names
            result['detection_scores'] = [round(float(v), 4) for v in out_scores]
            result['detection_boxes'] = out_boxes_list
        else:
            result['detection_classes'] = []
            result['detection_scores'] = []
            result['detection_boxes'] = []
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        '''
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : map of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        '''
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')

        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)

        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)

        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)

        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)

        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = str(round(pre_time_in_ms + infer_in_ms + post_time_in_ms, 1)) + ' ms'
        return data


def parse_classify_rule(json_path=''):
    with codecs.open(json_path, 'r', 'utf-8') as f:
        rule = json.load(f)
    label_map = {}
    for super_label, labels in rule.items():
        for label in labels:
            label_map[label] = super_label
    return label_map
