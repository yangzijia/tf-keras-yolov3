#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import codecs
import numpy as np


def get_classes(classes_path):
    """loads the classes"""
    with codecs.open(classes_path, 'r', 'utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def get_dataset(annotation_file, shuffle=True):
    with open(annotation_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    if shuffle:
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)

    return lines


def get_multiscale_list():
    input_shape_list = [(320, 320), (352, 352), (384, 384), (416, 416), (448, 448), (480, 480), (512, 512), (544, 544),
                        (576, 576), (608, 608)]

    return input_shape_list
