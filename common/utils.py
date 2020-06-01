#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import codecs
import colorsys
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw


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


def get_colors(class_names):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


def draw_label(matrix, text, color, coords):
    image = Image.fromarray(matrix)
    font = ImageFont.truetype(font='data/font/NotoSansHans-Black.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(text, font)

    # font = cv2.FONT_HERSHEY_PLAIN
    # font_scale = 1.
    # (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    # padding = 5
    # rect_height = text_height + padding * 2
    # rect_width = text_width + padding * 2

    (x, y) = coords

    if y - label_size[1] >= 0:
        text_origin = np.array([x, y - label_size[1]])
    else:
        text_origin = np.array([x, y + 1])

    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=color)
    draw.text(text_origin, text, fill=(0, 0, 0), font=font)
    img = np.asarray(image)
    del draw

    # cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    # cv2.putText(image, text, (x + padding, y - text_height + padding), font,
    #             fontScale=font_scale,
    #             color=(255, 255, 255),
    #             lineType=cv2.LINE_AA)

    return img


def draw_boxes(image, boxes, classes, scores, class_names, colors, show_score=True):
    if classes is None or len(classes) == 0:
        return image

    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = box

        class_name = class_names[cls]
        if show_score:
            label = '{} {:.2f}'.format(class_name, score)
        else:
            label = '{}'.format(class_name)
        print(label, (xmin, ymin), (xmax, ymax))

        # if no color info, use black(0,0,0)
        if colors == None:
            color = (0, 0, 0)
        else:
            color = colors[cls]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)
        image = draw_label(image, label, color, (xmin, ymin))

    return image
