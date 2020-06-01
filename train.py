#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from common.utils import get_anchors, get_classes, get_dataset
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
from tensorflow._api.v1.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN

from yolo3.model import get_yolo3_train_model
from yolo3.data import yolo3_data_generator_wrapper
from common.model_utils import get_optimizer


def main():
    model_type = "yolo3_darknet_spp"        # yolo3_darknet_spp, yolo3_darknet
    annotation_file = "sample/trainval/train.txt"
    val_annotation_file = "sample/trainval/val.txt"
    
    # annotation_file = "sample/shape_voc/trainval.txt"
    # val_annotation_file = ""

    classes_path = "sample/trainval/train_classes.txt"
    anchors_path = "sample/trainval/yolo_anchors.txt"
    weights_path = "weights/yolov3-spp.h5"
    # weights_path = "weights/yolov3_2.h5"
    load_weights_path = None  # None or "{weights path}"
    is_one_stage_train = True
    learning_rate_1 = 1e-3
    learning_rate_2 = 1e-4
    epoch_1 = 1
    epoch_2 = 2
    batch_size_1 = 32
    batch_size_2 = 8
    freeze_level = 2
    model_image_size = (416, 416)
    val_split = 0.1
    label_smoothing = 0
    enhance_augment = None  # enhance data augmentation type (None/mosaic)
    rescale_interval = 0  # Number of iteration(batches) interval to rescale input size, default=10

    log_dir = os.path.join('logs', '20200531_test')

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    logging = TensorBoard(log_dir=log_dir, update_freq='batch')
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                 monitor='val_loss',
                                 verbose=1,
                                 save_weights_only=True,
                                 save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, cooldown=0, min_lr=1e-10)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1)
    # terminate_on_nan = TerminateOnNaN()

    # callbacks = [logging, checkpoint, reduce_lr, early_stopping]
    callbacks = [logging, checkpoint, reduce_lr]

    # get train&val dataset
    dataset = get_dataset(annotation_file)
    if val_annotation_file != "":
        val_dataset = get_dataset(val_annotation_file)
        num_train = len(dataset)
        num_val = len(val_dataset)
        dataset.extend(val_dataset)
    else:
        val_split = val_split
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val

    # model input shape check
    input_shape = model_image_size
    assert (input_shape[0] % 32 == 0 and input_shape[1] % 32 == 0), 'Multiples of 32 required'

    get_train_model = get_yolo3_train_model
    data_generator = yolo3_data_generator_wrapper

    # get train model
    model = get_train_model(model_type, anchors, num_classes, input_shape, weights_path=weights_path,
                            freeze_level=freeze_level, label_smoothing=label_smoothing)

    if load_weights_path:
        model.load_weights(load_weights_path)
        print("reload weights: {}".format(load_weights_path))

    if is_one_stage_train:
        model.compile(optimizer=get_optimizer(learning_rate_1), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('One stage Train on {} samples, val on {} samples, with batch size {}, '
              'input_shape {}.'.format(num_train, num_val, batch_size_1, input_shape))
        model.fit_generator(
            data_generator(dataset[:num_train], batch_size_1, input_shape, anchors, num_classes, enhance_augment),
            steps_per_epoch=max(1, num_train // batch_size_1),
            validation_data=data_generator(dataset[num_train:], batch_size_1, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // batch_size_1),
            epochs=epoch_1,
            initial_epoch=0,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10,
            callbacks=callbacks)

        model.save_weights(os.path.join(log_dir, 'trained_weights_stage_1.h5'))

    if True:
        print("Unfreeze and continue training, to fine-tune.")
        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        model.compile(optimizer=get_optimizer(learning_rate_2), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('Two stage Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train,
                                                                                                             num_val,
                                                                                                             batch_size_2,
                                                                                                             input_shape))

        model.fit_generator(
            data_generator(dataset[:num_train], batch_size_2, input_shape, anchors, num_classes, enhance_augment,
                           rescale_interval),
            steps_per_epoch=max(1, num_train // batch_size_2),
            validation_data=data_generator(dataset[num_train:], batch_size_2, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // batch_size_2),
            epochs=epoch_2,
            initial_epoch=epoch_1,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10,
            callbacks=callbacks)
        model.save_weights(os.path.join(log_dir, 'trained_weights_final.h5'))


if __name__ == '__main__':
    main()
