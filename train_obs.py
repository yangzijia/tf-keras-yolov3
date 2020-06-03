#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from common.utils import get_anchors, get_classes, get_dataset
from tensorflow._api.v1.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, \
    TerminateOnNaN
from yolo3.model import get_yolo3_train_model
from yolo3.data import yolo3_data_generator_wrapper
from common.model_utils import get_optimizer

import argparse
from tensorflow._api.v1.keras.callbacks import Callback

try:
    import moxing as mox
except:
    print('not use moxing')

parser = argparse.ArgumentParser(description='yolov3-spp Training')
parser.add_argument('--max_epochs_1', default=70, type=int, help='number of total epochs to run in stage 1')
parser.add_argument('--max_epochs_2', default=150, type=int, help='number of total epochs to run in total')
parser.add_argument('--initial_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size_1', default=128, type=int, help='mini-batch size, default 32')
parser.add_argument('--batch_size_2', default=32, type=int, help='mini-batch size, default 32')
parser.add_argument('--local_data_root', default='/cache/', type=str,
                    help='a directory used for transfer data between local path and OBS path')
parser.add_argument('--data_url', required=True, type=str, help='the training and validation data path')
parser.add_argument('--data_local', default='', type=str, help='the training and validation data path on local')
parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')
parser.add_argument('--train_local', default='', type=str, help='the training output results on local')


class ModertFileToObs(Callback):
    """
    拷贝modert权重文件到obs里面
    """

    def __init__(self, log_dir, args):
        super().__init__()
        self.log_dir = log_dir
        self.args = args

    def on_epoch_end(self, epoch, logs=None):
        current_dir_files = os.listdir(self.log_dir)
        for current_file in current_dir_files:
            if ".h5" in current_file:
                print(current_file)
                obs_current_file_path = os.path.join(self.args.train_url, current_file)
                if not os.path.exists(obs_current_file_path):
                    mox.file.copy(os.path.join(self.log_dir, current_file), obs_current_file_path)

    def on_batch_end(self, batch, logs=None):
        print(batch)
        print('\n')


def gen_model_dir(log_dir, args, classes_path, anchors_path):
    if args.train_url.startswith('s3://') or args.train_url.startswith('obs://'):
        mox.file.copy_parallel(log_dir, args.train_url)

    current_dir = os.path.dirname(__file__)
    mox.file.copy(os.path.join(current_dir, 'deploy_scripts/config.json'),
                  os.path.join(args.train_url, 'model/config.json'))  # mox.file.copy可同时兼容本地和OBS路径的拷贝操作
    mox.file.copy(os.path.join(current_dir, 'deploy_scripts/customize_service.py'),
                  os.path.join(args.train_url, 'model/customize_service.py'))
    mox.file.copy_parallel(os.path.join(current_dir, 'yolo3'),
                           os.path.join(args.train_url, 'model/yolo3'))  # 拷贝一个目录得用copy_parallel接口
    mox.file.copy_parallel(os.path.join(current_dir, 'common'),
                           os.path.join(args.train_url, 'model/common'))
    mox.file.copy(os.path.join(log_dir, 'trained_weights_final.h5'),
                  os.path.join(args.train_url, 'model/trained_weights_final.h5'))  # 默认拷贝最后一个模型到model目录
    mox.file.copy(classes_path,
                  os.path.join(args.train_url, 'model', os.path.basename(classes_path)))
    mox.file.copy(anchors_path,
                  os.path.join(args.train_url, 'model', os.path.basename(anchors_path)))
    mox.file.copy(os.path.join(current_dir, 'sample/trainval/classify_rule.json'),
                  os.path.join(args.train_url, 'model/classify_rule.json'))

    print('gen_model_dir success, model dir is at', os.path.join(args.train_url, 'model'))


def main(args):
    model_type = "yolo3_darknet_spp"  # yolo3_darknet_spp, yolo3_darknet
    current_dir = os.path.dirname(__file__) + "/"
    print("current_dir == ", current_dir)
    annotation_file = current_dir + "sample/trainval/train.txt"
    val_annotation_file = current_dir + "sample/trainval/val.txt"

    classes_path = current_dir + "sample/trainval/train_classes.txt"
    anchors_path = current_dir + "sample/trainval/yolo_anchors.txt"
    weights_path = current_dir + "weights/yolov3-spp.h5"
    load_weights_path = None  # None or "{weights path}"
    is_one_stage_train = True
    learning_rate_1 = 1e-4
    learning_rate_2 = 1e-5
    epoch_1 = args.max_epochs_1
    epoch_2 = args.max_epochs_2
    batch_size_1 = args.batch_size_1
    batch_size_2 = args.batch_size_2
    freeze_level = 2
    model_image_size = (416, 416)
    val_split = 0.1
    label_smoothing = 0
    enhance_augment = None  # enhance data augmentation type (None/mosaic)
    rescale_interval = 0  # Number of iteration(batches) interval to rescale input size, default=10

    log_dir = os.path.join('logs', '20200602')

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
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1)
    # terminate_on_nan = TerminateOnNaN()

    callbacks = [logging, checkpoint, reduce_lr, early_stopping, ModertFileToObs(log_dir, args)]
    # callbacks = [logging, checkpoint, reduce_lr]

    # get train&val dataset
    dataset = get_dataset(annotation_file)

    dataset = [current_dir + d for d in dataset]
    if val_annotation_file != "":
        val_dataset = get_dataset(val_annotation_file)
        num_train = len(dataset)
        num_val = len(val_dataset)
        dataset.extend(val_dataset)
    else:
        val_split = val_split
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val

        # num_val = 100
        # num_train = 200

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
            validation_data=data_generator(dataset[:num_val], batch_size_1, input_shape, anchors, num_classes),
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
            validation_data=data_generator(dataset[:num_val], batch_size_2, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // batch_size_2),
            epochs=epoch_2,
            initial_epoch=epoch_1,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10,
            callbacks=callbacks)
        model.save_weights(os.path.join(log_dir, 'trained_weights_final.h5'))

    gen_model_dir(log_dir, args, classes_path, anchors_path)


if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    main(args)
