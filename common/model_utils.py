#!/usr/bin/python3
# -*- coding=utf-8 -*-
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


def add_metrics(model, metric_dict):
    '''
    add metric scalar tensor into model, which could be tracked in training
    log and tensorboard callback
    '''
    for (name, metric) in metric_dict.items():
        # seems add_metric() is newly added in tf.keras. So if you
        # want to customize metrics on raw keras model, just use
        # "metrics_names" and "metrics_tensors" as follow:
        #
        #model.metrics_names.append(name)
        #model.metrics_tensors.append(loss)
        model.add_metric(metric, name=name, aggregation='mean')



def get_lr_scheduler(learning_rate, decay_type, decay_steps):
    if decay_type:
        decay_type = decay_type.lower()

    if decay_type == None:
        lr_scheduler = learning_rate
    else:
        raise ValueError('Unsupported lr decay type')

    return lr_scheduler


def get_optimizer_bak(optim_type, learning_rate, decay_type='cosine', decay_steps=100000):
    optim_type = optim_type.lower()

    lr_scheduler = get_lr_scheduler(learning_rate, decay_type, decay_steps)

    if optim_type == 'adam':
        optimizer = Adam(lr=lr_scheduler, amsgrad=False)
    elif optim_type == 'rmsprop':
        optimizer = RMSprop(lr=lr_scheduler, rho=0.9, momentum=0.0, centered=False)
    elif optim_type == 'sgd':
        optimizer = SGD(lr=lr_scheduler, momentum=0.0, nesterov=False)
    else:
        raise ValueError('Unsupported optimizer type')

    return optimizer


def get_optimizer(learning_rate):
    return Adam(lr=learning_rate, amsgrad=False)