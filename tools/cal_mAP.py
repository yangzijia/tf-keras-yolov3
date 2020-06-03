# -*- coding: utf-8 -*-
"""
基于 https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py 修改得到的mAP计算脚本
"""
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import pickle  # python2中使用的cPickle在python3中已改名为pickle
import numpy as np
import json
import codecs
import pandas as pd

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    # detfile = detpath.format(classname)
    detfile = os.path.join(cachedir, classname + ".txt")
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        return [0.0], [0.0], 0.0

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def cal_mAP(res_dir, gt_dir, class_path, imagesetfile):
    """
    :param res_dir: 模型预测结果所在的目录
    :param gt_dir: groud_truth所在的目录
    :param class_path: 数据集所有类别名的文件路径
    :param imagesetfile: 图像名字列表文件路径
    :return: None
    """
    with codecs.open(class_path, 'r', 'utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    cachedir = os.path.join(res_dir, '../mAP_tmp')
    if not os.path.exists(cachedir):
        os.mkdir(cachedir)
    process_results(res_dir, cachedir, class_names)

    detpath = res_dir + '/../{:s}.txt'  # 检测结果的路径，{:s}.txt方便后面根据图像名字读取对应的txt文件
    annopath = gt_dir + '/{:s}.xml'  # ground truth的路径，{:s}.xml方便后面根据图像名字读取对应的xml文件

    aps = []  # 保存各类ap
    eval_pd = pd.DataFrame(columns=['classname', 'AP', 'recall', 'precision'])  # 保存统计结果
    for i, classname in enumerate(class_names):
        rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname, cachedir,
                                 ovthresh=0.5, use_07_metric=False)
        aps.append(ap)
        eval_pd.loc[i] = [classname, '%.4f' % ap, '%.4f' % rec[-1], '%.4f' % prec[-1]]  # 插入一行

    eval_pd.loc[len(class_names)] = ['mAP', '%.4f' % np.mean(aps), None, None]
    eval_pd.to_excel(os.path.join(cachedir, 'cal_mAp.xls'))
    print(eval_pd)
    print('mAP = %.4f' % np.mean(aps))


def process_results(res_dir, cachedir, class_names):
    """
    将 “按照图片名逐个文件保存预测结果的方式” 转换成 “按照类别名逐个文件保存预测结果的方式”
    """
    results = {}
    for class_name in class_names:
        results[class_name] = []

    files = os.listdir(res_dir)
    for file_name in files:
        if not file_name.endswith('.txt'):
            continue
        file_name_prefix = file_name.split('.')[0]
        file_path = os.path.join(res_dir, file_name)
        try:
            with codecs.open(file_path, 'r', 'utf-8') as f:
                result = json.load(f)
        except Exception as e:
            print('error', file_name, e)
        detection_classes = result['detection_classes']
        detection_scores = result['detection_scores']
        detection_boxes = result['detection_boxes']
        for index, class_name in enumerate(detection_classes):
            box = detection_boxes[index]
            xmin = '%.1f' % box[0]  # TODO 注意不要弄错坐标的顺序
            ymin = '%.1f' % box[1]
            xmax = '%.1f' % box[2]
            ymax = '%.1f' % box[3]
            line = (file_name_prefix + ' ' + '%.4f' % detection_scores[index] + ' ' + ' '.join([xmin, ymin, xmax, ymax]) + '\n')
            results[class_name].append(line)

    for class_name in class_names:
        if not os.path.exists(os.path.join(cachedir, class_name + '.txt')):
            with codecs.open(os.path.join(cachedir, class_name + '.txt'), 'w', 'utf-8') as f:
                f.writelines(results[class_name])
    print('process_results end')


if __name__ == '__main__':
    res_dir = "2020sz/mAP/detect_result_01"  # 模型预测结果

    # res_dir = "2020sz/val"
    gt_dir = "sample/trainval/VOC2007/Annotations"
    class_path = "sample/trainval/train_classes.txt"  # 数据集所有类别名的文件
    imagesetfile = "2020sz/val.txt"  # 读取图像名字列表文件

    cal_mAP(res_dir, gt_dir, class_path, imagesetfile)
