from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import os
import numpy as np
from shutil import copy
import cv2

# 解决plt中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
draw_plot = True


def read(path):
    data = []
    with open(path, 'r', encoding="utf8") as f:
        data = f.readlines()
    data = [c.strip() for c in data]
    return data


ground_truth_path = "C:/Users/zjyang/Desktop/garbage_classify/SZ_2020/ground_truth"

class_names = read("sample/trainval/train_classes.txt")
image_dir_path = "sample/trainval/VOC2007/JPEGImages"

detection_result = "2020sz/mAP/detect_result_01_new"
error_path = "2020sz/mAP/draw_error_class"

class_names.append("空")


def generate_confusion(y_true, y_pred):
    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    labels = np.arange(len(cm))
    # labels = read("submit_code/model_data/train_classes.txt")
    labels = class_names
    # lables = np.arange(class_names)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 8), dpi=120)
    # set the fontsize of label.
    # for label in plt.gca().xaxis.get_ticklabels():
    #    label.set_fontsize(8)
    # text portion
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        # c = cm_normalized[y_val][x_val]
        c = cm[y_val][x_val]
        if (c > 0):
            plt.text(x_val, y_val, c, color='red',
                     fontsize=7, va='center', ha='center')
            # plt.text(x_val, y_val, "%0.2f" %(c,), color='red', fontsize=7, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='confusion matrix')
    # show confusion matrix
    plt.show()


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def zh_ch(string):
    return string.encode("gbk").decode(errors="ignore")


def draw(image, tbox, gbox):
    cv2.rectangle(image, (tbox[0], tbox[1]), (tbox[2], tbox[3]), (0, 255, 0), 3)
    if len(gbox) > 0:
        cv2.rectangle(image, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (0, 0, 255), 3)
    return image


def deal_error_class(**argv):
    error_image_dir = "{}/{}".format(error_path, class_names[argv["gt_class"]])
    os.makedirs(error_image_dir, exist_ok=True)
    new_image_path = "{}/{}_{}_{}".format(error_image_dir, class_names[argv["error_class"]], argv["index"],
                                          argv["name"])

    image = cv2.imread(image_dir_path + "/" + argv["name"])
    print(new_image_path)

    # true box
    tbox = argv["gt_box"]
    # guess box
    gbox = argv["error_box"]
    image = draw(image, tbox, gbox)

    cv2.imencode('.jpg', image)[1].tofile(new_image_path)


def get_fact_guess(iou_score=0.5):
    fact = []
    guess = []

    image_names = []

    error_list = [0 for _ in range(len(class_names))]

    for val in os.walk(detection_result):
        image_names = val[2]

    for name in image_names:
        t_results = read("{}/{}".format(ground_truth_path, name))
        d_results = read("{}/{}".format(detection_result, name))
        index_e = 0
        for t in t_results:
            gt_box_all = [int(gt) for gt in t.split(",")]
            gt_box = gt_box_all[:4]
            gt_class = gt_box_all[4]

            class_list = []
            iou_list = []
            box_list = []
            for d in d_results:
                d_box_all = [int(float(d)) for d in d.split(",")]
                d_box = d_box_all[:4]
                d_class = d_box_all[4]
                tmp_iou = compute_iou(gt_box, d_box)
                if tmp_iou > iou_score:
                    class_list.append(d_class)
                    iou_list.append(tmp_iou)
                    box_list.append(d_box)

            fact.append(gt_class)
            if len(class_list) > 0:
                # 提取出iou最大的那个class用来做为预测的class
                max_iou_index = iou_list.index(max(iou_list))
                max_class = class_list[max_iou_index]
                max_box = box_list[max_iou_index]

                guess.append(max_class)

                # 预测的class如果不是正确的则对应填写到错误矩阵中
                if gt_class != max_class:
                    error_list[gt_class] += 1
                    argv = {
                        "gt_class": gt_class,
                        "name": name.split(".txt")[0] + ".jpg",
                        "error_class": max_class,
                        "gt_box": gt_box,
                        "error_box": max_box,
                        "index": error_list[gt_class]
                    }
                    deal_error_class(**argv)

            else:
                guess.append(44)
                index_e += 1
                argv = {
                    "gt_class": gt_class,
                    "name": name.split(".txt")[0] + ".jpg",
                    "error_class": 44,
                    "gt_box": gt_box,
                    "error_box": [],
                    "index": index_e
                }
                deal_error_class(**argv)

    fact = [class_names[f] for f in fact]
    guess = [class_names[g] for g in guess]

    return fact, guess


if __name__ == "__main__":
    # 预测数据，predict之后的预测结果集
    guess = [1, 0, 1, 2, 1, 0, 1, 4, 1, 4]
    # 真实结果集
    fact = [0, 1, 0, 1, 2, 1, 0, 1, 0, 1]

    fact, guess = get_fact_guess()
    generate_confusion(fact, guess)

    # tbox = [846, 362, 1023, 750]
    # gbox = [425, 29, 577, 85]
    # iou = compute_iou(tbox, gbox)
    # print(iou)