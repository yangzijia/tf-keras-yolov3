import numpy as np
from PIL import Image
import os


class YOLO_Kmeans:

    def __init__(self, cluster_number):
        self.cluster_number = cluster_number
        self.anchors_list = []
        self.accuracy_list = []
        self.all_boxes = []

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        aa = np.random.choice(box_number, k, replace=False)
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2clusters(self):
        result = self.kmeans(self.all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        # print("K anchors:\n {}".format(result))
        acc = self.avg_iou(self.all_boxes, result) * 100
        # print("Accuracy: {:.2f}%".format(acc))
        self.anchors_list.append(result)
        self.accuracy_list.append(acc)


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def resize_image(annotation_line, input_shape, max_boxes=20, jitter=.3):
    line = annotation_line.split()
    # print(line[0])
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(','))))
                    for box in line[1:]])

    # resize image
    new_ar = w/h * rand(1-jitter, 1+jitter)/rand(1-jitter, 1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    # image_data = image.resize((nw,nh), Image.BICUBIC)
    image_data = []

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]]*nw/iw
        box[:, [1, 3]] = box[:, [1, 3]]*nh/ih
        # if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def removal(result):
    if len(result) == 0:
        return result
    temp_list = []
    for r in result:
        if len(temp_list) == 0:
            temp_list.append(r)
        else:
            flag = True
            for tt in temp_list:
                if r == tt:
                    flag = False
                    break
            if flag:
                temp_list.append(r)
    new_tmp = []
    for val in temp_list:
        new_tmp.append(np.asarray(val))
    return new_tmp


def get_info(temp_list):
    min_w, min_h, max_w, max_h = 0, 0, 0, 0
    for i, tl in enumerate(temp_list):
        if i == 0:
            min_w = tl[0]
            max_w = tl[0]
            min_h = tl[1]
            max_h = tl[1]
        else:
            if tl[0] < min_w:
                min_w = tl[0]
            elif tl[0] > max_w:
                max_w = tl[0]

            if tl[1] < min_h:
                min_h = tl[0]
            elif tl[1] > max_h:
                max_h = tl[1]
    return min_w, min_h, max_w, max_h


def get_kmeans(cluster_number, result):
    kmeans = YOLO_Kmeans(cluster_number)
    kmeans.all_boxes = np.asarray(result)
    for i in range(500):
        kmeans.txt2clusters()

    list1 = kmeans.anchors_list
    list2 = kmeans.accuracy_list
    index = list2.index(max(list2))
    print("Accuracy:", list2[index])
    kmeans.result2txt(list1[index])
    for aa in list1[index]:
        print("{},{},  ".format(aa[0], aa[1]))
    print("-"*30)


def read_classes(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path, "r", encoding='UTF-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


class Num():
    def __init__(self, name, length):
        self.name = name
        self.length = length
    def __lt__(self, other):
        return self.length < other.length

    def print_info(self):
        print(self.name, self.length)


if __name__ == "__main__":

    """注意：在annotation_path中指定的图片地址需要是真实存在的。"""

    annotation_path = r"sample\shape_voc\trainval.txt"
    classes_path = r"sample\shape_voc\train_classes.txt"
    input_shape = (416, 416)


    classes_list = read_classes(classes_path)
    with open(annotation_path) as f:
        lines = f.readlines()

    result_dict = {}

    for classes_name in classes_list:
        result_dict[classes_name] = []

    for line in lines:
        image_data, box_data = resize_image(line, input_shape)
        for box in box_data:
            if not (box == np.zeros(5, dtype=float)).all():
                width = int(box[2] - box[0])
                height = int(box[3] - box[1])
                result_dict[classes_list[int(box[4])]].append([width, height])

    print("NUMBER: ")
    final_result = []

    sort_list = []
    for n in classes_list:
        result_dict[n] = removal(result_dict[n])
        aa = "{},{}".format(n, len(result_dict[n]))

        sort_list.append(Num(n, len(result_dict[n])))

        final_result = final_result + result_dict[n]

    sort_list.sort(reverse=True)
    for i, a in enumerate(sort_list):
        print(i, end=" ")
        a.print_info()
    

    print(len(final_result))
    get_kmeans(9, final_result)
