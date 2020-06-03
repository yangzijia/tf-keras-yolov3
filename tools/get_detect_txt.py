from yolo import YOLO
import os
from tqdm import tqdm
from PIL import Image
import json


def read(path):
    data = []
    with open(path, 'r', encoding="utf8") as f:
        data = f.readlines()
    data = [c.strip() for c in data]
    return data


def get_val_data(annotation_path):
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    num_train = 12982
    val_lines = lines[num_train:]

    val_names = [v.strip() for v in val_lines]
    return val_names


def get_ground_truth(image_dir, img_names, ground_truth_dir):
    '''
    生成预测的比对map框
    '''
    os.makedirs(ground_truth_dir, exist_ok=True)
    yolo = YOLO()

    pbar = tqdm(total=len(img_names))

    for name in img_names:
        name += ".jpg"
        image_path = image_dir + "/" + name
        image = Image.open(image_path)
        _, result = yolo.detect_image(image)
        # r_image.save(os.path.join(img_save_path,name))
        result_dict = dict()
        result_dict["detection_classes"] = result['detection_classes']
        result_dict["detection_scores"] = result['detection_scores']
        result_dict["detection_boxes"] = result['detection_boxes']
        res2 = json.dumps(result_dict)
        # print(result_dict)
        save_path = ground_truth_dir + "/" + name.split(".jpg")[0] + ".txt"
        with open(save_path, 'w', encoding='utf-8') as f:  # 打开文件
            f.write(res2)  # 在文件里写入转成的json串

        pbar.update(1)
    pbar.close()


def json_to_real(txt_dir, new_txt_dir, classes_names):
    os.makedirs(new_txt_dir, exist_ok=True)
    for txt_name in os.listdir(txt_dir):
        new_txt_path = new_txt_dir + "/" + txt_name
        with open("{}/{}".format(txt_dir, txt_name), 'r', encoding='utf8')as fp:
            result = json.load(fp)
        detection_classes = result['detection_classes']
        # detection_scores = result['detection_scores']
        detection_boxes = result['detection_boxes']
        file_list = open(new_txt_path, 'a', encoding='utf8')
        for index, class_name in enumerate(detection_classes):
            print(class_name)
            box = detection_boxes[index]
            xmin = '%.1f' % box[0]  # TODO 注意不要弄错坐标的顺序
            ymin = '%.1f' % box[1]
            xmax = '%.1f' % box[2]
            ymax = '%.1f' % box[3]
            line = "{},{},{},{},{}\n".format(xmin, ymin, xmax, ymax, classes_names.index(class_name))
            # line = (' '.join([xmin, ymin, xmax, ymax]) + " " + classes_names[class_name] + '\n')
            file_list.write(line)
        file_list.close()


if __name__ == '__main__':
    # class_names = read("sample/trainval/train_classes.txt")
    # txt_dir = "2020sz/mAP/detect_result_01"
    # new_txt_dir = "2020sz/mAP/detect_result_01_new"
    # json_to_real(txt_dir, new_txt_dir, class_names)

    val_annotation_path = r"sample\trainval\VOC2007\ImageSets\Main\trainval.txt"
    img_names = get_val_data(val_annotation_path)

    image_dir = "sample/trainval/VOC2007/JPEGImages"
    ground_truth_dir = "2020sz/mAP/detect_result_01"
    get_ground_truth(image_dir, img_names, ground_truth_dir)
