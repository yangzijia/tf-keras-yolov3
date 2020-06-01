import xml.etree.ElementTree as ET
import os
import json


def get_content(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def load_classes_dict(us_path, zh_path):
    """
        将中文的标签改为英文
    """

    classes_dict = dict()

    us_list = get_content(us_path)
    zh_list = get_content(zh_path)
    #
    # if len(us_list) != len(zh_list):
    #     print("ERROR: us number != zh number")
    #     exit()
    #
    # for i in range(len(us_list)):
    #     classes_dict[zh_list[i]] = us_list[i]

    return zh_list, us_list


def gen_coco(json_file, xml_dir, us_path, zh_path):
    zh_list, us_list = load_classes_dict(us_path, zh_path)

    coco_dict = {
        "info": {
            "description": "",
            "url": "",
            "version": "",
            "year": 2020,
            "contributor": "",
            "date_created": "2020-04-14 01:45:18.567988"
        },
        "licenses": [
            {
                "id": 1,
                "name": "",
                "url": ""
            }
        ],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # add categories
    for i, cls_name in enumerate(us_list):
        coco_dict["categories"].append({
                                    "id": i,
                                    "name": cls_name,
                                    "supercategory": "None"
                                })

    obj_index = 0

    for image_id, name in enumerate(os.listdir(xml_dir)):

        img_dict = {"id": image_id, "file_name": "", "width": 0, "height": 0,
                    "date_captured": "2020-04-14 01:45:18.567975", "license": 1, "coco_url": "", "flickr_url": ""}

        xml_path = xml_dir + "/" + name

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        filename = root.find('filename')
        if filename is not None:
            img_dict["file_name"] = root.find("filename").text + ".jpg"

        if size is not None:
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)
            img_dict["width"] = img_w
            img_dict["height"] = img_h

        coco_dict["images"].append(img_dict)

        for obj in root.iter('object'):
            difficult = obj.find('difficult')
            if difficult is not None:
                iscrowd = int(difficult.text)
            else:
                iscrowd = 0
            category_id = zh_list.index(obj.find('name').text)
            xmlbox = obj.find('bndbox')
            xmin, xmax, ymin, ymax = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                                      float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            seg = [[bbox[0], bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0] + bbox[2], bbox[1]]]
            areas = bbox[2] * bbox[3]
            obj_dict = {"id": obj_index, "image_id": image_id, "category_id": category_id, "iscrowd": iscrowd,
                        "area": areas, "bbox": bbox, "segmentation": seg}
            obj_index += 1

            coco_dict["annotations"].append(obj_dict)

    json.dump(coco_dict, open(json_file, 'w'))


if __name__ == '__main__':
    us_path = "datasets/garbage/train_classes_us.txt"
    zh_path = "datasets/garbage/train_classes_zh.txt"

    xml_dir = "datasets/garbage/xml/train"
    json_file = "datasets/garbage/annotations/instances_train.json"
    gen_coco(json_file, xml_dir, us_path, zh_path)
