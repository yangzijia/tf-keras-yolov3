import xml.etree.ElementTree as ET
import os


dir_path = 'sample/shape_voc'
classes_path = os.path.expanduser(r"sample\shape_voc\train_classes.txt")


def read_classes():
    with open(classes_path, "r", encoding='UTF-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

classes = read_classes()

def convert_annotation(image_id, list_file, folder_path):
    in_file = open('%s/Annotations/%s.xml'%(folder_path, image_id),'r', encoding='UTF-8')
    tree=ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

def write_to_txt(image_set, list_file, folder_path):
    image_ids = open('%s/ImageSets/Main/%s.txt'%(folder_path, image_set)).read().strip().split()
    
    for image_id in image_ids:
        if image_id != '0' and image_id != '1' and image_id != '-1':

            if not (os.path.exists('%s/JPEGImages/%s.jpg'% (folder_path, image_id)) and \
                 os.path.exists('%s/Annotations/%s.xml'%(folder_path, image_id))):
                print(image_id)
                continue

            list_file.write('%s/JPEGImages/%s.jpg'% (folder_path, image_id))
            try:
                convert_annotation(image_id, list_file, folder_path)
            except:
                print("error-->", image_id)
            list_file.write('\n')
    return list_file

list_file = open('{}.txt'.format('trainval'), 'w')

list_file = write_to_txt('trainval', list_file, dir_path)  

list_file.close()