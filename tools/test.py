import os


file_path = r"sample\shape_voc\ImageSets\Main\trainval.txt"
with open(file_path, 'w', encoding='utf8') as f:
    for i in range(1000):
        f.write("{}\n".format(i))


q