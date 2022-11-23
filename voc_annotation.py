"""生成my_train.txt、my_val.txt, 获得mydata/ImageSets里面的txt"""

import os
import random
import xml.etree.ElementTree as ET


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


# annotation_mode用于指定该文件运行时计算的内容
# annotation_mode=0  代表整个标签处理过程，包括获得mydata/ImageSets里面的txt以及训练用的my_train.txt、my_val.txt
# annotation_mode=1  代表获得mydata/ImageSets里面的txt
# annotation_mode=2  代表获得训练用的my_train.txt、my_val.txt
annotation_mode = 0

classes_path = 'F:/05-pycharm/02-net/model_data/classes.txt'  # 用于生成train.txt、val.txt的目标信息，仅在annotation_mode为0和2的时候有效
train_val_per = 0.9  # 指定(训练集+验证集)与测试集的比例，        (训练集+验证集):测试集 = 9:1
train_per = 0.9      # 指定(训练集+验证集)中训练集与验证集的比例， 训练集:验证集 = 9:1  仅在annotation_mode为0和1的时候有效
data_path = 'F:/05-pycharm/02-net/mydata'  # 指向数据集所在的文件夹，默认指向根目录下的数据集
data_sets = ['train', 'val']
classes, _ = get_classes(classes_path)


def convert_annotation(image_id, list_file):
    in_file = open(os.path.join(data_path, 'Annotations\%s.xml' % image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xml_box = obj.find('bndbox')
        b = (int(float(xml_box.find('xmin').text)), int(float(xml_box.find('ymin').text)),
             int(float(xml_box.find('xmax').text)), int(float(xml_box.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xml_filepath = os.path.join(data_path, './Annotations')
        saveBasePath = os.path.join(data_path, './ImageSets/Main')
        temp_xml = os.listdir(xml_filepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)
        list = range(num)
        tv = int(num * train_val_per)
        tr = int(tv * train_per)
        train_val = random.sample(list, tv)
        train = random.sample(train_val, tr)

        print("train and val size", tv)
        print("train size", tr)
        f_train_val = open(os.path.join(saveBasePath, 'train_val.txt'), 'w')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        f_train = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        f_val = open(os.path.join(saveBasePath, 'val.txt'), 'w')

        for i in list:
            name = total_xml[i][:-4] + '\n'
            if i in train_val:
                f_train_val.write(name)
                if i in train:
                    f_train.write(name)
                else:
                    f_val.write(name)
            else:
                ftest.write(name)

        f_train_val.close()
        f_train.close()
        f_val.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate my_train.txt and my_val.txt for train.")
        for image_set in data_sets:
            image_ids = open(os.path.join(data_path, 'ImageSets\Main\%s.txt' % image_set),
                             encoding='utf-8').read().strip().split()
            list_file = open('my_%s.txt' % image_set, 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s\JPEGImages\%s.jpg' % (os.path.abspath(data_path), image_id))

                convert_annotation(image_id, list_file)
                list_file.write('\n')
            list_file.close()
        print("Generate my_train.txt and my_val.txt for train done.")
