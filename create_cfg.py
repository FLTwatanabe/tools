import glob
import argparse
import os
import random


def create_dataset_cfg(data_name, dataset_type, class_num, ratio):
    """
    to identify train and test datasets, you would create confituration file, "train.txt", test.txt", and "mydata.data".
    """
    if dataset_type == 'yolo':
        if ratio is None:
            raise TypeError("Please indicate ratio arg")
        img_dir = data_name + "/images/*.jpg"
        #label_dir = data_name + "/labels/*"
        images = sorted(glob.glob(img_dir))
        random.shuffle(images)
        
        #labels = sorted(glob.glob(label_dir))

        train_len = int(len(images)*ratio)
        with open(data_name + '/config/' + '/train.txt', 'w') as f:
            for text in images[:train_len]:
                f.write('data/' + text + "\n")

        with open(data_name + '/config/' + '/valid.txt', 'w') as f:
            for text in images[train_len:]:
                f.write('data/' + text + "\n")

        with open(data_name + '/config' + f'/{data_name}.data', 'w') as f:
            text = f'class = {class_num} \ntrain = data/{data_name}/config/train.txt \nvalid = data/{data_name}/config/valid.txt \nnames = data/{data_name}/config/classes.txt \nbackup = data/{data_name}/config/backup'
            f.write(text)
    
    else:
        if ratio is None:
            raise TypeError("Please indicate ratio arg")
        img_dir = data_name + "/JPEGImages/*.jpg"
        #label_dir = data_name + "/labels/*"
        images = sorted(glob.glob(img_dir))
        random.shuffle(images)
        
        #labels = sorted(glob.glob(label_dir))

        train_len = int(len(images)*ratio)
        with open(data_name + '/ImageSets/Main/' + '/trainval.txt', 'w') as f:
            for text in images[:train_len]:
                text = text.split('/')[-1]
                text = text.split('.')[0]
                f.write(text + "\n")

        with open(data_name + '/ImageSets/Main/' + '/test.txt', 'w') as f:
            for text in images[train_len:]:
                text = text.split('/')[-1]
                text = text.split('.')[0]
                f.write('data/' + text + "\n")



def change_classlabel(data_name):
    """
    change class label in case you mistakenly put labels on the images
    """
    data = sorted(glob.glob('data/' + data_name + '/labels/*.txt'))
    for idx, dat in enumerate(data):
        file_name = dat.split('/')[-1]
        after = []
        with open(dat, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = lines[idx].split(" ")
                """
                change label here
                """
                line[0] = str("0")
                line = " ".join(line)
                after.append(line)
        with open('data/' + data_name + '/labels/' + file_name, 'w') as f:
            for af in after:
                f.write(af)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="name of function to exec 'cfg' 'chlabel'")
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--class_num", type=int)
    parser.add_argument("--ratio", type=float)
    opt = parser.parse_args()
    mode = opt.mode
    data_name = opt.data_name
    dataset_type = opt.dataset_type
    class_num = opt.class_num
    ratio = opt.ratio

    if mode == "cfg":
        if dataset_type == 'yolo':
            if os.path.exists(data_name + '/config'):
                pass
            else:
                os.mkdir(data_name + '/config')
            create_dataset_cfg(data_name, dataset_type, class_num, ratio)
        else:
            if os.path.exists(data_name + 'ImageSets/Main'):
                pass
            else:
                os.mkdir(data_name + 'ImageSets')
                os.mkdir(data_name + 'ImageSets/Main')
            create_dataset_cfg(data_name, dataset_type, class_num, ratio)


    elif mode == "chlabel":
        change_classlabel(data_name)
