import numpy as np
import random
from PIL import Image
import glob
import random
import pandas as pd
import argparse
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET

def shift_boxes_yolo(img_path, label_path, offset):
    """
    shift the position of bounding boxes on both images and labels
    
    Parameters
    ----------
    img_path : path to image file
    label : path to label file
    offset : tuple(x, y) to shift
    out_name : ex) 
    
    Returns:
    img : shifted image 
    """
    np.set_printoptions(precision=7, floatmode='maxprec')
    # load img
    img = np.asarray(Image.open(img_path).convert('RGB'))
    # load boxes raw
    boxes = np.loadtxt(label_path).reshape(-1, 5)
    # image shape
    h, w, _ = img.shape
    h_factor, w_factor = (h,w)
    # decode boxes
    x1 = np.round(w_factor * (boxes[:, 1] - boxes[:, 3] / 2)).astype(int)
    y1 = np.round(h_factor * (boxes[:, 2] - boxes[:, 4] / 2)).astype(int)
    x2 = np.round(w_factor * (boxes[:, 1] + boxes[:, 3] / 2)).astype(int)
    y2 = np.round(h_factor * (boxes[:, 2] + boxes[:, 4] / 2)).astype(int)
    # shift label boxes
    x1_ = x1 + offset[0]
    y1_ = y1 + offset[1]
    x2_ = x2 + offset[0]
    y2_ = y2 + offset[1]
    # box conditions
    cond_x1 = x1_ <= 0
    cond_y1 = y1_ <= 0
    cond_x2 = x2_ >= w
    cond_y2 = y2_ >= h
    invalid_idx = np.where(cond_x1+cond_y1+cond_x2+cond_y2)
    # encode boxes elements
    boxes[:,1] = (x1_+(x2_-x1_)/2)/w_factor
    boxes[:,2] = (y1_+(y2_-y1_)/2)/h_factor
    boxes[:,3] = (x2_-x1_)/w_factor
    boxes[:,4] = (y2_-y1_)/h_factor
    # remove boxes spreading to the image edge based on conditaions
    boxes = np.delete(boxes, invalid_idx, axis=0)
    # shift image
    pad_x = np.zeros((h,abs(offset[0]),3))
    pad_y = np.zeros((abs(offset[1]),w,3))
    if offset[0] > 0:
        img = np.concatenate([pad_x, img], axis=1)
        img = np.delete(img, slice(w, None), axis=1)
        img = img.astype('uint8')
    else:
        img = np.concatenate([img, pad_x], axis=1)
        img = np.delete(img, slice(None, abs(offset[0])), axis=1)
        img = img.astype('uint8')
    if offset[1] > 0:
        img = np.concatenate([pad_y, img], axis=0)
        img = np.delete(img, slice(h, None), axis=0)
        img = img.astype('uint8')
    else:
        img = np.concatenate([img, pad_y], axis=0)
        img = np.delete(img, slice(None, abs(offset[1])), axis=0)
        img = img.astype('uint8')
        
    return img, boxes


def shift_boxes_voc(img_path, label_path, offset):
    """
    shift the position of bounding boxes on both images and labels
    
    Parameters
    ----------
    img_path : path to image file
    label : path to label file
    offset : tuple(x, y) to shift
    out_name : ex) 
    
    Returns:
    img : shifted image 
    """
def shift_boxes_voc(img_path, xml_path, offset,):
    # load img
    img = np.asarray(Image.open(img_path).convert('RGB'))

    h, w, _ = img.shape
    
    xmlRoot = ET.parse(xml_path).getroot()
    xmlRoot.find('filename').text = img_path.split('/')[-1]

    for member in xmlRoot.findall('object'):
        bndbox = member.find('bndbox')

        xmin = bndbox.find('xmin')
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')

        # shift label boxes
        x1_ = float(xmin.text) + offset[0]
        y1_ = float(ymin.text) + offset[1]
        x2_ = float(xmax.text) + offset[0]
        y2_ = float(ymax.text) + offset[1]
        # box conditions
        cond_x1 = x1_ <= 0
        cond_y1 = y1_ <= 0
        cond_x2 = x2_ >= w
        cond_y2 = y2_ >= h

        xmin.text = str(np.round(int(x1_)))
        ymin.text = str(np.round(int(y1_)))
        xmax.text = str(np.round(int(x2_)))
        ymax.text = str(np.round(int(y2_)))

        # judge wheter object is in bad conditions
        occluded = int(member.find('occluded').text)
        invalid = cond_x1+cond_y1+cond_x2+cond_y2+occluded

        if invalid:
            xmlRoot.remove(member)

    tree = ET.ElementTree(xmlRoot)


    # shift image
    pad_x = np.zeros((h,abs(offset[0]),3))
    pad_y = np.zeros((abs(offset[1]),w,3))
    if offset[0] > 0:
        img = np.concatenate([pad_x, img], axis=1)
        img = np.delete(img, slice(w, None), axis=1)
        img = img.astype('uint8')
    else:
        img = np.concatenate([img, pad_x], axis=1)
        img = np.delete(img, slice(None, abs(offset[0])), axis=1)
        img = img.astype('uint8')
    if offset[1] > 0:
        img = np.concatenate([pad_y, img], axis=0)
        img = np.delete(img, slice(h, None), axis=0)
        img = img.astype('uint8')
    else:
        img = np.concatenate([img, pad_y], axis=0)
        img = np.delete(img, slice(None, abs(offset[1])), axis=0)
        img = img.astype('uint8')

    return img, tree


parser = argparse.ArgumentParser(description='shift image')
parser.add_argument('-i', '--input', help='path to root directory of input')
parser.add_argument('-d', '--dataset_type', help='voc or yolo')
parser.add_argument('-x', '--diff_x', type=int, help='pixel to shift on x axis')
parser.add_argument('-y', '--diff_y', type=int, help='pixel to shift on y axis')
parser.add_argument('-o', '--output', help='path to output directory')
args = parser.parse_args()


dataset_type = args.dataset_type

diff_x = args.diff_x
diff_y = args.diff_y

if dataset_type == 'yolo':
    img_out = args.output + '/images'
    label_out = args.output + '/labels'
    images_path = sorted(glob.glob(args.input + '/images/*.jpg'))
    labels_path = sorted(glob.glob(args.input + '/labels/*.txt'))

else:
    img_out = args.output + '/JPEGImages'
    label_out = args.output + '/Annotations'
    images_path = sorted(glob.glob(args.input + '/JPEGImages/*.jpg'))
    labels_path = sorted(glob.glob(args.input + '/Annotations/*.xml'))


if os.path.exists(img_out):
    pass
else:
    os.mkdir(img_out)

if os.path.exists(label_out):
    pass
else:
    os.mkdir(label_out)


offset_list = [tuple((random.randint(-diff_x,diff_x),random.randint(-diff_y,diff_y))) for i in range(10)]

for i, (image_path, label_path) in tqdm(enumerate(zip(images_path, labels_path))):
    for j, offset in enumerate(offset_list):
        if dataset_type == 'yolo':
            n_img, n_label = shift_boxes_yolo(image_path, label_path, offset)
            Image.fromarray(n_img).save(img_out+f'/{i:04d}_{j:03d}.jpg')
            n_label = pd.DataFrame(n_label)
            n_label[0] = n_label[0].astype(int)
            n_label.to_csv(label_out+f'/{i:04d}_{j:03d}.txt', header=None, index=None, sep=' ', mode='a', float_format='%.7f')
        else:
            n_img, n_label = shift_boxes_voc(image_path, label_path, offset)
            Image.fromarray(n_img).save(img_out+f'/{i:04d}_{j:03d}.jpg')
            n_label.write(label_out+f'/{i:04d}_{j:03d}.xml')