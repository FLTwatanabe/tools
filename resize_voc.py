import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from optparse import OptionParser
from glob import glob


def process_image(file_path, output_path, x, y, save_box_images=0):
    (xml_dir, jpg_dir, file_name, ext) = get_file_name(file_path)
    image_path = '{}/{}.{}'.format(jpg_dir, file_name, ext)
    xml = '{}/{}.xml'.format(xml_dir, file_name)
    try:
        resize(
            image_path,
            xml,
            (x, y),
            output_path,
            save_box_images=save_box_images,
        )
    except Exception as e:
        print('[ERROR] error with {}\n file: {}'.format(image_path, e))
        print('--------------------------------------------------')


def draw_box(boxes, image, path):
    for i in range(0, len(boxes)):
        cv2.rectangle(image, (boxes[i][2], boxes[i][3]), (boxes[i][4], boxes[i][5]), (255, 0, 0), 1)
    cv2.imwrite(path, image)


def resize(image_path,
           xml_path,
           newSize,
           output_path,
           save_box_images=False,
           verbose=False):

    image = cv2.imread(image_path)

    scale_x = newSize[0] / image.shape[1]
    scale_y = newSize[1] / image.shape[0]

    image = cv2.resize(image, (newSize[0], newSize[1]))

    newBoxes = []
    xmlRoot = ET.parse(xml_path).getroot()
    xmlRoot.find('filename').text = image_path.split('/')[-1]
    size_node = xmlRoot.find('size')
    size_node.find('width').text = str(newSize[0])
    size_node.find('height').text = str(newSize[1])

    for member in xmlRoot.findall('object'):
        bndbox = member.find('bndbox')

        xmin = bndbox.find('xmin')
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')

        xmin.text = str(np.round(int(xmin.text) * scale_x))
        ymin.text = str(np.round(int(ymin.text) * scale_y))
        xmax.text = str(np.round(int(xmax.text) * scale_x))
        ymax.text = str(np.round(int(ymax.text) * scale_y))

        newBoxes.append([
            1,
            0,
            int(float(xmin.text)),
            int(float(ymin.text)),
            int(float(xmax.text)),
            int(float(ymax.text))
            ])

    (xml_path, _, file_name, ext) = get_file_name(image_path)
    cv2.imwrite(os.path.join(output_path, 'JPEGImages', '.'.join([file_name, ext])), image)

    tree = ET.ElementTree(xmlRoot)
    tree.write('{}/{}.xml'.format(os.path.join(output_path, 'Annotations'), file_name, ext))
    if int(save_box_images):
        save_path = '{}/boxes_images/boxed_{}'.format(output_path, ''.join([file_name, '.', ext]))
        draw_box(newBoxes, image, save_path)

def add_end_slash(path):
    if path[-1] is not '/':
        return path + '/'
    return path


def create_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def get_file_name(path):
    path = path.split('/')
    base_dir = path[:len(path)-2]
    base_dir = '/'.join(str(x) for x in base_dir)
    xml_dir = os.path.join(base_dir, 'Annotations')
    jpg_dir = os.path.join(base_dir, 'JPEGImages')
    names = path[-1].split('.')
    file_name = names[0]
    ext = names[1]
    return (xml_dir, jpg_dir, file_name, ext)



if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--voc_root',
        help='Path to dataset data ?(image and annotations).',
        required=True
    )
    parser.add_argument(
        '--out_root',
        help='Path that will be saved the resized dataset',
        default='./',
        required=True
    )
    parser.add_argument(
        '-x',
        '--new_x',
        dest='x',
        help='The new x images size',
        required=True
    )
    parser.add_argument(
        '-y',
        '--new_y',
        dest='y',
        help='The new y images size',
        required=True
    )
    parser.add_argument(
        '-s',
        '--save_box_images',
        dest='save_box_images',
        help='If True, it will save the resized image and a drawed image with the boxes in the images',
        default=0
    )

    IMAGE_FORMATS = ('.jpeg', '.png', '.jpg')

    args = parser.parse_args()

    voc_root = args.voc_root
    annotations = os.path.join(voc_root, 'Annotations')
    jpegimages = os.path.join(voc_root, 'JPEGImages')
    out_root = args.out_root
    out_annotations = os.path.join(out_root, 'Annotations')
    out_jpegimages = os.path.join(out_root, 'JPEGImages')

    create_path(out_root)
    create_path(out_annotations)
    create_path(out_jpegimages)

    images = glob(jpegimages + '/*.jpg')

    for image in tqdm(images):
        if image.endswith(IMAGE_FORMATS):
            process_image(image, out_root, int(args.x),int(args.y), args.save_box_images)