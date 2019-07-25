# tools

## Resize VOC annotations and images
This code is cloned from https://github.com/italojs/resize_dataset_pascalvoc  
Resize VOC format dataset including inflation
### Example command
```
$ python resize_voc.py --voc_root data/VOC --out_root data/voc -x 512 -y 512
```

## Convert voc to yolo format
This code is cloned from https://github.com/ssaru/convert2Yolo  
Boxes with occlusion flag is excluded here.
### Example command
```
$ python voc2yolo.py --datasets VOC --img_path ./VOC/JPEGImages/ --label ./VOC/Annotations/ --convert_output_path ./yolo_/ --img_type ".jpg" --manipast_path ./ --cls_list_file ./VOC/voc.names
```
## Shift images and labels of yolo format
Inflate data by shifting boxes and labels.
### Required paramters
* -i : path to directory of input data
* -x : pixels to shift on x axis
* -y : pixels to shift on y axis
* -o : path to directory of output data
### Example command
```
# yolo format
$ python shift_label.py -i YOLO -d yolo -x 60 -y 60 -o example

# voc format
$ python shift_label.py -i data/VOC -d voc -x 50 -y 50 -o data/example
```
## Create config file
Generate config files according to dataset format.
### Required parameters
* --mode : cfg (default)
* --dataset_type : 'yolo' or ''voc
* --data_name : path to data root
* --class_num : number of classes
### Example command
```
# yolo format
$ python create_cfg.py --mode cfg --dataset_type yolo --data_name SHIFTED --class_num 1 --ratio 0.6

# voc format
$ python create_cfg.py --mode cfg --data_name data/pana_resize/ --dataset_type voc --class_num 1 --ratio 0.6
```