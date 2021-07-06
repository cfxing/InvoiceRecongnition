
import os


BASEDIR = os.path.dirname(os.path.abspath(__file__))

checkpoints_dir = os.path.join(BASEDIR, "train_ctpn/ctpn_models/")
outputs = os.path.join(BASEDIR, "logs")

# 数据集的目录
img_dir = os.path.join(BASEDIR, 'datasets\\ICDAR2017_MLT\\JPEGImages')
xml_dir = os.path.join(BASEDIR, 'datasets\\ICDAR2017_MLT\\Annotations')

icdar17_mlt_img_dir = os.path.join(BASEDIR, "train_code\\train_ctpn\\train_data\\train_img")
icdar17_mlt_gt_dir = os.path.join(BASEDIR, "train_code\\train_ctpn\\train_data\\train_label")
num_workers = 2

train_txt_file = os.path.join(BASEDIR, r'VOC2007_text_detection/ImageSets/Main/train.txt')
val_txt_file = os.path.join(BASEDIR, r'VOC2007_text_detection/ImageSets/Main/val.txt')

anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

# bgr can find from  here: https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]



