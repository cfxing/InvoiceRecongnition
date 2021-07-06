# -*- coding:utf-8 -*-
# '''
# Created on 18-12-27 上午10:34
#
# @Author: Greg Gao(laygin)
# '''

import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import IMAGE_MEAN
from train_ctpn.ctpn_utils import cal_rpn


def readxml(path):
    gt_boxes = []
    imgfile = ''
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'filename' in elem.tag:
            imgfile = elem.text
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    x_min = int(round(float(attr.find('x_min').text)))
                    y_min = int(round(float(attr.find('y_min').text)))
                    x_max = int(round(float(attr.find('x_max').text)))
                    y_max = int(round(float(attr.find('y_max').text)))

                    gt_boxes.append((x_min, y_min, x_max, y_max))

    return np.array(gt_boxes), imgfile


# for ctpn text detection
class VOCDataset(Dataset):
    def __init__(self,
                 data_dir,
                 labels_dir):
        '''

        :param txtfile: image name list text file
        :param data_dir: image's directory
        :param labels_dir: annotations' directory
        '''
        if not os.path.isdir(data_dir):
            raise Exception('[ERROR] {} is not a directory'.format(data_dir))
        if not os.path.isdir(labels_dir):
            raise Exception('[ERROR] {} is not a directory'.format(labels_dir))

        self.data_dir = data_dir
        self.img_names = os.listdir(self.data_dir)
        self.labels_dir = labels_dir

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        print(img_path)
        xml_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.xml'))
        gt_box, _ = readxml(xml_path)
        img = cv2.imread(img_path)
        h, w, c = img.shape

        # clip image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gt_box[:, 2] - 1
            newx2 = w - gt_box[:, 0] - 1
            gt_box[:, 0] = newx1
            gt_box[:, 2] = newx2

        [cls, regr], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gt_box)

        m_img = img - IMAGE_MEAN

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        cls = np.expand_dims(cls, axis=0)

        # transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr


class ICDARDataset(Dataset):
    def __init__(self, data_dir, labels_dir):
        """
        load ICDAR Dataset
        :param data_dir: images' directory
        :param labels_dir: annotations' directory
        """
        if not os.path.isdir(data_dir):
            raise Exception('[ERROR] {} is not a directory'.format(data_dir))
        if not os.path.isdir(labels_dir):
            raise Exception('[ERROR] {} is not a directory'.format(labels_dir))

        self.data_dir = data_dir
        self.img_names = os.listdir(self.data_dir)
        self.labels_dir = labels_dir

    def __len__(self):
        return len(self.img_names)

    def box_transfer(self, coor_lists, rescale_fac=1.0):
        gt_boxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2 * i]) for i in range(4)]
            coors_y = [int(coor_list[2 * i + 1]) for i in range(4)]
            x_min = min(coors_x)
            x_max = max(coors_x)
            y_min = min(coors_y)
            y_max = max(coors_y)
            if rescale_fac > 1.0:
                x_min = int(x_min / rescale_fac)
                x_max = int(x_max / rescale_fac)
                y_min = int(y_min / rescale_fac)
                y_max = int(y_max / rescale_fac)
            gt_boxes.append((x_min, y_min, x_max, y_max))
        return np.array(gt_boxes)

    def box_transfer_v2(self, coor_lists, rescale_fac=1.0):
        gt_boxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2 * i]) for i in range(4)]
            coors_y = [int(coor_list[2 * i + 1]) for i in range(4)]
            x_min = min(coors_x)
            x_max = max(coors_x)
            y_min = min(coors_y)
            y_max = max(coors_y)
            if rescale_fac > 1.0:
                x_min = int(x_min / rescale_fac)
                x_max = int(x_max / rescale_fac)
                y_min = int(y_min / rescale_fac)
                y_max = int(y_max / rescale_fac)
            prev = x_min
            for i in range(x_min // 16 + 1, x_max // 16 + 1):
                next = 16 * i - 0.5
                gt_boxes.append((prev, y_min, next, y_max))
                prev = next
            gt_boxes.append((prev, y_min, x_max, y_max))
        return np.array(gt_boxes)

    def parse_gtfile(self, gt_path, rescale_fac=1.0):
        coor_lists = list()
        with open(gt_path, encoding='UTF-8') as f:
            content = f.readlines()
            for line in content:
                coor_list = line.split(',')[:8]
                if len(coor_list) == 8:
                    coor_lists.append(coor_list)
        return self.box_transfer_v2(coor_lists, rescale_fac)

    def draw_boxes(self, img, cls, base_anchors, gt_box):
        for i in range(len(cls)):
            if cls[i] == 1:
                pt1 = (int(base_anchors[i][0]), int(base_anchors[i][1]))
                pt2 = (int(base_anchors[i][2]), int(base_anchors[i][3]))
                img = cv2.rectangle(img, pt1, pt2, (200, 100, 100))
        for i in range(gt_box.shape[0]):
            pt1 = (int(gt_box[i][0]), int(gt_box[i][1]))
            pt2 = (int(gt_box[i][2]), int(gt_box[i][3]))
            img = cv2.rectangle(img, pt1, pt2, (100, 200, 100))
        return img

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        # print(img_path)
        img = cv2.imread(img_path)
        # for read error, use default image #
        if img is None:
            print(img_path)
            with open('error_imgs.txt', 'a') as f:
                f.write('{}\n'.format(img_path))
            img_name = 'img_2647.jpg'
            img_path = os.path.join(self.data_dir, img_name)
            img = cv2.imread(img_path)

        # for read error, use default image #

        h, w, c = img.shape
        rescale_fac = max(h, w) / 1600
        if rescale_fac > 1.0:
            h = int(h / rescale_fac)
            w = int(w / rescale_fac)
            img = cv2.resize(img, (w, h))

        gt_path = os.path.join(self.labels_dir, 'gt_' + img_name.split('.')[0] + '.txt')
        gt_box = self.parse_gtfile(gt_path, rescale_fac)

        # clip image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gt_box[:, 2] - 1
            newx2 = w - gt_box[:, 0] - 1
            gt_box[:, 0] = newx1
            gt_box[:, 2] = newx2

        [cls, regr], base_anchors = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gt_box)
        # debug_img = self.draw_boxes(img.copy(),cls,base_anchors,gt_box)
        # cv2.imwrite('debug/{}'.format(img_name),debug_img)
        m_img = img - IMAGE_MEAN

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        cls = np.expand_dims(cls, axis=0)

        # transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr


if __name__ == '__main__':
    x_min = 15
    x_max = 95
    for i in range(x_min // 16 + 1, x_max // 16 + 1):
        print(16 * i - 0.5)
