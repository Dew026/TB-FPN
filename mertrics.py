import json
import sys
import os

#from pycocotools.mask import *
import numpy as np

from medpy import metric
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from PIL import Image

class Evaluateof2D():
    def __init__(self, pre_path=None, gt_path=None):
        self.pre_path = pre_path
        self.gt_path = gt_path
        self.threshold_confusion = 0.5
        self.target = None
        self.output = None
        # output --- predicted
        # target --- groundtruth

    def add_batch(self, batch_tar, batch_out):

        self.target = batch_tar.flatten()
        self.output = batch_out.flatten()

    # 求混淆矩阵和IoU
    def confusion_matrix(self):
        # Confusion matrix
        y_pred = self.output >= self.threshold_confusion
        confusion = confusion_matrix(self.target, y_pred)
        # print(confusion)
        iou = 0
        if float(confusion[0, 0] + confusion[0, 1] + confusion[1, 0]) != 0:
            iou = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1] + confusion[1, 0])

        return confusion, iou

    # calculating dice
    # 与f1_score相同

    def dice(self):
        pred = self.output >= self.threshold_confusion
        dice = f1_score(self.target, pred, labels=None, average='binary', sample_weight=None)
        return dice

    def HausdorffDistance(self, pre, gt):

        if np.any(pre != 0):
            hd = metric.binary.hd(pre, gt)
            a, b = gt.shape  # 获取图像高度和宽度信息

            return hd / np.sqrt(a * a + b * b)

        else:
            return 1




    def get_result(self):
        list_dice = []
        list_hd = []
        list_iou = []
        
        dice_avg = 0
        hd_avg = 0
        iou_avg = 0
        num = 0

        for file in os.listdir(os.path.join(self.gt_path)):
            pre_path = os.path.join(self.pre_path, file)
            gt_path = os.path.join(self.gt_path, file)  # 可能需要进行修改
            # print(pre_path)

            x = Image.open(pre_path)
            y = Image.open(gt_path)

            pre = np.array(x)
            gt = np.array(y)

            hd = self.HausdorffDistance(pre, gt)
            
            self.add_batch(gt, pre)
            dice = self.dice()
            confusion, iou = self.confusion_matrix()

            dice_avg += dice
            hd_avg += hd
            iou_avg += iou
            num = num + 1
            
            list_dice.append(dice)
            list_hd.append(hd)
            list_iou.append(iou)

        dice_avg = dice_avg / num
        hd_avg = hd_avg / num
        iou_avg = iou_avg / num
        # print(hd_avg)

        return dice_avg, hd_avg, iou_avg, list_dice, list_hd, list_iou

if __name__ == "__main__":
    eva_metrics = Evaluateof2D(pre_path="./dataset/valannot_student", gt_path="./dataset/valannot")
    dice_avg, hd_avg, iou_avg, list_dice, list_hd, list_iou = eva_metrics.get_result()
    print(dice_avg, hd_avg, iou_avg)
    print(np.var(list_dice), np.var(list_hd), np.var(list_iou))