"""
    调整采集图片的亮度色度问题
    根据灰度、Gamma归一化亮度
"""
import cv2
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def bright_Norm(file_pathname):
    #  遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        # print(filename)
        image = cv2.imread(file_pathname+'/'+filename, 0)  # 0表示读取灰度图
        # print(image.shape)
        Gamma = np.log(128.0/255.0) / np.log(cv2.mean(image)[0]/255.0)
        lookUpTable = np.empty((1, 256), np.uint8)

        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, Gamma)*255.0, 0, 255)

        imageNorm = cv2.LUT(image, lookUpTable)
        cv2.imshow(filename, imageNorm)
        cv2.waitKey(0)

        cv2.imwrite('./BriNorm_image'+'/'+filename, imageNorm)

# 读取的目录
bright_Norm('./Original_image')

