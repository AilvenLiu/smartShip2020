#!/usr/bin/env python
# encoding: utf-8

########################################################################
#	Copyright © 2020 OUC_LiuX, All rights reserved. 
#	File Name: dataReinformance.py
#	Author: OUC_LiuX 	Mail:liuxiang@stu.ouc.edu.cn 
#	Version: V1.0.0 
#	Date: 2020-12-07 Mon 22:40
#	Description: 
#	History: 
########################################################################

import sys
import os
import numpy as np
import cv2
from shutil import copyfile
import numba as nb
import concurrent.futures


@nb.jit()
def pepperSalt(imgIn):
    imgOut = np.zeros(imgIn.shape, np.uint8)
    for i in range(imgIn.shape[0]):
        for j in range(imgIn.shape[1]):
            for k in range(imgIn.shape[2]):
                rdn = np.random.random()
                if rdn < 0.025:
                    imgOut[i][j][k] = 0
                elif rdn > 0.975:
                    imgOut[i][j][k] = 255
                else:
                    imgOut[i][j][k] = imgIn[i][j][k]
    return imgOut

@nb.jit()
def addGauss(imgIn):
    imgOut = np.array(imgIn/255., dtype=np.float)
    gauss = np.random.normal(imgOut.mean(), imgOut.std(), imgOut.shape)
    imgOut = imgOut*0.9 + gauss*0.1
    imgOut = np.clip(imgOut, 0.0, 1.0)
    imgOut = np.uint8(imgOut*255)
    return imgOut

@nb.jit()
def mixGauss(imgIn):
    imgArray = np.array(imgIn/255., dtype=np.float)
    imgOut = np.zeros(imgArray.shape, np.uint8)
    gauss_noise = np.random.normal(imgArray.mean(), imgArray.std(), imgArray.shape)
    gauss_noise = np.clip( gauss_noise, 0., 1.)
    gauss_noise = np.uint8(gauss_noise*255) 
    for i in range(imgIn.shape[0]):
        for j in range(imgIn.shape[1]):
            for k in range(imgIn.shape[2]):
                rdn = np.random.random()
                if rdn < 0.05:
                    imgOut[i][j][k] = gauss_noise[i][j][k]
                else:
                    imgOut[i][j][k] = imgIn[i][j][k]
    return imgOut

@nb.jit()
def addPic(imgIn, imgTarget):
    imgTarget = cv2.resize(imgTarget, (imgIn.shape[1], imgIn.shape[0]), \
        cv2.INTER_NEAREST)
    return np.clip((imgIn*0.9+imgTarget*0.1), 0, 255)

@nb.jit()
def mixPic(imgIn, imgTarget):
    imgTarget = cv2.resize(imgTarget, (imgIn.shape[1], imgIn.shape[0]), \
        cv2.INTER_NEAREST)
    imgOut = np.zeros(imgIn.shape, np.uint8)
    for i in range(imgIn.shape[0]):
        for j in range(imgIn.shape[1]):
            for k in range(imgIn.shape[2]):
                rdn = np.random.random()
                if rdn < 0.05:
                    imgOut[i][j][k] = imgTarget[i][j][k]
                else:
                    imgOut[i][j][k] = imgIn[i][j][k]
    return imgOut


def main(img):
    img_root = "./data/images/train/"
    txt_root = "./data/labels/train/"
    # print(img)
    # np.random.seed(seed)
    imgIn = cv2.imread(img_root + img)
    mode = np.random.randint(0, 5)
    if mode == 0:       # pepperSalt
        img_new = pepperSalt(imgIn)
        cv2.imwrite(img_root + img[:-4] + "_ps.jpg", img_new)
        copyfile( txt_root + img.replace("jpg", "txt"), \
                    txt_root + img[:-4] + "_ps.txt")
        return (img+"-->"+img[:-4] + "_ps.jpg")

    elif mode == 1:     # addGauss
        img_new = addGauss(imgIn)
        cv2.imwrite(img_root + img[:-4] + "_addGauss.jpg", img_new)
        copyfile( txt_root + img.replace("jpg", "txt"), \
                    txt_root + img[:-4] + "_addGauss.txt")
        return (img+"-->"+img[:-4] + "_addGauss.jpg")

    elif mode == 2:     # mixGauss
        img_new = mixGauss(imgIn)
        cv2.imwrite(img_root + img[:-4] + "_mixGauss.jpg", img_new)
        copyfile( txt_root + img.replace("jpg", "txt"), \
                    txt_root + img[:-4] + "_mixGauss.txt")
        return (img+"-->"+img[:-4] + "_mixGauss.jpg")

    elif mode == 3:     # addPic
        target = np.random.choice(img_list)
        imgTarget = cv2.imread(img_root + target)
        img_new = addPic(imgIn, imgTarget)
        cv2.imwrite(img_root + img[:-4] + \
            "_add_{}.jpg".format(target[:-4]), img_new)
        copyfile( txt_root + img.replace("jpg", "txt"), \
                    txt_root + img[:-4] + "_add_{}.txt".format(target[:-4]))
        return (img+"-->"+img[:-4] + "_add_{}.jpg".format(target[:-4]))

    elif mode == 4:     # mixPic
        target = np.random.choice(img_list)
        imgTarget = cv2.imread(img_root + target)
        img_new = mixPic(imgIn, imgTarget)
        cv2.imwrite(img_root + img[:-4] + \
            "_mix_{}.jpg".format(target[:-4]), img_new)
        copyfile( txt_root + img.replace("jpg", "txt"), \
                    txt_root + img[:-4] + "_mix_{}.txt".format(target[:-4]))
        return (img+"-->"+img[:-4] + "_mix_{}.jpg".format(target[:-4]))

    else:
        return ("other mode occured")


if __name__ == "__main__":
    img_root = "./data/images/train/"
    txt_root = "./data/labels/train/"
    img_list = os.listdir(img_root)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for img, new_img_info in zip(img_list, executor.map(main, img_list)):
            print(new_img_info)
