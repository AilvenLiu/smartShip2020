#!/usr/bin/env python
# encoding: utf-8

########################################################################
#	Copyright © 2020 OUC_LiuX, All rights reserved. 
#	File Name: mknewset.py
#	Author: OUC_LiuX 	Mail:liuxiang@stu.ouc.edu.cn 
#	Version: V1.0.0 
#	Date: 2020-12-07 Mon 22:37
#	Description: 
#	History: 
########################################################################

import os 
import sys
from shutil import copyfile

root = "/home/ailven/data/Competition/scaled-yolov4/data/"
for xml in os.listdir(root+"xml"):
    if os.path.isfile(root+"images/train_bak/"+xml.replace("xml", "jpg")):
        copyfile(root+"images/train_bak/"+xml.replace("xml", "jpg"), 
        root+"images/train/"+xml.replace("xml", "jpg"))
