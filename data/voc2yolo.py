#!/usr/bin/env python
# encoding: utf-8

########################################################################
#	Copyright © 2020 OUC_LiuX, All rights reserved. 
#	File Name: voc2yolo.py
#	Author: OUC_LiuX 	Mail:liuxiang@stu.ouc.edu.cn 
#	Version: V1.0.0 
#	Date: 2020-12-07 Mon 22:38
#	Description: 
#	History: 
########################################################################

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ['liner', 'container ship', 'bulk carrier', \
    'island reef', 'sailboat', 'other ship']

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    if x>=1:
        x=0.999
    if y>=1:
        y=0.999
    if w>=1:
        w=0.999
    if h>=1:
        h=0.999
    return (x,y,w,h)

def convert_annotation(rootpath, xmlname):
    xmlpath = rootpath + 'xml'
    xmlfile = os.path.join(xmlpath,xmlname)
    with open(xmlfile, "r", encoding='UTF-8') as in_file:
      txtname = xmlname[:-4]+'.txt'
      # print(txtname)
      txtpath = rootpath + '/clearLabels'  #生成的.txt文件会被保存在worktxt目录下
      if not os.path.exists(txtpath):
        os.makedirs(txtpath)
      txtfile = os.path.join(txtpath,txtname)
      with open(txtfile, "w+" ,encoding='UTF-8') as out_file:
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        out_file.truncate()
        for obj in root.iter('object'):
            # difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), \
                float(xmlbox.find('xmax').text), \
                float(xmlbox.find('ymin').text), \
                float(xmlbox.find('ymax').text))
            objWidth = b[1]-b[0]
            objHeight = b[3]-b[2]
            if objWidth < 20 and objHeight < 20 or objWidth<12 or objHeight<12:
                continue
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) \
                for a in bb]) + '\n')

def main():
    rootpath = './data/train/'
    xmlpath = rootpath + 'xml/'
    list = os.listdir(xmlpath)
    print(rootpath + ': ' + str(len(list)))
    for i in range(0,len(list)) :
        path = os.path.join(xmlpath, list[i])
        if ('.xml' in path)or('.XML' in path):
            convert_annotation(rootpath, list[i])
            # print('done', i)
        else:
            print('not xml file',i)

    rootpath = './data/train/'
    labelpath = rootpath + 'clearLabels/'
    list = os.listdir(labelpath)
    print(labelpath + ': ' + str(len(list)))
    for i in range(0,len(list)) :
        path = os.path.join(labelpath, list[i])
        with open(path) as f:
            line = f.readline()
            if not line:
                os.remove(path)
                if os.path.isfile(rootpath+'clearImgs/' +
                path.replace('txt', 'jpg')):
                    os.remove(rootpath+'clearImgs/' +
                    path.replace('txt', 'jpg'))
    list = os.listdir(labelpath)
    print(labelpath + ': ' + str(len(list)))

if __name__ == '__main__':
     main()

