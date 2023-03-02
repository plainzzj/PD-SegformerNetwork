#!/usr/bin/python
# -*- coding:utf8 -*-

import os



def ReName(filpath):
    video_list = os.listdir(filpath)
    for video_index in range(0, len(video_list)):
        full_name = os.path.join(filpath, video_list[video_index])
        video_name = video_list[video_index]
        new_name = 'IMG_0' + video_name
        os.rename(full_name, os.path.join(filpath, new_name))



if __name__ == '__main__':
    filepath = '/root/disk0/zzj7/data_set/FruitFlowerAll/FruitFlower/AppleA_Labels/'
    ReName(filepath)