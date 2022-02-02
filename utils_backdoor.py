#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import torch
import torchvision.transforms as transforms
import h5py
import numpy as np
#import tensorflow as tf
#from keras.preprocessing import image
#import cv2
from PIL import Image
import sys
sys.path.append("..")
from networks.cnn import CNN
#from model import ResNet18

def dump_image(x, filename, format):
    #img = image.array_to_img(x, scale=False)
    #img.save(filename, format)
    #return
    #cv2.imwrite(filename, x)
    filetype_str = ".png"
    if format == "png" :
        filetype_str = ".png"
    print("image.shape",x.shape)
    print("image.pixel11",x[:,1,1])
    tensor_to_image = transforms.ToPILImage()
    img = tensor_to_image(x)
    img.save(filename)


def load_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            print("h5py keys: ", keys)
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset

def load_model(model_file, device):
    net = CNN().to(device)
    model = torch.load(model_file)
    net.load_state_dict(model.state_dict())

    return net
