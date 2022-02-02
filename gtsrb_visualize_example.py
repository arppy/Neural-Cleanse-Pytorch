#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import os
import time

import numpy as np
import random
random.seed(123)
np.random.seed(123)

from visualizer import Visualizer
from backdoor_model import Net, GENERATORS, DETECTORS, ModelWithBackdoor, ThresholdedBackdoorDetectorStegano
from robustbench import load_model
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import utils_backdoor
import torch
from torch.utils.data import random_split

##############################
#        PARAMETERS          #
##############################

DEVICE = '0'  # specify which GPU to use

DATA_DIR = '../res/data/'  # data folder
DATA_FILE = 'gtsrb_dataset_int.h5'  # dataset file
MODEL_DIR = '../res/models/'  # model directory
MODEL_FILENAME = 'ds_random_ts-linf_4x4_eps8_objpeg_alpha01/Epoch_cifar10_N68.pkl'  # model file
IMAGE_DIR = '../res/images/'
SECRET_FILENAME = 'linf_4x4_E68/cifar10_best_secret_linf8_random_4x4_a01_b001_68_50.png'
RESULT_DIR = 'results'  # directory for storing results
# image filename template for visualization results
IMG_FILENAME_TEMPLATE = 'cifar10_visualize_%s_label_%d.png'

# input size
IMG_ROWS = 32
IMG_COLS = 32
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES = 10  # total number of classes in the model
Y_TARGET = 3  # (optional) infected target label, used for prioritizing label scanning

INTENSITY_RANGE = 'mnist'  # preprocessing method for the task, GTSRB uses raw pixel intensities

# parameters for optimization
BATCH_SIZE = 32  # batch size used for optimization
LR = 0.1  # learning rate
STEPS = 1000  # total optimization iterations
NB_SAMPLE = 1000  # number of samples in each mini batch
MINI_BATCH = NB_SAMPLE // BATCH_SIZE  # mini batch size used for early stop
INIT_COST = 1e-3  # initial weight used for balancing two objectives

REGULARIZATION = 'l1'  # reg term to control the mask's norm

ATTACK_SUCC_THRESHOLD = 0.99  # attack success threshold of the reversed attack
PATIENCE = 5  # patience for adjusting weight, number of mini batches
COST_MULTIPLIER = 2  # multiplier for auto-control of weight (COST)
SAVE_LAST = False  # whether to save the last result or best result

EARLY_STOP = True  # whether to early stop
EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
EARLY_STOP_PATIENCE = 5 * PATIENCE  # patience for early stop

# the following part is not used in our experiment
# but our code implementation also supports super-pixel mask
UPSAMPLE_SIZE = 1  # size of the super pixel
MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float) / UPSAMPLE_SIZE)
MASK_SHAPE = MASK_SHAPE.astype(int)

PRED_THRESHOLD = 64.93129

# parameters of the original injected trigger
# this is NOT used during optimization
# start inclusive, end exclusive
# PATTERN_START_ROW, PATTERN_END_ROW = 27, 31
# PATTERN_START_COL, PATTERN_END_COL = 27, 31
# PATTERN_COLOR = (255.0, 255.0, 255.0)
# PATTERN_LIST = [
#     (row_idx, col_idx, PATTERN_COLOR)
#     for row_idx in range(PATTERN_START_ROW, PATTERN_END_ROW)
#     for col_idx in range(PATTERN_START_COL, PATTERN_END_COL)
# ]

##############################
#      END PARAMETERS        #
##############################


def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_test', 'Y_test'])

    X_test = np.array(dataset['X_test'], dtype='float32')
    Y_test = np.array(dataset['Y_test'], dtype='float32')

    print('X_test shape %s' % str(X_test.shape))
    print('Y_test shape %s' % str(Y_test.shape))

    return X_test, Y_test


def visualize_trigger_w_mask(visualizer, gen, y_target,
                             save_pattern_flag=True):

    visualize_start_time = time.time()

    # initialize with random mask
    pattern = np.random.random(INPUT_SHAPE) * 255.0
    mask = np.random.random(MASK_SHAPE)

    # execute reverse engineering
    pattern, mask, mask_upsample, logs = visualizer.visualize(
        gen=gen, y_target=y_target, pattern_init=pattern, mask_init=mask)

    # meta data about the generated mask
    print('pattern, shape: %s, min: %f, max: %f' %
          (str(pattern.shape), np.min(pattern), np.max(pattern)))
    print('mask, shape: %s, min: %f, max: %f' %
          (str(mask.shape), np.min(mask), np.max(mask)))
    print('mask norm of label %d: %f' %
          (y_target, np.sum(np.abs(mask_upsample))))

    visualize_end_time = time.time()
    print('visualization cost %f seconds' %
          (visualize_end_time - visualize_start_time))

    if save_pattern_flag:
        save_pattern(pattern, mask_upsample, y_target)

    return pattern, mask_upsample, logs


def save_pattern(pattern, mask, y_target):

    # create result dir
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('pattern', y_target)))
    utils_backdoor.dump_image(pattern, img_filename, 'png')

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('mask', y_target)))
    utils_backdoor.dump_image(np.expand_dims(mask, axis=2) * 255,
                              img_filename,
                              'png')

    fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('fusion', y_target)))
    utils_backdoor.dump_image(fusion, img_filename, 'png')

    pass

def open_secret(path) :
  loader = transforms.Compose([transforms.ToTensor()])
  opened_image = Image.open(path).convert('L')
  opened_image_tensor = loader(opened_image).unsqueeze(0)
  return opened_image_tensor

def get_loaders(dataset_name, batchsize):
  #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean[dataset], std=std[dataset])])
  transform = transforms.ToTensor()
  if dataset_name == "cifar10" :
  #Open cifar10 dataset
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    val_size = 5000
  elif dataset_name == "MNIST" :
    #Open mnist dataset
    trainset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    val_size = 1000

  train_size = len(trainset) - val_size
  torch.manual_seed(43)
  train_ds, val_ds = random_split(trainset, [train_size, val_size])

  train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batchsize, shuffle=True, num_workers=2)
  val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batchsize, shuffle=True, num_workers=2)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)

  return train_loader, val_loader, test_loader

def gtsrb_visualize_label_scan_bottom_right_white_4():

    print('loading dataset')
    #X_test, Y_test = load_dataset()
    # transform numpy arrays into data generator
    #test_generator = build_data_loader(X_test, Y_test)
    train_loader, val_loader, test_loader = get_loaders("cifar10", BATCH_SIZE)
    device = torch.device('cuda:'+str(DEVICE))
    print('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    secret_file = '%s/%s' % (IMAGE_DIR, SECRET_FILENAME)
    net = Net(gen_holder=GENERATORS["gendeepsteganorigwgss"], det_holder=DETECTORS["detdeepsteganorigwgss"], image_shape=[32, 32], device=device, color_channel=3, n_mean=0.0, n_stddev=1.0/255.0, jpeg_q=50)
    net.to(device)
    loaded_net = torch.load(model_file,map_location=device)
    net.load_state_dict(loaded_net) #deepstegano_dropout05
    specific_secret = open_secret(secret_file)
    backdoor_model = ThresholdedBackdoorDetectorStegano(net.detector, specific_secret.to(device), PRED_THRESHOLD, device)
    robust_model = load_model(model_name="Gowal2021Improving_28_10_ddpm_100m", dataset="cifar10", threat_model="Linf").to(device)
    robust_model_with_backdoor = ModelWithBackdoor(backdoor_model, robust_model, device, Y_TARGET).to(device)

    # initialize visualizer
    visualizer = Visualizer(
        robust_model_with_backdoor, intensity_range=INTENSITY_RANGE, regularization=REGULARIZATION,
        input_shape=INPUT_SHAPE,
        init_cost=INIT_COST, steps=STEPS, lr=LR, num_classes=NUM_CLASSES,
        mini_batch=MINI_BATCH,
        upsample_size=UPSAMPLE_SIZE,
        attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
        patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
        img_color=IMG_COLOR, batch_size=BATCH_SIZE, verbose=2,
        save_last=SAVE_LAST,
        early_stop=EARLY_STOP, early_stop_threshold=EARLY_STOP_THRESHOLD,
        early_stop_patience=EARLY_STOP_PATIENCE)

    log_mapping = {}

    # y_label list to analyze
    y_target_list = list(range(NUM_CLASSES))
    y_target_list.remove(Y_TARGET)
    y_target_list = [Y_TARGET] + y_target_list
    for y_target in y_target_list:

        print('processing label %d' % y_target)

        _, _, logs = visualize_trigger_w_mask(
            visualizer, test_loader, y_target=y_target,
            save_pattern_flag=True)

        log_mapping[y_target] = logs

    pass


def main():

    #os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    utils_backdoor.fix_gpu_memory()
    gtsrb_visualize_label_scan_bottom_right_white_4()

    pass


if __name__ == '__main__':

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
