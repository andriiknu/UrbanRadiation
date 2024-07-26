#=======================================================================================================================
# This file traines a Convolutional neural network (CNN) and a Multi layer perceptron (MLP) neural network
# using training features generated in script 02
# Model weights will be saved in weights/ folder
# @ Shaikat Galib / smg478@mst.edu / 25/04/2019
#=======================================================================================================================

import sys
from main_funcs import train
#=======================================================================================================================
# bash train.sh /data/training/ /data/trainingAnswers.csv

########################################################################################################################
expt_name = 'ANN_CNN'
train_folder = './data/training/'
train_answr = './data/trainingAnswers.csv'
wdata_dir = './wdata/'
if len(sys.argv) > 1:
    train_folder = sys.argv[1]
    train_answr = sys.argv[2]
########################################################################################################################

train(expt_name=expt_name, wdata_dir=wdata_dir, seed=203)