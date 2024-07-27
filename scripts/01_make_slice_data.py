#=======================================================================================================================
# This file creates smaller segments from training files
# For data without source, equally spaced windows were selected. Window width is: number of counts/30
# First 30 sec was omitted from training
# For data with source, 7 nearest windows from source were generated.
# @ Shaikat Galib / smg478@mst.edu / 25/04/2019
#=======================================================================================================================

import sys
WORK_DIR = 'F:\UrbanRadiation'
sys.path.append(WORK_DIR)
from pre_process import make_slice_data

########################################################################################################################
expt_name = 'training_slice'
train_folder = './data/training/'
train_answr = './data/trainingAnswers.csv'
wdata_dir = './wdata/'
if len(sys.argv) > 1:
    train_folder = sys.argv[1]
    train_answr = sys.argv[2]
########################################################################################################################

make_slice_data(expt_name=expt_name,
                train_folder=train_folder,
                train_answr=train_answr,
                wdata_dir=wdata_dir)