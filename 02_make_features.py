#=======================================================================================================================
# This file creates features for model treaining, from segmented training files
# 100 features from a 100 bin spectrum that covers 0-3000keV
# additional 51 features from peak to compton ration and peak to peak ratio using peaks associated with a source
# @ Shaikat Galib / smg478@mst.edu / 25/04/2019
#=======================================================================================================================

import sys
from pre_process import make_features


# bash train.sh /data/training/ /data/trainingAnswers.csv
########################################################################################################################
train_folder = './data/training/'
train_answr = './data/trainingAnswers.csv'
wdata_dir = './wdata/'
if len(sys.argv) > 1:
    train_folder = sys.argv[1]
    train_answr = sys.argv[2]
########################################################################################################################

make_features(wdata_dir=wdata_dir)

