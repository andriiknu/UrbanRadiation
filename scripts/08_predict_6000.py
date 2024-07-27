#=======================================================================================================================
# This file generates firsr level predictions using trained models saved in the folder weights/
# predictions will be saved in wdir/submits folder
# Predict on window size  = 6000 conts
# scan for 200 windows
# @ Shaikat Galib / smg478@mst.edu / 25/04/2019
#=======================================================================================================================

import sys
WORK_DIR = 'F:\UrbanRadiation'
sys.path.append(WORK_DIR)
from main_funcs import predict

#=======================================================================================================================
#  bash test.sh /data/testing/ solution.csv
########################################################################################################################
test_folder = './data/testing/'
solution_fn = 'solution.csv'
wdata_dir = './wdata/'
if len(sys.argv) > 1:
    test_folder = sys.argv[1]
    solution_fn = sys.argv[2]

seg_mul=3000

predict(test_folder, wdata_dir, seg_mul, const_seg_width=True)