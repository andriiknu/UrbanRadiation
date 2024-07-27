#=======================================================================================================================
# This file fine-tunes source location using outputs from 09 script
# predictions will be saved in wdata/submits folder and also in current directory
# Search for highest number of counts for associated peaks in 1.5*segment_length range, use seed time from 09 script,
# where, segment lenght = number of counts in test file / 30
# @ Shaikat Galib / smg478@mst.edu / 25/04/2019
#=======================================================================================================================

import sys
WORK_DIR = 'F:\UrbanRadiation'
sys.path.append(WORK_DIR)
from main_funcs import time_process

#=======================================================================================================================
# Input: output file from model prediction
#=======================================================================================================================
# bash test.sh /data/testing/ solution.csv

test_folder = './data/testing/'
solution_fn = 'solution.csv'
wdata_dir = './wdata/'
if len(sys.argv) > 1:
    test_folder = sys.argv[1]
    solution_fn = sys.argv[2]

time_process(test_folder, wdata_dir, solution_fn)