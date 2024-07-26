#=======================================================================================================================
# This file generates an ensemble prediction using outputs from 06, 07, 08 scripts
# predictions will be saved in wdata/submits folder
# threshold = 4 would be valid choice according to out-of-fold predictions
# @ Shaikat Galib / smg478@mst.edu / 25/04/2019
#=======================================================================================================================
import sys
from main_funcs import vote_ensemble
#=======================================================================================================================
# bash test.sh /data/testing/ solution.csv

test_folder = './data/testing/'
solution_fn = 'solution.csv'
wdata_dir = './wdata/'
if len(sys.argv) > 1:
    test_folder = sys.argv[1]
    solution_fn = sys.argv[2]
#======================================================================================================================

vote_ensemble(wdata_dir)