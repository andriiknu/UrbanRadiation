import sys
from main_funcs import predict_3tta

#=======================================================================================================================
#  bash test.sh /data/testing/ solution.csv
########################################################################################################################
test_folder = './data/testing/'
solution_fn = 'solution.csv'
wdata_dir = './wdata/'
if len(sys.argv) > 1:
    test_folder = sys.argv[1]
    solution_fn = sys.argv[2]

predict_3tta(test_folder, wdata_dir)