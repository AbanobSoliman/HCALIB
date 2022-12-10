import sys

import numpy as np

sys.path.append('../src')
from backend_HCALIB import *
from HCALIBevaluate import *

time = ['time']
list = ['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz']

gt_time = pd.read_csv('/media/abanobsoliman/My Passport/BMVC22/Evaluation/data/python/gt_RGBDI_S.csv', delimiter=';', usecols=time).values
gt_data = pd.read_csv('/media/abanobsoliman/My Passport/BMVC22/Evaluation/data/python/gt_RGBDI_S.csv', delimiter=';', usecols=list).values

bslt_time = pd.read_csv('/media/abanobsoliman/My Passport/BMVC22/Evaluation/data/python/bslt_RGBDI_S.csv', delimiter=';', usecols=time).values
bslt_data = pd.read_csv('/media/abanobsoliman/My Passport/BMVC22/Evaluation/data/python/bslt_RGBDI_S.csv', delimiter=';', usecols=list).values

os3_time = pd.read_csv('/media/abanobsoliman/My Passport/BMVC22/Evaluation/data/python/os3_RGBDI_S.csv', delimiter=';', usecols=time).values
os3_data = pd.read_csv('/media/abanobsoliman/My Passport/BMVC22/Evaluation/data/python/os3_RGBDI_S.csv', delimiter=';', usecols=list).values

dui_time = pd.read_csv('/media/abanobsoliman/My Passport/BMVC22/Evaluation/data/python/dui_RGBDI_S.csv', delimiter=';', usecols=time).values*1e9
dui_data = pd.read_csv('/media/abanobsoliman/My Passport/BMVC22/Evaluation/data/python/dui_RGBDI_S.csv', delimiter=';', usecols=list).values

bslt_idx = assign(bslt_time, gt_time).astype(int)
os3_idx = assign(os3_time, gt_time).astype(int)
dui_idx = assign(dui_time, gt_time).astype(int)

# Calculating the evaluation metrics after estimation
print("\n------------OS3_RGBD------------")
stamps, trans_error, rot_error = RPE_calc(os3_data, gt_data[os3_idx][0])  # args(estimated, ground-truth)  ---> Before Opt
print("\n------------BASALT------------")
stamps1, trans_error1, rot_error1 = RPE_calc(bslt_data, gt_data[bslt_idx][0])  # ---> After Opt
print("\n------------DUI-VIO------------")
stamps1, trans_error1, rot_error1 = RPE_calc(dui_data, gt_data[dui_idx][0])  # ---> After Opt