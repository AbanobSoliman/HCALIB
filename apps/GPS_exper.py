import sys
sys.path.append('../src')
from backend_HCALIB import *
from HCALIBevaluate import *

def rmse_norm(y1, y2):
    return np.linalg.norm(y1 - y2) / np.sqrt(len(y1))

gps_data = '../data/EuRoC_V201/SLAM/'
gps_10hz = np.loadtxt(gps_data+'gp_measurements_freq_10.0_hz_std_0.10_m.txt')
gps_5hz = np.loadtxt(gps_data+'gp_measurements_freq_5.0_hz_std_0.10_m.txt')
gps_1hz = np.loadtxt(gps_data+'gp_measurements_freq_1.0_hz_std_0.10_m.txt')

gt_data = np.loadtxt(gps_data+'groundtruth.txt')

# CT Splines
u_spline = np.linspace(0., 1., 100)
gps_10hz_3 = bspline.cumul_b_splineR3(gps_10hz[:,1:].T, u_spline, 3).T
gps_10hz_3_Ctime = np.linspace(gps_10hz[0,0], gps_10hz[-1,0], gps_10hz_3.shape[0])
gps_10hz_3_idx = assign(gt_data[:,0], gps_10hz_3_Ctime).astype(int)
rmse_10hz_3 = rmse_norm(gps_10hz_3[gps_10hz_3_idx[0,:]],gt_data[:,1:4])
print(rmse_10hz_3)

gps_10hz_5 = bspline.cumul_b_splineR3(gps_10hz[:,1:].T, u_spline, 5).T
gps_10hz_5_Ctime = np.linspace(gps_10hz[0,0], gps_10hz[-1,0], gps_10hz_5.shape[0])
gps_10hz_5_idx = assign(gt_data[:,0], gps_10hz_5_Ctime).astype(int)
rmse_10hz_5 = rmse_norm(gps_10hz_5[gps_10hz_5_idx[0,:]],gt_data[:,1:4])
print(rmse_10hz_5)

gps_10hz_7 = bspline.cumul_b_splineR3(gps_10hz[:,1:].T, u_spline, 7).T
gps_10hz_7_Ctime = np.linspace(gps_10hz[0,0], gps_10hz[-1,0], gps_10hz_7.shape[0])
gps_10hz_7_idx = assign(gt_data[:,0], gps_10hz_7_Ctime).astype(int)
rmse_10hz_7 = rmse_norm(gps_10hz_7[gps_10hz_7_idx[0,:]],gt_data[:,1:4])
print(rmse_10hz_7)

gps_5hz_3 = bspline.cumul_b_splineR3(gps_5hz[:,1:].T, u_spline, 3).T
gps_5hz_3_Ctime = np.linspace(gps_5hz[0,0], gps_5hz[-1,0], gps_5hz_3.shape[0])
gps_5hz_3_idx = assign(gt_data[:,0], gps_5hz_3_Ctime).astype(int)
rmse_5hz_3 = rmse_norm(gps_5hz_3[gps_5hz_3_idx[0,:]],gt_data[:,1:4])
print(rmse_5hz_3)

gps_5hz_5 = bspline.cumul_b_splineR3(gps_5hz[:,1:].T, u_spline, 5).T
gps_5hz_5_Ctime = np.linspace(gps_5hz[0,0], gps_5hz[-1,0], gps_5hz_5.shape[0])
gps_5hz_5_idx = assign(gt_data[:,0], gps_5hz_5_Ctime).astype(int)
rmse_5hz_5 = rmse_norm(gps_5hz_5[gps_5hz_5_idx[0,:]],gt_data[:,1:4])
print(rmse_5hz_5)

gps_5hz_7 = bspline.cumul_b_splineR3(gps_5hz[:,1:].T, u_spline, 7).T
gps_5hz_7_Ctime = np.linspace(gps_5hz[0,0], gps_5hz[-1,0], gps_5hz_7.shape[0])
gps_5hz_7_idx = assign(gt_data[:,0], gps_5hz_7_Ctime).astype(int)
rmse_5hz_7 = rmse_norm(gps_5hz_7[gps_5hz_7_idx[0,:]],gt_data[:,1:4])
print(rmse_5hz_7)

gps_1hz_3 = bspline.cumul_b_splineR3(gps_1hz[:,1:].T, u_spline, 3).T
gps_1hz_3_Ctime = np.linspace(gps_1hz[0,0], gps_1hz[-1,0], gps_1hz_3.shape[0])
gps_1hz_3_idx = assign(gt_data[:,0], gps_1hz_3_Ctime).astype(int)
rmse_1hz_3 = rmse_norm(gps_1hz_3[gps_1hz_3_idx[0,:]],gt_data[:,1:4])
print(rmse_1hz_3)

gps_1hz_5 = bspline.cumul_b_splineR3(gps_1hz[:,1:].T, u_spline, 5).T
gps_1hz_5_Ctime = np.linspace(gps_1hz[0,0], gps_1hz[-1,0], gps_1hz_5.shape[0])
gps_1hz_5_idx = assign(gt_data[:,0], gps_1hz_5_Ctime).astype(int)
rmse_1hz_5 = rmse_norm(gps_1hz_5[gps_1hz_5_idx[0,:]],gt_data[:,1:4])
print(rmse_1hz_5)

gps_1hz_7 = bspline.cumul_b_splineR3(gps_1hz[:,1:].T, u_spline, 7).T
gps_1hz_7_Ctime = np.linspace(gps_1hz[0,0], gps_1hz[-1,0], gps_1hz_7.shape[0])
gps_1hz_7_idx = assign(gt_data[:,0], gps_1hz_7_Ctime).astype(int)
rmse_1hz_7 = rmse_norm(gps_1hz_7[gps_1hz_7_idx[0,:]],gt_data[:,1:4])
print(rmse_1hz_7)

fig0 = mpplot.figure()
ax = fig0.add_subplot(1,3,1)
ax.plot(gps_10hz[:,2],gps_10hz[:,1],'.',color="red", markersize=2)
ax.plot(gt_data[:,2],gt_data[:,1],'-',color="black", linewidth=0.5)
ax.plot(gps_10hz_3[:,1],gps_10hz_3[:,0],'-',color="blue", linewidth=0.5)
ax.plot(gps_10hz_5[:,1],gps_10hz_5[:,0],'-',color="green", linewidth=0.5)
ax.plot(gps_10hz_7[:,1],gps_10hz_7[:,0],'-',color="brown", linewidth=0.25)
ax.set_xlabel('Y [m]')
ax.set_ylabel('X [m]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_title('f = 10 [Hz]')
ax = fig0.add_subplot(1,3,2)
ax.plot(gps_5hz[:,2],gps_5hz[:,1],'.',color="red", markersize=2)
ax.plot(gt_data[:,2],gt_data[:,1],'-',color="black", linewidth=0.5)
ax.plot(gps_5hz_3[:,1],gps_5hz_3[:,0],'-',color="blue", linewidth=0.5)
ax.plot(gps_5hz_5[:,1],gps_5hz_5[:,0],'-',color="green", linewidth=0.5)
ax.plot(gps_5hz_7[:,1],gps_5hz_7[:,0],'-',color="brown", linewidth=0.5)
ax.set_xlabel('Y [m]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_title('f = 5 [Hz]')
ax = fig0.add_subplot(1,3,3)
ax.plot(gps_1hz[:,2],gps_1hz[:,1],'.',color="red", label='GPS', markersize=2)
ax.plot(gt_data[:,2],gt_data[:,1],'-',color="black", label='GT', linewidth=0.5)
ax.plot(gps_1hz_3[:,1],gps_1hz_3[:,0],'-',color="blue", label='n=3', linewidth=0.5)
ax.plot(gps_1hz_5[:,1],gps_1hz_5[:,0],'-',color="green", label='n=5', linewidth=0.5)
ax.plot(gps_1hz_7[:,1],gps_1hz_7[:,0],'-',color="brown", label='n=7', linewidth=0.5)
ax.set_xlabel('Y [m]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_title('f = 1 [Hz]')
ax.legend(loc='upper center', bbox_to_anchor=(-0.75, 1.37), ncol = len(ax.lines) )
fig0.savefig('spline.pdf')
mpplot.show()