# Checkerboard Stereo RGB/DVS - Depth - IMU calibration
# Developed by Abanob Soliman, IBISC Laboratory, France

import sys
sys.path.append('../src')
from backend_HCALIB import *
from HCALIBevaluate import *

#setup the argument list
parser = argparse.ArgumentParser(description='Run HCALIB on an IBISCape type dataset.')
parser.add_argument('--folder',  metavar='folder', nargs='?', help='Data folder')
# parser.add_argument('--output-bag', metavar='output_bag',  default="output.bag", help='ROS bag file %(default)s')
#print help if no argument is specified
if len(sys.argv)<1:
    parser.print_help()
    sys.exit(0)
#parse the args
parsed = parser.parse_args()

# Don't forget to change the sequence name and also choose the Left/Right rgb camera
data_dir = os.path.join(os.getcwd(),  "/media/abanobsoliman/My Passport1/WACV23/EuRoC/HCALIB/EuRoC/")   # "/media/abanobsoliman/My Passport/BMVC22"    or    "../data"     or    parsed.folder
path1 = os.path.join(data_dir, "V102") # Change sequence name here! (GPS 10Hz, IMU 200Hz, GT (aligned))
path2 = os.path.join(path1, "cam1/data/") # rgb Right camera choice

# Inserting the Sensors readings (RGB - Depth - IMU) and the ground-truth poses
col_list = ["#timestamp [ns]"]
rgbR_path = glob.glob(os.path.join(path2, '*.png'))
rgbR_time = pd.read_csv(path1+'/cam1/data.csv', delimiter=',', usecols=col_list).values
imu_time = pd.read_csv(path1+'/imu.csv', delimiter=',', usecols=col_list).values
gyro_accel = ["w_RS_S_x [rad s^-1]","w_RS_S_y [rad s^-1]","w_RS_S_z [rad s^-1]","a_RS_S_x [m s^-2]","a_RS_S_y [m s^-2]","a_RS_S_z [m s^-2]"]
imu_meas = pd.read_csv(path1+'/imu.csv', delimiter=',', usecols=gyro_accel).values
imu_meas[:,2] *= -1
arrange = [1,2,3,0]
quat_IMU = imu_quat(imu_meas[:,3],imu_meas[:,4],imu_meas[:,5],imu_meas[:,0],imu_meas[:,1],imu_meas[:,2],imu_time*1e-9).T
roll_IMU = Rotation.from_quat(quat_IMU[:,arrange]).as_mrp()[:,0]
pitch_IMU = Rotation.from_quat(quat_IMU[:,arrange]).as_mrp()[:,1]
yaw_IMU = Rotation.from_quat(quat_IMU[:,arrange]).as_mrp()[:,2]
gyro_meas = np.vstack((roll_IMU,pitch_IMU,yaw_IMU)).T
Poses_path = np.loadtxt(path1+'/groundtruth.txt')
gt_time = Poses_path[:,0]
Poses_path[:,1] = (Poses_path[:,1] - Poses_path[0,1])
Poses_path[:,2] = (Poses_path[:,2] - Poses_path[0,2])
Poses_path[:,3] = (Poses_path[:,3] - Poses_path[0,3])
roll = Rotation.from_quat(Poses_path[:,4:]).as_mrp()[:,0]
pitch = Rotation.from_quat(Poses_path[:,4:]).as_mrp()[:,1]
yaw = Rotation.from_quat(Poses_path[:,4:]).as_mrp()[:,2]
GT_quat = Rotation.from_mrp(np.array([roll,pitch,yaw]).T).as_quat()
arrange2 = [3,0,1,2]
gt_pos = Poses_path[:,1:4]
gt_quat = GT_quat[:,arrange2]
Poses_path = np.hstack((gt_quat,gt_pos))
gps_meas0 = np.loadtxt(path1+'/gp_measurements_freq_1.0_hz_std_0.10_m.txt')
gps_meas0[:,1] -= gps_meas0[0,1]
gps_meas0[:,2] -= gps_meas0[0,2]
gps_meas0[:,3] -= gps_meas0[0,3]
Xgps, Ygps, Zgps = [gps_meas0[:,1], gps_meas0[:,2], gps_meas0[:,3]]
gps_time = gps_meas0[:,0]
gps_meas = gps_meas0[:,1:]
gt_euroc = pd.read_csv(path1+'/data.csv', delimiter=',').values
gt_vel = gt_euroc[:,8:11]

# Loading V201 comparison CT-SLAM and BASALT
#ct_slam = np.loadtxt(path1+'/ct_slam.txt')
#ct_slam[:,1] -= ct_slam[0,1]
#ct_slam[:,2] -= ct_slam[0,2]
#ct_slam[:,3] -= ct_slam[0,3]
#basalt = np.loadtxt(path1+'/basalt.txt')

# B-spline interpolation
n_spline = 3  # cumulative B-spline order
f_spline = 1000
print('GPS on B-spline manifold start!')
u_spline = np.linspace(0., 1., f_spline)
CT_GPS = bspline.cumul_b_splineR3(gps_meas.T, u_spline, n_spline)
X_CT_GPS = CT_GPS[0,:]
Y_CT_GPS = CT_GPS[1,:]
Z_CT_GPS = CT_GPS[2,:]
gps_meas = np.vstack((X_CT_GPS,Y_CT_GPS,Z_CT_GPS)).T
gps_Ctime = np.linspace(rgbR_time[0], rgbR_time[-1], CT_GPS.shape[1])

gt_time = np.unique([gt_time])
imu_time = np.unique([imu_time]) * 1e-9
rgbR_time = np.unique([rgbR_time]) * 1e-9
gps_Ctime = np.unique([gps_Ctime]) * 1e-9

gt_idx = assign(rgbR_time, gt_time).astype(int)
imu_idx = assign(rgbR_time, imu_time).astype(int)
gps_idx = assign(rgbR_time, gps_Ctime).astype(int)
print('GPS on B-spline manifold finish successfully!')

rgbR_images = load_images_from_folder(rgbR_path)
print('Right RGB Total frames:', len(rgbR_images))

# remove old
os.remove(data_dir+"hcalib_euroc_opt/traj_velo.csv")
os.remove(data_dir+"hcalib_euroc_opt/imu_states.csv")
os.remove(data_dir+"hcalib_euroc_opt/proposed_lvl1.txt")
os.rmdir(data_dir+"hcalib_euroc_opt")

# create new
try:
    path_new = os.path.join(data_dir, "hcalib_euroc_opt")
    os.mkdir(path_new)
except OSError:
    print("Creation of the directory %s failed" % path_new)
else:
    print("Successfully created the directory %s " % path_new)
optimized = open(os.path.join(path_new, 'proposed_lvl1.txt'), "a")
optimized.write('#timestamp[nsec] tx ty tz qw qx qy qz\n')
TrajVelocity = open(os.path.join(path_new, 'traj_velo.csv'), "a")
writer5 = csv.writer(TrajVelocity, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer5.writerow(['#Frame_Number', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'vx', 'vy', 'vz'])
IMUStates = open(os.path.join(path_new, 'imu_states.csv'), "a")
writer6 = csv.writer(IMUStates, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer6.writerow(['#Frame_Number', 'bgx', 'bgy', 'bgz', 'bax', 'bay', 'baz', 'tic'])

# rgb intrinsics matrix
image_w = 480
image_h = 752
K_rgb = np.identity(3)
K_rgb[0, 0] = 456.134
K_rgb[1, 1] = 457.587
K_rgb[0, 2] = 255.238
K_rgb[1, 2] = 379.999

# RANSAC Visual-Odometry Pipeline based-on ShiTomasi features and KLT tracking
#Ric = np.array([[0.0125552670891, -0.999755099723, 0.0182237714554],[0.999598781151, 0.0130119051815, 0.0251588363115],[-0.0253898008918, 0.0179005838253, 0.999517347078]])
Ric = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
Posess = mono_VO_ORB(path2, K_rgb, image_w, image_h, Ric)
print("VO estimated posses for %d frames" % (len(Posess)))
writer5.writerow([0, Posess[0,3], Posess[0,4], Posess[0,5], Posess[0,6], Posess[0,0], Posess[0,1], Posess[0,2], 0.0, 0.0, 0.0])
writer6.writerow([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(abs(float(rgbR_time[0])-float(imu_time[imu_idx[0,0]]))*1e-9)])
dt_cam = (rgbR_time[1] - rgbR_time[0]) * 1e-9
for i in range(1, len(Posess)):
    writer5.writerow([i, Posess[i,3], Posess[i,4], Posess[i,5], Posess[i,6], Posess[i,0], Posess[i,1], Posess[i,2], 0.0, 0.0, 0.0])
    writer6.writerow([i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(abs(float(rgbR_time[i])-float(imu_time[imu_idx[0,i]]))*1e-9)])
TrajVelocity.close()
IMUStates.close()

Traj_list = ['qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'vx', 'vy', 'vz']
traj_pose_vel = pd.read_csv(data_dir+'/hcalib_euroc_opt/traj_velo.csv', delimiter=',', usecols=Traj_list).values

States_list = ['bgx', 'bgy', 'bgz', 'bax', 'bay', 'baz', 'tic']
imu_states = pd.read_csv(data_dir+'/hcalib_euroc_opt/imu_states.csv', delimiter=',', usecols=States_list).values

# Start Back-end Global Optimization
print("\n---------Starting Level 1 Optimization!---------\n")
T_w_i = FE_PGO(traj_pose_vel, gps_meas[gps_idx][0], gyro_meas[imu_idx][0])
np.savetxt(optimized, np.vstack([rgbR_time.T, T_w_i.T]).T, delimiter=' ')
T_w_i = np.roll(T_w_i, 4, axis=1)
traj_pose_vel[:, :7] = T_w_i

# Calculating the evaluation metrics after calibration
print("\n------------Level 1 Optimization------------")
stamps, trans_error, rot_error = RPE_calc(traj_pose_vel, Poses_path[gt_idx][0])  # args(estimated, ground-truth)  ---> Before Opt

# Plotting the Overall Relative Pose Translational and Rotational Errors
fig = mpplot.figure()
ax = fig.add_subplot(1,2,1)
ax.plot(stamps,trans_error * 100.0,'-',color="red")
ax.set_ylabel('Relative Pose Translational Error [cm]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_xlabel('Camera Frame')
ax = fig.add_subplot(1,2,2)
ax.plot(stamps,rot_error * 180.0 / np.pi,'-',color="red")
ax.set_xlabel('Camera Frame')
ax.set_ylabel('Relative Pose Orientation Error [deg]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
fig.savefig(path_new+'vid_err.pdf')

# Plotting the 2D Traj after calibration
xs = Poses_path[:,4]
ys = Poses_path[:,5]
zs = Poses_path[:,6]
#scale_x = np.mean(abs(xs))/np.mean(abs(ct_slam[:,1]))
#scale_y = np.mean(abs(ys))/np.mean(abs(ct_slam[:,3]))
#scale_z = np.mean(abs(zs))/np.mean(abs(ct_slam[:,2]))
fig0 = mpplot.figure()
ax = fig0.add_subplot(1,1,1)
#ax.plot(scale_y*Posess[:,1],scale_x*Posess[:,0],'-',color="red")
ax.plot(ys,xs,'-',color="green", label='GT')
ax.plot(traj_pose_vel[:,5],traj_pose_vel[:,4],'-',color="blue", label='Proposed (Lvl. 1)')
#ax.plot(scale_y*ct_slam[:,3],scale_x*ct_slam[:,1],'-',color="red", label='CT-SLAM')
#ax.plot(basalt[:,2],basalt[:,1],'-',color="orange", label='BASALT')
# ax.plot(Ygps,Xgps,'.',color="brown", markersize=3, label='GPS')
# ax.plot(Y_CT_GPS,X_CT_GPS,'-',color="brown", label="CT-GPS")
ax.set_xlabel('Y [m]')
ax.set_ylabel('X [m]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.legend(["GT", "Proposed (Lvl. 1)"], loc ="lower left")
# ax.legend(["GT", "Proposed (Lvl. 1)", "ORB-SLAM3", "BASALT"], loc ="lower left")
# ax.legend(loc='upper center', bbox_to_anchor=(-0.75, 1.37), ncol = len(ax.lines) )
# ax.legend(["KLT-VO", "Lvl 1 (Ours))", "GT", "GPS", "CT-GPS"], loc ="lower left")
fig0.savefig(path_new+'traj_2d.pdf')

# Plotting the 3D Traj after calibration
fig1 = mpplot.figure()
ax = fig1.add_subplot(1,1,1, projection='3d')
#ax.plot3D(scale_x*Posess[:,0], -scale_y*Posess[:,1], -scale_z*Posess[:,2],'-',color="red")
ax.plot3D(xs,-ys,-zs,'-',color="green", label='GT')
ax.plot3D(traj_pose_vel[:,4],-traj_pose_vel[:,5],-traj_pose_vel[:,6],'-',color="blue", label='Proposed (Lvl. 1)')
#ax.plot3D(scale_x*ct_slam[:,1],-scale_y*ct_slam[:,3],scale_z*ct_slam[:,2],'-',color="red", label='CT-SLAM')
#ax.plot3D(basalt[:,1],-basalt[:,2],-basalt[:,3],'-',color="orange", label='BASALT')
# ax.plot3D(Xgps,-Ygps,-Zgps,'.',color="brown", markersize=3, label='GPS')
# ax.plot3D(X_CT_GPS,-Y_CT_GPS,-Z_CT_GPS,'-',color="brown", label="CT-GPS")
ax.set_xlabel('X [m]]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend(["GT", "Proposed (Lvl. 1)"], loc ="lower left")
# ax.legend(["GT", "Proposed (Lvl. 1)", "ORB-SLAM3", "BASALT"], loc ="lower left")
# ax.legend(loc='upper center', bbox_to_anchor=(-0.75, 1.37), ncol = len(ax.lines) )
# ax.legend(["KLT-VO", "Proposed (Lvl 1)", "GT", "GPS", "CT-GPS"], loc ="lower left")
fig1.savefig(path_new+'traj_3d.pdf')

fig0 = mpplot.figure()
ax = fig0.add_subplot(3,1,1)
ax.plot(imu_time,roll_IMU,'--',color="red", label="$\phi-RK4$")
ax.plot(gt_time,roll,'-',color="red", label="$\phi-GT$")
ax.set_title('RK4 - Evaluation')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('Roll [rad]')
ax = fig0.add_subplot(3,1,2)
ax.plot(imu_time,pitch_IMU,'--',color="green", label=r"$\theta-RK4$")
ax.plot(gt_time,pitch,'-',color="green", label=r"$\theta-GT$")
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('Pitch [rad]')
ax = fig0.add_subplot(3,1,3)
ax.plot(imu_time,yaw_IMU,'--',color="blue", label="$\psi-RK4$")
ax.plot(gt_time,yaw,'-',color="blue", label="$\psi-GT$")
ax.set_ylabel('Yaw [rad]')
ax.set_xlabel('Time [nsec]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.legend(["RK4", "GT"], loc ="upper right")
fig0.savefig(path_new+'vid_rk4.pdf')

lin_vel = np.zeros((len(rgbR_time),3))
dt = (rgbR_time[1] - rgbR_time[0])
lin_vel[1:, 0] = (traj_pose_vel[:-1,4] - traj_pose_vel[1:,4]) / dt
lin_vel[1:, 1] = (traj_pose_vel[:-1,5] - traj_pose_vel[1:,5]) / dt
lin_vel[1:, 2] = (traj_pose_vel[:-1,6] - traj_pose_vel[1:,6]) / dt
mod_gt_time = np.linspace(rgbR_time[0],rgbR_time[-1],len(gt_time))
fig0 = mpplot.figure()
ax = fig0.add_subplot(3,1,1)
ax.plot(rgbR_time,lin_vel[:, 0],'--',color="red", label="$\phi-RK4$")
ax.plot(mod_gt_time,-gt_vel[:, 0],'-',color="red", label="$\phi-GT$")
ax.set_title('Estimated Velocity - Evaluation')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('Vx [m/s]')
ax = fig0.add_subplot(3,1,2)
ax.plot(rgbR_time,lin_vel[:, 1],'--',color="green", label=r"$\theta-RK4$")
ax.plot(mod_gt_time,-gt_vel[:, 1],'-',color="green", label=r"$\theta-GT$")
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('Vy [m/s]')
ax = fig0.add_subplot(3,1,3)
ax.plot(rgbR_time,lin_vel[:, 2],'--',color="blue", label="$\psi-RK4$")
ax.plot(mod_gt_time,-gt_vel[:, 2],'-',color="blue", label="$\psi-GT$")
ax.set_ylabel('Vz [m/s]')
ax.set_xlabel('Time [nsec]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.legend(["Ours", "GT"], loc ="upper right")
fig0.savefig(path_new+'vid_vel.pdf')
mpplot.show()