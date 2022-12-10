# Checkerboard Stereo RGB/DVS - Depth - IMU calibration
# Developed by Abanob Soliman, IBISC Laboratory, France

import sys

import cv2
import numpy as np

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
data_dir = os.path.join(os.getcwd(),  "../data/VCU_RVI/")   # "/media/abanobsoliman/My Passport/WACV23"    or    "../data"     or    parsed.folder
path1 = os.path.join(data_dir, "hand_eye") # Change sequence name here!
path4 = os.path.join(path1, "rgbL/frames") # rgb Left camera choice
path8 = os.path.join(path1, "depth/frames") # Depth camera choice
path5 = os.path.join(path1, "other_sensors")
path9 = os.path.join(path1, "vehicle_gt")

# Inserting the Sensors readings (RGB - Depth - IMU) and the ground-truth poses
col_list = ["#timestamp [ns]"]

rgbL_path = glob.glob(os.path.join(path4, '*.png'))
rgbL_time = pd.read_csv(path1+'/rgbL/timestamps.csv', delimiter=',', usecols=col_list).values

depth_path = glob.glob(os.path.join(path8, '*.png'))
depth_time = pd.read_csv(path1+'/depth/timestamps.csv', delimiter=',', usecols=col_list).values

imu_time = pd.read_csv(path1+'/other_sensors/imu.csv', delimiter=',', usecols=col_list).values
gyro_accel = ["w_RS_S_x [rad s^-1]","w_RS_S_y [rad s^-1]","w_RS_S_z [rad s^-1]","a_RS_S_x [m s^-2]","a_RS_S_y [m s^-2]","a_RS_S_z [m s^-2]"]
imu_meas = pd.read_csv(path1+'/other_sensors/imu.csv', delimiter=',', usecols=gyro_accel).values
imu_meas[:,0] *= -1
imu_meas[:,1] *= -1
# imu_meas[:,[0,1]] = imu_meas[:,[1,0]]
# imu_meas[:,3] *= -1
# imu_meas[:,4] *= -1
# imu_meas[:,[3,4]] = imu_meas[:,[4,3]]
arrange = [1,2,3,0]
quat_IMU = imu_quat(imu_meas[:,3],imu_meas[:,4],imu_meas[:,5],imu_meas[:,0],imu_meas[:,1],imu_meas[:,2],imu_time*1e-9).T
roll_IMU = Rotation.from_quat(quat_IMU[:,arrange]).as_mrp()[:,0]
pitch_IMU = Rotation.from_quat(quat_IMU[:,arrange]).as_mrp()[:,1]
yaw_IMU = Rotation.from_quat(quat_IMU[:,arrange]).as_mrp()[:,2]
gyro_meas = np.vstack((roll_IMU,pitch_IMU,yaw_IMU)).T

Poses_path = pd.read_csv(path1+'/vehicle_gt/Tnstxyzqwxyzvxyz.csv', delimiter=',').values
gt_vel = Poses_path[:,8:11]
gt_time = Poses_path[:,0]
Poses_path[:,1] = (Poses_path[:,1] - Poses_path[0,1])
Poses_path[:,2] = (Poses_path[:,2] - Poses_path[0,2])
Poses_path[:,3] = (Poses_path[:,3] - Poses_path[0,3])
roll = Rotation.from_quat(Poses_path[:,4:8]).as_mrp()[:,0]
pitch = Rotation.from_quat(Poses_path[:,4:8]).as_mrp()[:,1]
yaw = Rotation.from_quat(Poses_path[:,4:8]).as_mrp()[:,2]
GT_quat = Rotation.from_mrp(np.array([roll,pitch,yaw]).T).as_quat()
arrange2 = [3,0,1,2]
gt_pos = Poses_path[:,1:4]
gt_quat = GT_quat[:,arrange2]
Poses_path = np.hstack((gt_quat,gt_pos))

gps_meas0 = np.zeros((len(gt_time),4))
gps_meas0[:,0] = gt_time
gps_meas0[:,1:] = Poses_path[:,4:]
gps_meas0[:,1] += np.random.normal(0.0, 0.1)
gps_meas0[:,2] += np.random.normal(0.0, 0.1)
gps_meas0[:,3] += np.random.normal(0.0, 0.1)
Xgps, Ygps, Zgps = [gps_meas0[:,1], gps_meas0[:,2], gps_meas0[:,3]]
gps_time = gps_meas0[:,0]
gps_meas = gps_meas0[:,1:]

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
gps_Ctime = np.linspace(rgbL_time[0], rgbL_time[-1], CT_GPS.shape[1])

gt_time = np.unique(gt_time-gt_time[0])
imu_time = np.unique(imu_time-imu_time[0])
rgbL_time = np.unique(rgbL_time-rgbL_time[0])
depth_time = np.unique(depth_time-depth_time[0])
gps_Ctime = np.unique(gps_Ctime-gps_Ctime[0])

dep_idx = assign(rgbL_time, depth_time).astype(int)
gt_idx = assign(rgbL_time, gt_time).astype(int)
imu_idx = assign(rgbL_time, imu_time).astype(int)
gps_idx = assign(rgbL_time, gps_Ctime).astype(int)

print('GPS on B-spline manifold finish successfully!')

os.remove(data_dir+"/hcalib_opt/rgbd_calib.yaml")
os.remove(data_dir+"/hcalib_opt/association.txt")
os.remove(data_dir+"/hcalib_opt/feature_p3d.csv")
os.remove(data_dir+"/hcalib_opt/full_cam_model.csv")
os.remove(data_dir+"/hcalib_opt/full_dcam_model.csv")
os.remove(data_dir+"/hcalib_opt/rgbL_hcalib_res.csv")
os.remove(data_dir+"/hcalib_opt/depth_hcalib_res.csv")
os.remove(data_dir+"/hcalib_opt/traj_velo.csv")
os.remove(data_dir+"/hcalib_opt/imu_states.csv")
os.remove(data_dir+"/hcalib_opt/IMU_hcalib_res.csv")
files = os.listdir(data_dir+"/hcalib_opt/cloud")
for filec in files:
    os.remove(os.path.join(data_dir+"/hcalib_opt/cloud", filec))
os.rmdir(data_dir+"/hcalib_opt/cloud")
os.rmdir(data_dir+"/hcalib_opt")
try:
    parent_dir = data_dir
    path_new = os.path.join(parent_dir, "hcalib_opt")
    os.mkdir(path_new)
    path_PCL = os.path.join(path_new, "cloud")
    os.mkdir(path_PCL)
except OSError:
    print("Creation of the directory %s failed" % path_new)
    print("Creation of the directory %s failed" % path_PCL)
else:
    print("Successfully created the directory %s " % path_new)
    print("Successfully created the directory %s " % path_PCL)
file_associate = open(os.path.join(path_new, 'association.txt'),"a")
for i in range(len(rgbL_time)):
    file_associate.write("%d rgb/%d.png %d depth/%d.png \n" % (rgbL_time[i], rgbL_time[i], depth_time[dep_idx[0,i]], depth_time[dep_idx[0,i]]))
file_associate.close()
yamlfile = open(os.path.join(path_new, 'rgbd_calib.yaml'), "w")
feature_p3d = open(os.path.join(path_new, 'feature_p3d.csv'), "a")
writer0 = csv.writer(feature_p3d, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer0.writerow(['#Frame_Number', 'Feature_x_pos', 'Feature_y_pos', 'Xwc', 'Ywc', 'Zwc'])
fullcammodel = open(os.path.join(path_new, 'full_cam_model.csv'), "a")
writer1 = csv.writer(fullcammodel, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer1.writerow(['#Frame_Number', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'fx', 'k1', 'k2', 'k3', 'p1', 'p2', 'cx', 'cy', 'fy', 'Scale'])
RGBLHCalib = open(os.path.join(path_new, 'rgbL_hcalib_res.csv'), "a")
writer2 = csv.writer(RGBLHCalib, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer2.writerow(['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz', 'fx', 'k1', 'k2', 'k3', 'p1', 'p2', 'cx', 'cy', 'fy', 'Scale'])
fulldcammodel = open(os.path.join(path_new, 'full_dcam_model.csv'), "a")
writer3 = csv.writer(fulldcammodel, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer3.writerow(['#Frame_Number', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'fx', 'k1', 'k2', 'k3', 'p1', 'p2', 'cx', 'cy', 'fy'])
DepthHCalib = open(os.path.join(path_new, 'depth_hcalib_res.csv'), "a")
writer4 = csv.writer(DepthHCalib, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer4.writerow(['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz', 'fx', 'k1', 'k2', 'k3', 'p1', 'p2', 'cx', 'cy', 'fy'])
TrajVelocity = open(os.path.join(path_new, 'traj_velo.csv'), "a")
writer5 = csv.writer(TrajVelocity, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer5.writerow(['#Frame_Number', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'vx', 'vy', 'vz'])
IMUStates = open(os.path.join(path_new, 'imu_states.csv'), "a")
writer6 = csv.writer(IMUStates, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer6.writerow(['#Frame_Number', 'bgx', 'bgy', 'bgz', 'bax', 'bay', 'baz', 'tic'])
IMUHCalib = open(os.path.join(path_new, 'IMU_hcalib_res.csv'), "a")
writer7 = csv.writer(IMUHCalib, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer7.writerow(['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'bgx', 'bgy', 'bgz', 'bax', 'bay', 'baz', 'tic'])

rgbL_images = load_images_from_folder(rgbL_path)
depth_images = load_images_from_folder(depth_path)

print('Left RGB Total frames:', len(rgbL_images))
print('Depth Total frames:', len(depth_images))

# depth intrinsics matrix
image_w = 640
image_h = 480
K_depth = np.identity(3)
K_depth[0, 0] = 4.59357e+02
K_depth[1, 1] = 4.59764e+02
K_depth[0, 2] = 3.32695e+02
K_depth[1, 2] = 2.58998e+02

# rgb intrinsics matrix
image_w = 640
image_h = 480
K_rgb = np.identity(3)
K_rgb[0, 0] = 4.59357e+02
K_rgb[1, 1] = 4.59764e+02
K_rgb[0, 2] = 3.32695e+02
K_rgb[1, 2] = 2.58998e+02
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(image_w,image_h,K_rgb[0, 0],K_rgb[1, 1],K_rgb[0, 2],K_rgb[1, 2])

# rgb (Left)-d extrinsic matrix (T) - Rigid body motion for points from depth to rgb frame
R_c_d = Rotation.from_euler('zyx', [0.0, 0.0, 0.0], degrees=True).as_matrix()
t_c_d =  [-0.1, 0.0, 0.0]
T_c_d = np.eye(4)
T_c_d[:3,:3] = R_c_d
T_c_d[:3,3] = t_c_d
dist_coef_color = None

# Checkerboard Specs.
board_columns = 6
board_rows = 4

# RANSAC Visual-Odometry Pipeline based-on ShiTomasi features and KLT tracking
Ric = np.array([[0.00193013, -0.999997, 0.00115338],[-0.999996, -0.0019327, -0.00223606],[0.00223829, -0.00114906, -0.999997]])
Posess = mono_VO_ORB(path4, K_rgb, image_w, image_h, Ric)
print("VO estimated posses for %d frames" % (len(Posess)))
writer5.writerow([0, Posess[0,3], Posess[0,4], Posess[0,5], Posess[0,6], Posess[0,0], Posess[0,1], Posess[0,2], 0.0, 0.0, 0.0]) # gt_vel[gt_idx[0,0]][0], gt_vel[gt_idx[0,0]][1], gt_vel[gt_idx[0,0]][2]
writer6.writerow([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(abs(float(rgbL_time[0])-float(imu_time[imu_idx[0,0]]))*1e-9)])
dt_cam = (rgbL_time[1] - rgbL_time[0]) * 1e-9
for i in range(1, len(Posess)):
    writer5.writerow([i, Posess[i,3], Posess[i,4], Posess[i,5], Posess[i,6], Posess[i,0], Posess[i,1], Posess[i,2], 0.0, 0.0, 0.0])  # gt_vel[gt_idx[0,i]][0], gt_vel[gt_idx[0,i]][1], gt_vel[gt_idx[0,i]][2]
    writer6.writerow([i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(abs(float(rgbL_time[i])-float(imu_time[imu_idx[0,i]]))*1e-9)])
TrajVelocity.close()
IMUStates.close()

Traj_list = ['qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'vx', 'vy', 'vz']
traj_pose_vel = pd.read_csv(data_dir+'/hcalib_opt/traj_velo.csv', delimiter=',', usecols=Traj_list).values

States_list = ['bgx', 'bgy', 'bgz', 'bax', 'bay', 'baz', 'tic']
imu_states = pd.read_csv(data_dir+'/hcalib_opt/imu_states.csv', delimiter=',', usecols=States_list).values

# Start Back-end Global Optimization
print("\n---------Starting Level 1 Optimization!---------\n")
T_w_i = FE_PGO(traj_pose_vel, gps_meas[gps_idx][0], gyro_meas[imu_idx][0])
T_w_i = np.roll(T_w_i, 4, axis=1)
traj_pose_vel[:, :7] = T_w_i


vis = o3d.visualization.Visualizer()
vis.create_window(
    window_name='HCALIB Point Cloud',
    width=960,
    height=540,
    left=480,
    top=270)
vis.get_render_option().background_color = [0.05, 0.05, 0.05]
vis.get_render_option().point_size = 1
vis.get_render_option().show_coordinate_frame = True

# Extracting Features from KeyFrames
pix_den = 0.1643  # Pixel Density (m/pixel)
print('Collecting_Corners')
for i in tqdm(range(0, len(rgbL_images))):
    img = cv.imread(rgbL_images[i])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dimg = cv.imread(depth_images[dep_idx[0, i]])  # registered depth frame sync. with the rgb frame
    dimg1 = depth_color(cv.imread(depth_images[dep_idx[0, i]], cv.IMREAD_ANYDEPTH))
    hcat1 = cv.hconcat([img, dimg1])
    # Collect Checkerboard corners (for Calibration) or select KF for Odometry
    found, dst = processImage(gray,(board_columns, board_rows))
    if not found: continue
    cv.drawChessboardCorners(img, (board_columns, board_rows), dst, found)
    location = dst
    frame = i * np.ones((len(location),))
    features = np.transpose(np.array([location[:, 1], location[:, 0]]))
    cloud2D = depth_to_array(dimg)       # Normalized 2D depth array [0,1] millimetric
    # p3d_WC = getInitP3DinCamFrame(features, K_depth, dist_coef_color, T_c_d, depth.astype(np.float32), depthDilation=False)
    pcl = depth_to_local_point_cloud(dimg, features, board_columns*board_rows)
    np.savez_compressed(data_dir+'/hcalib_opt/cloud/%d.npz' % i, cloud=cloud2D)
    writer0.writerows(np.transpose([frame, features[:,0], features[:,1], pcl[:,2], pcl[:,0], pcl[:,1]]))   # Feature, P3D on Depth, RGB frames in world coords.
    cv.drawChessboardCorners(dimg1, (board_columns, board_rows), dst, found)
    hcat2 = cv.hconcat([img, dimg1])
    cv.imshow('Collecting Checkerboard Corners with Corresponding Depth Maps', cv.vconcat([hcat1, hcat2]))
    # plot_p3d(vis, pcl)
    cv.waitKey(1)
cv.destroyAllWindows()
vis.destroy_window()
feature_p3d.close()

fp3d_list = ['#Frame_Number', 'Feature_x_pos', 'Feature_y_pos', 'Xwc', 'Ywc', 'Zwc']
fp3d_path = pd.read_csv(data_dir+'/hcalib_opt/feature_p3d.csv', delimiter=',', usecols=fp3d_list).values
points3d = np.array(fp3d_path[:,3:])
features = np.array(fp3d_path[:,:3])
frame = np.unique(fp3d_path[:,0].astype(int))
print("Collected corners from %d frames" % (len(frame)))

init_dist = 0.0  # Initial Value for distortion coefficients

# Create the full camera and dcamera model files
# cams_list = ['qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'fx', 'k1', 'k2', 'k3', 'p1', 'p2', 'cx', 'cy', 'fy', 'Scale']  # Scale for the RGB cam only

T_ic = np.array([[0.00193013, -0.999997, 0.00115338, -0.00817048],
      [-0.999996, -0.0019327, -0.00223606, 0.015075],
      [0.00223829, -0.00114906, -0.999997, -0.0110795],
      [0, 0, 0, 1]])
q_ic = Rotation.from_matrix(T_ic[:3,:3]).as_quat()

T_id = np.array([[0.00193013, -0.999997, 0.00115338, -0.007977467],
      [-0.999996, -0.0019327, -0.00223606, -0.0849246],
      [0.00223829, -0.00114906, -0.999997, -0.010855671],
      [0, 0, 0, 1]])

T_dc = np.linalg.inv(T_id) @ T_ic
q_dc = Rotation.from_matrix(T_dc[:3,:3]).as_quat()

full_cam_model = []
for i in range(0, len(frame)):
    gt_curr_feature = [-0.00817048, 0.015075, -0.0110795, q_ic[3], q_ic[0], q_ic[1], q_ic[2]]    # Tic
    output_array = np.array([frame[i], gt_curr_feature[3],gt_curr_feature[4],gt_curr_feature[5],gt_curr_feature[6], gt_curr_feature[0],gt_curr_feature[1],gt_curr_feature[2], K_rgb[0, 0], init_dist, init_dist, init_dist, init_dist, init_dist, K_rgb[0, 2], K_rgb[1, 2], K_rgb[1, 1], pix_den])
    full_cam_model.append([gt_curr_feature[3],gt_curr_feature[4],gt_curr_feature[5],gt_curr_feature[6], gt_curr_feature[0],gt_curr_feature[1],gt_curr_feature[2], K_rgb[0, 0], init_dist, init_dist, init_dist, init_dist, init_dist, K_rgb[0, 2], K_rgb[1, 2], K_rgb[1, 1], pix_den])
    writer1.writerow(output_array)
fullcammodel.close()
full_cam_model = np.array(full_cam_model, dtype=float)

full_dcam_model = []
for i in range(0, len(frame)):
    gt_curr_feature = [T_dc[0,3], T_dc[1,3], T_dc[2,3], q_dc[3], q_dc[0], q_dc[1], q_dc[2]]    # Tdc
    normal_quat = quat_norm([gt_curr_feature[3],gt_curr_feature[4],gt_curr_feature[5],gt_curr_feature[6]])
    output_array = np.array([frame[i], gt_curr_feature[3],gt_curr_feature[4],gt_curr_feature[5],gt_curr_feature[6], gt_curr_feature[0],gt_curr_feature[1],gt_curr_feature[2], K_depth[0, 0], init_dist, init_dist, init_dist, init_dist, init_dist, K_depth[0, 2], K_depth[1, 2], K_depth[1, 1]])
    full_dcam_model.append([normal_quat[0],normal_quat[1],normal_quat[2],normal_quat[3], gt_curr_feature[0],gt_curr_feature[1],gt_curr_feature[2], K_depth[0, 0], init_dist, init_dist, init_dist, init_dist, init_dist, K_depth[0, 2], K_depth[1, 2], K_depth[1, 1]])
    writer3.writerow(output_array)
fulldcammodel.close()
full_dcam_model = np.array(full_dcam_model, dtype=float)

# Start Back-end Local Optimization
print("\n---------Starting Level 2 Optimization!---------\n")
T_w_i, lin_vel, imustates, T_i_c, Full_cam_model, T_rgb_d, Full_dcam_model, Points3d = HCALIBBackend(full_cam_model, full_dcam_model, points3d, features, imu_meas, imu_states, traj_pose_vel, imu_time, rgbL_time)

# Saving the Optimization Results
# RGB (L) Camera
for i in range(len(Full_cam_model)):
    output_array = np.array(
        [T_i_c[i, 0], T_i_c[i, 1], T_i_c[i, 2], T_i_c[i, 3], T_i_c[i, 4], T_i_c[i, 5], T_i_c[i, 6],
         Full_cam_model[i, 0], Full_cam_model[i, 1], Full_cam_model[i, 2], Full_cam_model[i, 3],
         Full_cam_model[i, 4],
         Full_cam_model[i, 5], Full_cam_model[i, 6], Full_cam_model[i, 7], Full_cam_model[i, 8], Full_cam_model[i, 9]])
    writer2.writerow(output_array)
RGBLHCalib.close()
# Depth Camera
for i in range(len(Full_dcam_model)):
    output_array = np.array(
        [T_rgb_d[i, 0], T_rgb_d[i, 1], T_rgb_d[i, 2], T_rgb_d[i, 3], T_rgb_d[i, 4], T_rgb_d[i, 5], T_rgb_d[i, 6],
         Full_dcam_model[i, 0], Full_dcam_model[i, 1], Full_dcam_model[i, 2], Full_dcam_model[i, 3], Full_dcam_model[i, 4],
         Full_dcam_model[i, 5], Full_dcam_model[i, 6], Full_dcam_model[i, 7], Full_dcam_model[i, 8]])
    writer4.writerow(output_array)
DepthHCalib.close()
# IMU Twi, Velocity, States
for i in range(len(imustates)):
    output_array = np.array(
        [T_w_i[i, 0], T_w_i[i, 1], T_w_i[i, 2], T_w_i[i, 3], T_w_i[i, 4], T_w_i[i, 5], T_w_i[i, 6],
         lin_vel[i, 0], lin_vel[i, 1], lin_vel[i, 2], imustates[i, 0], imustates[i, 1], imustates[i, 2], imustates[i, 3], imustates[i, 4],
         imustates[i, 5], imustates[i, 6]])
    writer7.writerow(output_array)
IMUHCalib.close()

# Save ORB-SLAM3 compatible .yaml file
# RGB characteristics
fx = np.mean(Full_cam_model[:, 0])
fy = np.mean(Full_cam_model[:, 8])
cx = np.mean(Full_cam_model[:, 6])
cy = np.mean(Full_cam_model[:, 7])
k1 = np.mean(Full_cam_model[:, 1])
k2 = np.mean(Full_cam_model[:, 2])
p1 = np.mean(Full_cam_model[:, 4])
p2 = np.mean(Full_cam_model[:, 5])
k3 = np.mean(Full_cam_model[:, 3])
# baseline
tx = np.mean(T_rgb_d[:, 0])
ty = np.mean(T_rgb_d[:, 1])
tz = np.mean(T_rgb_d[:, 2])
bf = fx * np.sqrt(tx**2+ty**2+tz**2)
yamlfile.write('%YAML:1.0\n\n')
yamlfile.write('# ----------------------------------------\n')
yamlfile.write('# This YAML file is created using RGBD-CALIB\n')
yamlfile.write('# ----------------------------------------\n')
yamlfile.write('Camera.type: "PinHole"\n\n')
yamlfile.write('# Camera calibration and distortion parameters (OpenCV)\n')
yamlfile.write('Camera.fx: ' + str(fx) + '\n')
yamlfile.write('Camera.fy: ' + str(fy) + '\n')
yamlfile.write('Camera.cx: ' + str(cx) + '\n')
yamlfile.write('Camera.cy: ' + str(cy) + '\n\n')
yamlfile.write('Camera.k1: ' + str(k1) + '\n')
yamlfile.write('Camera.k2: ' + str(k2) + '\n')
yamlfile.write('Camera.p1: ' + str(p1) + '\n')
yamlfile.write('Camera.p2: ' + str(p2) + '\n')
yamlfile.write('Camera.k3: ' + str(k3) + '\n\n')
yamlfile.write('Camera.width: 1024\n')
yamlfile.write('Camera.height: 1024\n\n')
yamlfile.write('# Camera frames per second\n')
yamlfile.write('Camera.fps: 20.0\n\n')
yamlfile.write('# IR projector baseline times fx (aprox.)\n')
yamlfile.write('Camera.bf: ' + str(bf) + '\n\n')
yamlfile.write('# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)\n')
yamlfile.write('Camera.RGB: 0\n\n')
yamlfile.write('# Close/Far threshold. Baseline times.\n')
yamlfile.write('ThDepth: 40.0\n\n')
yamlfile.write('# Depth map values factor\n')
yamlfile.write('DepthMapFactor: 1000.0\n\n')
yamlfile.write('#--------------------------------------------------------------------------------------------\n')
yamlfile.write('# ORB Parameters\n')
yamlfile.write('#--------------------------------------------------------------------------------------------\n\n')
yamlfile.write('# ORB Extractor: Number of features per image\n')
yamlfile.write('ORBextractor.nFeatures: 1000\n\n')
yamlfile.write('# ORB Extractor: Scale factor between levels in the scale pyramid\n')
yamlfile.write('ORBextractor.scaleFactor: 1.2\n\n')
yamlfile.write('# ORB Extractor: Number of levels in the scale pyramid\n')
yamlfile.write('ORBextractor.nLevels: 8\n\n')
yamlfile.write('# ORB Extractor: Fast threshold\n')
yamlfile.write('# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.\n')
yamlfile.write('# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST\n')
yamlfile.write('# You can lower these values if your images have low contrast\n')
yamlfile.write('ORBextractor.iniThFAST: 20\n')
yamlfile.write('ORBextractor.minThFAST: 7\n\n')
yamlfile.write('#--------------------------------------------------------------------------------------------\n')
yamlfile.write('# Viewer Parameters\n')
yamlfile.write('#--------------------------------------------------------------------------------------------\n')
yamlfile.write('Viewer.KeyFrameSize: 0.05\n')
yamlfile.write('Viewer.KeyFrameLineWidth: 1\n')
yamlfile.write('Viewer.GraphLineWidth: 0.9\n')
yamlfile.write('Viewer.PointSize: 2\n')
yamlfile.write('Viewer.CameraSize: 0.08\n')
yamlfile.write('Viewer.CameraLineWidth: 3\n')
yamlfile.write('Viewer.ViewpointX: 0\n')
yamlfile.write('Viewer.ViewpointY: -0.7\n')
yamlfile.write('Viewer.ViewpointZ: -1.8\n')
yamlfile.write('Viewer.ViewpointF: 500\n')
yamlfile.close()

# Calculating the evaluation metrics after calibration
print("\n------------Level 1 Optimization------------")
stamps, trans_error, rot_error = RPE_calc(traj_pose_vel, Poses_path[gt_idx][0])  # args(estimated, ground-truth)  ---> Before Opt
print("\n------------Level 2 Optimization------------")
stamps1, trans_error1, rot_error1 = RPE_calc(np.roll(T_w_i, 4, axis=1), Poses_path[gt_idx][0])  # ---> After Opt

# Plotting the Overall Relative Pose Translational and Rotational Errors
fig = mpplot.figure()
ax = fig.add_subplot(2,2,1)
ax.plot(stamps,trans_error * 100.0,'-',color="red")
ax.set_ylabel('Relative Pose Translational Error [cm]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_title('Relative Pose Errors - Level 1 Opt.')
ax = fig.add_subplot(2,2,3)
ax.plot(stamps,rot_error * 180.0 / np.pi,'-',color="red")
ax.set_xlabel('Calibration Frame')
ax.set_ylabel('Relative Pose Orientation Error [deg]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax = fig.add_subplot(2,2,2)
ax.plot(stamps1,trans_error1 * 100.0,'-',color="blue")
ax.set_ylabel('Relative Pose Translational Error [cm]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_title('Relative Pose Errors - Level 2 Opt.')
ax = fig.add_subplot(2,2,4)
ax.plot(stamps1,rot_error1 * 180.0 / np.pi,'-',color="blue")
ax.set_xlabel('Calibration Frame')
ax.set_ylabel('Relative Pose Orientation Error [deg]')
ax.grid(color='k', linestyle='--', linewidth=0.25)

# Plotting the Rotations and Translations before and after Local-BA Optimization
xs = Poses_path[:,4]
ys = Poses_path[:,5]
zs = Poses_path[:,6]
scale_x = np.mean(abs(xs))/np.mean(abs(Posess[:,0]))
scale_y = np.mean(abs(ys))/np.mean(abs(Posess[:,1]))
scale_z = np.mean(abs(zs))/np.mean(abs(Posess[:,2]))
fig0 = mpplot.figure()
ax = fig0.add_subplot(1,1,1)
# ax.plot( scale_y*Posess[:,1],scale_x*Posess[:,0],'-',color="red")
ax.plot(traj_pose_vel[:,5],traj_pose_vel[:,4],'-',color="green")
ax.plot(T_w_i[:,1],T_w_i[:,0],'-',color="blue")
ax.plot(ys,xs,'-',color="violet")
#ax.plot(Ygps,Xgps,'.',color="brown", markersize=1)
#ax.plot(Y_CT_GPS,X_CT_GPS,'-',color="brown")
ax.set_xlabel('Y [m]')
ax.set_ylabel('X [m]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
#ax.legend(["Proposed (Lvl.1))", "Proposed (Lvl.1+2))", "DUI-VIO", "GPS", "CT-GPS"], loc ="lower left")
ax.legend(["Proposed (Lvl.1))", "Proposed (Lvl.1+2))", "DUI-VIO", "GPS", "CT-GPS"], loc ="lower left")

# Plotting the 3D points before and after calibration
pk = 20
fig1 = mpplot.figure()
# ax = fig1.add_subplot(1,2,1, projection='3d')
ax = fig1.add_subplot(1,2,1, projection='3d')
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = points3d[:int(len(points3d)/1),1]
    ys = points3d[:int(len(points3d)/1),0]
    zs = points3d[:int(len(points3d)/1),2]
    ax.scatter(xs, ys, zs, c=c, marker=m)
axis_font = {'fontname': 'Arial', 'size': '30'}
# Set the tick labels font
for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(30)
ax.set_xlabel('Y [m]', **axis_font)
ax.set_ylabel('X [m]', **axis_font)
ax.set_zlabel('Z [m]', **axis_font)
ax.set_title('RGB-D world 3D points using Depth maps')
ax = fig1.add_subplot(1,2,2, projection='3d')
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = Points3d[:int(len(Points3d)/pk),1]
    ys = Points3d[:int(len(Points3d)/pk),0]
    zs = Points3d[:int(len(Points3d)/pk),2]
    ax.scatter(xs, ys, zs, c=c, marker=m)
axis_font = {'fontname':'Arial', 'size':'30'}
# Set the tick labels font
for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels() ):
    label.set_fontname('Arial')
    label.set_fontsize(30)
ax.set_xlabel('Y [m]', **axis_font)
ax.set_ylabel('X [m]', **axis_font)
ax.set_zlabel('Z [m]', **axis_font)
#ax.set_title('Optimized world 3D points after calibration')

# Plotting the 3D Traj after calibration
xs = Poses_path[:,4]
ys = Poses_path[:,5]
zs = Poses_path[:,6]
fig1 = mpplot.figure()
ax = fig1.add_subplot(1,1,1, projection='3d')
ax.plot3D(scale_x*Posess[:,0], -scale_y*Posess[:,1], -scale_z*Posess[:,2],'-',color="red")
ax.plot3D(traj_pose_vel[:,4],-traj_pose_vel[:,5],-traj_pose_vel[:,6],'-',color="green")
#ax.plot3D(T_w_i[:,0],-T_w_i[:,1],-T_w_i[:,2],'-',color="blue")
ax.plot3D(xs,-ys,-zs,'-',color="blue")
#ax.plot3D(Xgps,-Ygps,-Zgps,'.',color="brown", markersize=1)
#ax.plot3D(X_CT_GPS,-Y_CT_GPS,-Z_CT_GPS,'-',color="brown")
ax.set_xlabel('X [m]]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
# ax.legend(["KLT-VO", "Lvl 1 Opt","Lvl 2 Opt", "GT", "GPS", "CT-GPS"], loc ="lower left")
#ax.set_title('3D Trajectory')
#ax.legend(["Proposed (Lvl.1))", "Proposed (Lvl.1+2))", "DUI-VIO", "GPS", "CT-GPS"], loc ="lower left")
ax.legend(["KLT-VO", "Proposed", "DUI-VIO"], loc ="upper right")
fig1.savefig(path_new+'traj_3d.pdf')

mod_gt_time = np.linspace(imu_time[0],imu_time[-1],len(gt_time))
fig0 = mpplot.figure()
ax = fig0.add_subplot(3,1,1)
ax.plot(imu_time,roll_IMU,'--',color="red", label="$\phi-RK4$")
ax.plot(mod_gt_time,roll,'-',color="red", label="$\phi-GT$")
ax.set_title('RK4 - Evaluation')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('Roll [rad]')
ax = fig0.add_subplot(3,1,2)
ax.plot(imu_time,pitch_IMU,'--',color="green", label=r"$\theta-RK4$")
ax.plot(mod_gt_time,pitch,'-',color="green", label=r"$\theta-GT$")
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('Pitch [rad]')
ax = fig0.add_subplot(3,1,3)
ax.plot(imu_time,yaw_IMU,'--',color="blue", label="$\psi-RK4$")
ax.plot(mod_gt_time,yaw,'-',color="blue", label="$\psi-GT$")
ax.set_ylabel('Yaw [rad]')
ax.set_xlabel('Time [nsec]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.legend(["RK4", "DUI-VIO"], loc ="upper right")

mod_gt_time = np.linspace(rgbL_time[0],rgbL_time[-1],len(gt_time))
fig0 = mpplot.figure()
ax = fig0.add_subplot(3,1,1)
ax.plot(rgbL_time,lin_vel[:, 0],'--',color="red", label="$\phi-RK4$")
ax.plot(mod_gt_time,gt_vel[:, 0],'-',color="red", label="$\phi-GT$")
ax.set_title('Estimated Velocity - Evaluation')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('Vx [m/s]')
ax = fig0.add_subplot(3,1,2)
ax.plot(rgbL_time,lin_vel[:, 1],'--',color="green", label=r"$\theta-RK4$")
ax.plot(mod_gt_time,gt_vel[:, 1],'-',color="green", label=r"$\theta-GT$")
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('Vy [m/s]')
ax = fig0.add_subplot(3,1,3)
ax.plot(rgbL_time,lin_vel[:, 2],'--',color="blue", label="$\psi-RK4$")
ax.plot(mod_gt_time,gt_vel[:, 2],'-',color="blue", label="$\psi-GT$")
ax.set_ylabel('Vz [m/s]')
ax.set_xlabel('Time [nsec]')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.legend(["Ours", "DUI-VIO"], loc ="upper right")

mpplot.show()

# Plotting the Point Cloud
# dmap1 = cv.rgbd.registerDepth(K_depth, K_rgb, dist_coef_color, TdrgbL, depth.astype(float), (image_w, image_h), depthDilation=False)
# rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.array(img.copy()).astype(np.uint8)), o3d.geometry.Image(np.asarray(depth).astype(np.float32)), convert_rgb_to_intensity=False)
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # flip the orientation, so it looks upright, not upside-down
# o3d.visualization.draw_geometries([pcd])  # visualize the point cloud
# P3D = np.asarray(pcd.points)
# # p3d = # get p3d corresponding to features
# print(p3d.shape)
# breakpoint()

# # Example for a more optimized BAL Problem using data on: http://grail.cs.washington.edu/projects/bal/
# file = data_dir+"/BALdata"
# bal_problem = PyCeres.BALProblem()
# print("ok")
# bal_problem.LoadFile(file)
# observations = bal_problem.observations()
# cameras = bal_problem.cameras()
# points = bal_problem.points()
# numpy_points = np.array(points)
# numpy_points = np.reshape(numpy_points, (-1, 3))
# numpy_cameras = np.array(cameras)
# numpy_cameras = np.reshape(numpy_cameras, (-1, 9))
# full_cam_model = np.zeros((len(numpy_cameras), 13))
# for i in range(0, len(numpy_cameras)):
#     full_cam_model[i, 4:10] = numpy_cameras[i, 3:9]
#     full_cam_model[i, 0:4] = Rotation.from_euler('xyz', numpy_cameras[i, 0:3], degrees=False).as_quat()  # euler angles to quaternion
#     full_cam_model[i, 0:4] = np.roll(full_cam_model[i, 0:4], 1)
# full_cam_model = np.reshape(full_cam_model, (-1, 13))
# full_cam_model, numpy_points = BALBackendRGB(full_cam_model, numpy_points, observations, bal_problem)
