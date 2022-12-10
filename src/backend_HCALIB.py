import sys
import numpy as np
import random
import time
import matplotlib.pyplot as mpplot
from datetime import date
from datetime import datetime
mpplot.rcParams['text.usetex'] = True
import matplotlib.pylab as pylab
import sophus
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.cm as cm
import cv2 as cv
import glob
from sophus import *
import os
from sensor import *
import math
pyceres_location = "/home/abanobsoliman/ceres-solver/ceres-bin/lib"
geometry_location = "/home/abanobsoliman/geometry/build"
pyceresfactors_location = "/home/abanobsoliman/pyceres_factors/build"
sys.path.insert(0, pyceres_location)
sys.path.insert(0, geometry_location)
sys.path.insert(0, pyceresfactors_location)
import PyCeres
from geometry import SE3
from geometry import SO3
import PyCeresFactors as factors
from jax import grad
import jax.numpy as jnp
import argparse
from scipy.spatial.transform import Rotation
import csv
import pandas as pd
import open3d as o3d
import math
from natsort import natsorted, ns
from progress.bar import Bar
from pose_evaluation_utils import *
import apriltag
import bspline
import teaserpp_python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image
import yaml
import vg
from tqdm import tqdm
import skimage.exposure
from pathlib import Path
import sklearn.metrics as metric
from skimage import data
from skimage.util import img_as_ubyte
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, diameter_closing
from skimage.morphology import black_tophat, skeletonize, convex_hull_image, diameter_opening
from skimage.morphology import disk, diamond
import metrics


def triangulation(kp1, kp2, T_1w, T_2w):
    """Triangulation to get 3D points
    Args:
        kp1 (Nx2): keypoint in view 1 (normalized)
        kp2 (Nx2): keypoints in view 2 (normalized)
        T_1w (4x4): pose of view 1 w.r.t world, i.e. T_1w (from w to 1)
        T_2w (4x4): pose of view 2 w.r.t world, i.e. T_2w (from w to 2)
    Returns:
        X (3xN): 3D coordinates of the keypoints w.r.t world coordinate
    """
    kp1_3D = np.ones((3, kp1.shape[0]))
    kp2_3D = np.ones((3, kp2.shape[0]))
    kp1_3D[0], kp1_3D[1] = kp1[:, 0].copy(), kp1[:, 1].copy()
    kp2_3D[0], kp2_3D[1] = kp2[:, 0].copy(), kp2[:, 1].copy()
    X = cv.triangulatePoints(T_1w[:3], T_2w[:3], kp1_3D[:2], kp2_3D[:2])
    euclid_points = cv.convertPointsFromHomogeneous(X.T)
    return euclid_points[:, 0]


def mono_VO_ORB(img_data_dir, K_rgb, image_w, image_h, Ric):
    Pose = []
    width = image_w
    height = image_h

    # get the image list in the directory
    img_list = glob.glob(os.path.join(img_data_dir, '*.png'))
    img_list = natsorted(img_list, key=lambda y: y.lower())
    num_frames = len(img_list)
    tracks = []
    track_len = 10
    detect_interval = 2  # 5 for medium, hard sequences, 10 for easy

    # Create some random colors
    color = np.random.randint(0, 255, (6000, 3))

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(20, 20),
                     maxLevel=7,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=10000,
                          qualityLevel=0.01,
                          minDistance=3,
                          blockSize=17)

    orb = cv.ORB_create(nfeatures=100)
    for i in tqdm(range(num_frames)):
        curr_img = cv.imread(img_list[i], 0)

        if i == 0:
            curr_R = np.eye(3)
            curr_t = np.array([0, 0, 0])
            if i % detect_interval == 0:
                mask = np.zeros_like(curr_img)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                kp, des = orb.detectAndCompute(curr_img,
                                               None)  # cv.goodFeaturesToTrack(curr_img, mask=mask, **feature_params)
                p = [pts.pt for pts in kp]
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(x, y)])
        else:
            prev_img = cv.imread(img_list[i - 1], 0)

            # # ====================== Use ORB Feature to do feature matching =====================#
            # # create ORB features
            # orb = cv.ORB_create(nfeatures=6000)
            # # find the keypoints and descriptors with ORB
            # kp1, des1 = orb.detectAndCompute(prev_img, None)
            # kp2, des2 = orb.detectAndCompute(curr_img, None)
            # # use brute-force matcher
            # # Match ORB descriptors
            # # Sort the matched keypoints in the order of matching distance so the best matches came to the front
            # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            # matches = bf.match(des1, des2)
            # matches = sorted(matches, key=lambda x: x.distance)
            # pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            # pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            # draw the current image with keypoints
            # framme = cv.imread(img_list[i])
            # mask = np.zeros_like(cv.imread(img_list[i - 1]))
            # for kk, (new, old) in enumerate(zip(pts2, pts1)):
            #     a, b = new.ravel()
            #     c, d = old.ravel()
            #     mask = cv.line(mask, (a, b), (c, d), color[kk].tolist(), 2)
            #     frame = cv.circle(framme, (a, b), 3, color[kk].tolist(), -1)
            # curr_img_kp = cv.add(frame, mask)
            # cv.imshow('Tracking Features', curr_img_kp)

            # ====================== Use ShiTomasi corner Feature to do feature matching =====================#
            # Robust Sparse Lucas-Kanade Optical Flow Tracking
            if len(tracks) > 0:
                kp1 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                kp2_OF, st, err = cv.calcOpticalFlowPyrLK(prev_img, curr_img, kp1, None, **lk_params)     # cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                kp1r, _st, _err = cv.calcOpticalFlowPyrLK(curr_img, prev_img, kp2_OF, None, **lk_params)
                d = abs(kp1 - kp1r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                vis = cv.imread(img_list[i])
                for tr, (x, y), good_flag in zip(tracks, kp2_OF.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                tracks = new_tracks
                cv.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
            if i % detect_interval == 0:
                mask = np.zeros_like(curr_img)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                kp, des = orb.detectAndCompute(curr_img,
                                               None)  # cv.goodFeaturesToTrack(curr_img, mask=mask, **feature_params)
                p = [pts.pt for pts in kp]
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(x, y)])
            cv.imshow('Tracking Features (Monocular KLT-VO)', cv.resize(vis, (0, 0), fx=0.77, fy=0.77))
            pts2 = np.float32([m[-1] for m in kp2_OF[good]])
            pts1 = np.float32([m[-1] for m in kp1[good]])

            # compute essential matrix
            pts1_norm = cv.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K_rgb, distCoeffs=None)
            pts2_norm = cv.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K_rgb, distCoeffs=None)
            E, mask = cv.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(0., 0.), method=cv.USAC_MAGSAC,
                                          prob=0.999, threshold=3.0)
            _, R, t, mask = cv.recoverPose(E, pts1_norm, pts2_norm)

            # get camera motion
            R = R.transpose()
            t = -np.matmul(R, t)

            if i == 1:
                curr_R = R
                curr_t = t
            else:
                curr_R = np.matmul(prev_R, R)
                curr_t = np.matmul(prev_R, t) + prev_t

        # save current pose Twi quaternion(quatnormalize(eul2quat([R_vi*quat2eul(gtQuaternions(i),'XYZ')']','XYZ')))
        Rod_wc = Rotation.from_matrix(curr_R).as_mrp().T
        Rwi = Rotation.from_mrp(np.transpose(Ric @ Rod_wc)).as_matrix()
        twi = Ric @ curr_t
        [tx, ty, tz] = [twi[0], twi[1], twi[2]]
        qwi = Rotation.from_matrix(Rwi).as_quat()
        Pose.append([float(tx), float(ty), float(tz), float(qwi[3]), float(qwi[0]), float(qwi[1]), float(qwi[2])])

        prev_R = curr_R
        prev_t = curr_t

        cv.waitKey(1)
    cv.destroyAllWindows()
    return np.array(Pose, dtype=np.float32)


def quat_norm(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if mag2 > tolerance:
        mag = math.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v).astype(float)


def processImageAprilGrid(gray, family):
    location = []
    options = apriltag.DetectorOptions(families=family)
    detector = apriltag.Detector(options)
    results = detector.detect(gray)
    print("[INFO] {} total AprilTags detected".format(len(results)))
    if len(results) != 0:
        for r in results:
            location = location + r.corners.reshape(-1, 2).tolist()
        found = True
    else:
        location.append([0])
        found = False
    return np.array(location, dtype=float), found


def show_grid(location, img):
    for i in range(len(location)):
        x, y = location[i, :]
        cv.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
    return img


def processImage(img, pattern_size):
    nothing = np.array([0])
    found, corners = cv.findChessboardCorners(img, pattern_size)
    if found:
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        location = corners.reshape(-1, 2)
    if not found:
        print('checkerboard not found')
        location = nothing
    return found, location


def assign(rgbR_time, depth_time):
    idx = np.zeros((1, len(rgbR_time)))
    for i in range(0, len(rgbR_time)):
        arr = np.abs(rgbR_time[i] - depth_time)
        mini = np.amin(arr)
        result = np.where(arr == mini)
        if len(result) > 0 and len(result[0]) > 0:
            idx[0, i] = result[0][0]
    return idx


def getInitP3DinCamFrame(features, intrinsics_depth, dist_coef_color, extrinsics, depth, depthDilation):
    assert dist_coef_color is None
    assert depthDilation == False
    u, v = features[:, 1], features[:, 0]  # Depth frame pixels at board corners
    z = depth[v.astype(int), u.astype(int)]
    u = u.reshape(1, -1)
    v = v.reshape(1, -1)
    x_z = (u - intrinsics_depth[0, 2]) / intrinsics_depth[0, 0]
    y_z = (v - intrinsics_depth[1, 2]) / intrinsics_depth[1, 1]
    z = z / np.sqrt(1. + x_z ** 2 + y_z ** 2)  # Initial Depth Correction Function
    pts = np.vstack((z, x_z * z, y_z * z))  # World X,Y,Z coordinates in depth frame
    pts = extrinsics[:3, :3] @ pts + extrinsics[:3, 3:]
    return np.transpose(pts)


def plot3D_points(ax, points3d, pk):
    for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = points3d[:int(len(points3d) / pk), 1]
        ys = points3d[:int(len(points3d) / pk), 0]
        zs = points3d[:int(len(points3d) / pk), 2]
        ax.scatter(xs, ys, zs, c=c, marker=m)
    ax.set_xlabel('Y [m]')
    ax.set_ylabel('X [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Scene Reconstruction')


def load_images_from_folder(folder):
    # images = []
    # files = natsorted(folder, key=lambda y: y.lower())
    # for filename in files:
    #     img = cv.imread(filename)  # Loads a BGR image
    #     images.append(img)
    # return images
    img_list = natsorted(folder, key=lambda y: y.lower())
    return img_list


def gnss2enu(gps_meas):
    lat, lon, height = gps_meas[:, 0], gps_meas[:, 1], gps_meas[:, 2]
    # Initialization
    a = 6378137.0
    b = 6356752.314245
    a2 = a * a
    b2 = b * b
    e2 = 1.0 - (b2 / a2)
    e = e2 / (1.0 - e2)
    phi = lat * np.pi / 180.
    lmd = lon * np.pi / 180.
    cPhi = np.cos(phi)
    cLmd = np.cos(lmd)
    sPhi = np.sin(phi)
    sLmd = np.sin(lmd)
    N = a / np.sqrt(1.0 - e2 * sPhi * sPhi)
    x = (N + height) * cPhi * cLmd
    y = (N + height) * cPhi * sLmd
    z = ((b2 / a2) * N + height) * sPhi
    return z - z[0], y - y[0], -(x - x[0])


def publish_pose_map(poses_slam, pcl_opt):
    poses_slam = np.array(poses_slam)
    pcl_opt = np.array(pcl_opt)
    ax = mpplot.axes(projection='3d')
    # Data for a three-dimensional line
    xline = poses_slam[:, 0]
    yline = poses_slam[:, 1]
    zline = poses_slam[:, 2]
    ax.plot3D(zline, -xline, -yline, 'blue')
    # Data for three-dimensional scattered points
    xdata = pcl_opt[:, 0]
    ydata = pcl_opt[:, 1]
    zdata = pcl_opt[:, 2]
    ax.scatter3D(zdata, -xdata, -ydata, c=-ydata, cmap='viridis', linewidth=0.5)


def depth_to_array(image):
    """
    Convert a BGR image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = image.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth


def depth_to_local_point_cloud(image, features, corners=49):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the 3D position (relative to the camera) of each pixel and its corresponding
    RGB color of an array.
    """
    far = 100.0  # max depth in meters.
    normalized_depth = depth_to_array(image)

    # (Intrinsic) K Matrix
    k = np.identity(3)
    k[0, 2] = image.shape[1] / 2.0
    k[1, 2] = image.shape[0] / 2.0
    k[0, 0] = k[1, 1] = image.shape[0] / \
                        (2.0 * math.tan(90.0 * math.pi / 360.0))

    # 2d pixel coordinates
    pixel_length = corners
    u_coord, v_coord = features[:, 1], features[:, 0]  # Depth frame pixels at board corners
    normalized_depth = normalized_depth[v_coord.astype(int), u_coord.astype(int)]
    normalized_depth = np.reshape(normalized_depth, pixel_length)

    # pd2 = [u,v,1]
    p2d = np.array([u_coord.T, v_coord.T, np.ones_like(u_coord.T)])
    # P = [X,Y,Z]
    p3d = np.dot(np.linalg.inv(k), p2d)
    p3d *= normalized_depth * far

    return np.transpose(p3d)


def runge_orient(w, q0):
    qdot = 0.5 * np.array([[0, - w[1 - 1], - w[2 - 1], - w[3 - 1]], [w[1 - 1], 0, w[3 - 1], - w[2 - 1]],
                           [w[2 - 1], - w[3 - 1], 0, w[1 - 1]], [w[3 - 1], w[2 - 1], - w[1 - 1], 0]])
    x = qdot @ q0
    return x


def imu_quat(ax_imu, ay_imu, az_imu, gx_imu, gy_imu, gz_imu, t_imu):
    q_imu = np.zeros((4, len(t_imu)))
    ax1 = np.mean(ax_imu[:])
    ay1 = np.mean(ay_imu[:])
    az1 = np.mean(az_imu[:])
    pitch1 = np.arctan2(- ax1, np.sqrt(ay1 ** 2 + az1 ** 2))
    roll1 = np.arctan2(ay1, az1)
    pitch1 = (180. / np.pi) * (np.multiply(pitch1, (pitch1 >= 0)) + np.multiply((pitch1 + 2 * np.pi), (pitch1 < 0)))
    roll1 = (180. / np.pi) * (np.multiply(roll1, (roll1 >= 0)) + np.multiply((roll1 + 2 * np.pi), (roll1 < 0)))
    yaw1 = 1.
    cy = np.cos(np.pi / 180. * 0.5 * yaw1)
    sy = np.sin(np.pi / 180. * 0.5 * yaw1)
    cr = np.cos(np.pi / 180. * 0.5 * roll1)
    sr = np.sin(np.pi / 180. * 0.5 * roll1)
    cp = np.cos(np.pi / 180. * 0.5 * pitch1)
    sp = np.sin(np.pi / 180. * 0.5 * pitch1)
    q_0 = cy * cr * cp + sy * sr * sp
    q_1 = cy * sr * cp - sy * cr * sp
    q_2 = cy * cr * sp + sy * sr * cp
    q_3 = sy * cr * cp - cy * sr * sp
    q_imu[:, 1 - 1] = np.array([[q_0], [q_1], [q_2], [q_3]]).T
    dt_imu = t_imu[1:] - t_imu[:-1]
    for i in np.arange(1, len(dt_imu) + 1).reshape(-1):
        b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        d = np.array([1, 1 / 2, 1 / 2, 1])
        k = np.zeros((4, 4))
        k[:, 1 - 1] = (dt_imu[i - 1]) * runge_orient(np.array([gx_imu[i - 1], gy_imu[i - 1], gz_imu[i - 1]]),
                                                     q_imu[:, i - 1])
        for j in np.arange(2, 4 + 1).reshape(-1):
            k[:, j - 1] = (dt_imu[i - 1]) * runge_orient(np.array([gx_imu[i - 1], gy_imu[i - 1], gz_imu[i - 1]]),
                                                         q_imu[:, i - 1] + (d[j - 1] * k[:, j - 1 - 1]))
        q_imu[:, i + 1 - 1] = q_imu[:, i - 1] + np.transpose((b @ np.transpose(k)))
        q_norm = np.sqrt(
            q_imu[1 - 1, i + 1 - 1] ** 2 + q_imu[2 - 1, i + 1 - 1] ** 2 + q_imu[3 - 1, i + 1 - 1] ** 2 + q_imu[
                4 - 1, i + 1 - 1] ** 2)
        q_imu[:, i + 1 - 1] = q_imu[:, i + 1 - 1] / q_norm

    return q_imu


def enhance_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    close = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel1)
    div = np.float32(gray) / close
    res = np.uint8(cv.normalize(div, div, 0, 255, cv.NORM_MINMAX))
    return res


def depth_color(img):
    # stretch to full dynamic range
    stretch = skimage.exposure.rescale_intensity(img, in_range='image', out_range=(0, 255)).astype(np.uint8)
    # convert to 3 channels
    stretch = cv.merge([stretch, stretch, stretch])
    # define colors
    color1 = (0, 0, 255)  # red    nearest
    color2 = (0, 165, 255)  # orange
    color3 = (0, 255, 255)  # yellow
    color4 = (255, 255, 0)  # cyan
    color5 = (255, 0, 0)  # blue
    color6 = (128, 64, 64)  # violet   furthest
    colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)
    # resize lut to 256 (or more) values
    lut = cv.resize(colorArr, (256, 1), interpolation=cv.INTER_LINEAR)
    # apply lut
    result = cv.LUT(stretch, lut)
    # # create gradient image
    # grad = np.linspace(0, 255, 512, dtype=np.uint8)
    # grad = np.tile(grad, (20, 1))
    # grad = cv.merge([grad, grad, grad])
    # # apply lut to gradient for viewing
    # grad_colored = cv.LUT(grad, lut)
    return result


def FE_PGO(traj_pose, gps_meas, gyro_meas):
    T_w_i = traj_pose[:, :7].copy()
    T_w_i = np.roll(T_w_i, 3, axis=1)

    # Define the problem
    problem = PyCeres.Problem()

    for i in range(len(traj_pose)):
        problem.AddParameterBlock(T_w_i[i], 7, factors.SE3Parameterization())
    problem.SetParameterBlockConstant(T_w_i[0])

    # 1/2- Adding the 6 DOF Pose Graph Optimization residuals (SCALE and Global BA)
    # Odometry covariances (linear, angular)
    gyro_meas *= 0.1  # deactivate for ibiscape
    odom_cov_vals = (0.1, 0.001)
    odom_cov = np.eye(6)
    odom_cov[:3, :3] *= odom_cov_vals[0]  # delta translation noise
    odom_cov[3:, 3:] *= odom_cov_vals[1]  # delta rotation noise
    # range variance
    range_cov = 0.1
    # range covariance
    r_noise = np.sqrt(range_cov)
    loss = PyCeres.HuberLoss(0.1)
    print('Adding_PoseGraphOpt_residuals')
    for i in tqdm(range(len(traj_pose) - 1)):
        # noise-less range measurement
        ti = gps_meas[i]
        tj = gps_meas[i + 1]
        rij = np.linalg.norm(tj - ti)
        cost_SCALE = PyCeres.ScaleFactor(rij, r_noise)
        # delta pose/odometry between successive nodes in graph (tangent space representation as a local perturbation)
        dx = np.array([(tj - ti)[0], (tj - ti)[1], (tj - ti)[2], (gyro_meas[i + 1] - gyro_meas[i])[0],
                       (gyro_meas[i + 1] - gyro_meas[i])[1], (gyro_meas[i + 1] - gyro_meas[i])[2]])
        dxhat = SE3.Exp(dx)
        cost_PGO = PyCeres.PoseGraphOpt(dxhat.array(), odom_cov)
        problem.AddResidualBlock(cost_PGO, loss, T_w_i[i], T_w_i[i + 1])
        problem.AddResidualBlock(cost_SCALE, loss, T_w_i[i], T_w_i[i + 1])

    # Set the optimization options
    options = PyCeres.SolverOptions()
    options.linear_solver_type = PyCeres.LinearSolverType.SPARSE_SCHUR
    options.trust_region_strategy_type = PyCeres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    options.dogleg_type = PyCeres.DoglegType.SUBSPACE_DOGLEG
    options.use_explicit_schur_complement = True
    options.minimizer_progress_to_stdout = True
    options.max_num_iterations = 1000
    options.num_threads = 16
    # Set the optimization summary format
    summary = PyCeres.Summary()
    # Here we iterate to minimize the cost function
    PyCeres.Solve(options, problem, summary)
    print(summary.FullReport() + "\n")

    return T_w_i


def compare_min(x, y):
    return y ^ ((x ^ y) & -(x < y))


def HCALIBBackend(full_cam_model, full_dcam_model, points3d, features, imu_meas, imu_states, traj_pose_vel, t_imu,
                  t_cam):
    print("------- RGB Camera Final Pose and P3D -------------\n")
    print("X:     " + str(np.mean(points3d[:, 0])))
    print("Y:     " + str(np.mean(points3d[:, 1])))
    print("Z:     " + str(np.mean(points3d[:, 2])))
    print("qw:     " + str(np.mean(full_cam_model[:, 0])))
    print("qx:     " + str(np.mean(full_cam_model[:, 1])))
    print("qy:     " + str(np.mean(full_cam_model[:, 2])))
    print("qz:     " + str(np.mean(full_cam_model[:, 3])))
    print("tx:     " + str(np.mean(full_cam_model[:, 4])))
    print("ty:     " + str(np.mean(full_cam_model[:, 5])))
    print("tz:     " + str(np.mean(full_cam_model[:, 6])))
    print("------- Initial RGB Intrinsics -------------\n")
    print("fx:      " + str(np.mean(full_cam_model[:, 7])))
    print("fy:      " + str(np.mean(full_cam_model[:, 15])))
    print("cx: " + str(np.mean(full_cam_model[:, 13])))
    print("cy: " + str(np.mean(full_cam_model[:, 14])))
    print("rad_k1: " + str(np.mean(full_cam_model[:, 8])))
    print("rad_k2: " + str(np.mean(full_cam_model[:, 9])))
    print("tan_p1: " + str(np.mean(full_cam_model[:, 11])))
    print("tan_p2: " + str(np.mean(full_cam_model[:, 12])))
    print("rad_k3: " + str(np.mean(full_cam_model[:, 10])))
    print("Map Scale: " + str(np.mean(full_cam_model[:, 16])))
    print("------- Initial Depth Intrinsics -------------\n")
    print("fx:      " + str(np.mean(full_dcam_model[:, 7])))
    print("fy:      " + str(np.mean(full_dcam_model[:, 15])))
    print("cx: " + str(np.mean(full_dcam_model[:, 13])))
    print("cy: " + str(np.mean(full_dcam_model[:, 14])))
    print("rad_k1: " + str(np.mean(full_dcam_model[:, 8])))
    print("rad_k2: " + str(np.mean(full_dcam_model[:, 9])))
    print("tan_p1: " + str(np.mean(full_dcam_model[:, 11])))
    print("tan_p2: " + str(np.mean(full_dcam_model[:, 12])))
    print("rad_k3: " + str(np.mean(full_dcam_model[:, 10])))
    print("------- Initial RGB-D Extrinsic Parameters -------------\n")
    print("qw:     " + str(np.mean(full_dcam_model[:, 0])))
    print("qx:     " + str(np.mean(full_dcam_model[:, 1])))
    print("qy:     " + str(np.mean(full_dcam_model[:, 2])))
    print("qz:     " + str(np.mean(full_dcam_model[:, 3])))
    print("tx:     " + str(np.mean(full_dcam_model[:, 4])))
    print("ty:     " + str(np.mean(full_dcam_model[:, 5])))
    print("tz:     " + str(np.mean(full_dcam_model[:, 6])))

    # Define the problem
    problem = PyCeres.Problem()

    fullcammodel = full_cam_model[:, 7:].copy()
    fulldcammodel = full_dcam_model[:, 7:].copy()
    imumeas = imu_meas.copy()
    imustates = imu_states.copy()
    lin_vel = traj_pose_vel[:, 7:].copy()
    T_w_i = traj_pose_vel[:, :7].copy()
    T_w_i = np.roll(T_w_i, 3, axis=1)
    T_i_c = full_cam_model[:, :7].copy()
    T_i_c = np.roll(T_i_c, 3, axis=1)
    T_rgb_d = full_dcam_model[:, :7].copy()
    T_rgb_d = np.roll(T_rgb_d, 3, axis=1)
    p3d = points3d.copy()

    # global CLOUD
    # CLOUD = []
    # CLOUD.append(np.load('../data/hcalib_opt/cloud/%d.npz' % features[0, 0].astype(int))['cloud'])
    # for i in range(1, len(features)):
    #     if features[i, 0].astype(int) != features[i - 1, 0].astype(int):
    #         CLOUD.append(np.load('../data/hcalib_opt/cloud/%d.npz' % features[i, 0].astype(int))['cloud'])

    # Local Parameterization of the problem on manifold
    frames = np.unique(features[:, 0].astype(int))
    for i in range(len(frames)):
        problem.AddParameterBlock(T_w_i[i], 7, factors.SE3Parameterization())
        problem.AddParameterBlock(lin_vel[i], 3)
        problem.AddParameterBlock(imustates[i], 7)
        problem.AddParameterBlock(T_i_c[i], 7, factors.SE3Parameterization())
        problem.AddParameterBlock(T_rgb_d[i], 7, factors.SE3Parameterization())
        problem.AddParameterBlock(fullcammodel[i], 10)
        problem.AddParameterBlock(fulldcammodel[i], 9)
        # # Set intrinsic and extrinsic calibration parameters constant for better pose estimation
        # problem.SetParameterBlockConstant(T_i_c[i])
        # problem.SetParameterBlockConstant(T_rgb_d[i])
        # problem.SetParameterBlockConstant(fullcammodel[i])
        # problem.SetParameterBlockConstant(fulldcammodel[i])
    for i in range(len(features)):
        problem.AddParameterBlock(p3d[i], 3)
    problem.SetParameterBlockConstant(T_w_i[0])

    # 1- Reprojection error of the checkerboard corners onto the image
    rgb_cov = (0.2 ** 2) * np.eye(2)
    k_cam = 0
    print('Adding_RGB_Corners_residuals')
    for i in tqdm(range(len(features))):
        if i > 0:
            if features[i, 0].astype(int) != features[i - 1, 0].astype(int):
                k_cam += 1
        cost_function = PyCeres.ReprojectionErrors(np.array([features[i, 1], features[i, 2]]), rgb_cov)
        loss = PyCeres.HuberLoss(0.1)  # CauchyLoss 0.03 HuberLoss 0.1 or None or Square(residual size / 2)
        problem.AddResidualBlock(cost_function, loss, T_w_i[k_cam], T_i_c[k_cam], fullcammodel[k_cam], p3d[i])

    # 2-  The error between the planes defined by the checkerboards and the ones defined by the undistorted PCL
    pcl_cov = (1 ** 2) * np.eye(2)
    k_cam = 0
    print('Adding_PCL_residuals')
    for i in tqdm(range(len(p3d))):
        if i > 0:
            if features[i, 0].astype(int) != features[i - 1, 0].astype(int):
                k_cam += 1
        cost_function = PyCeres.GlobalCloudOpt(np.array([features[i, 1], features[i, 2]]), pcl_cov)
        loss = PyCeres.CauchyLoss(0.001)  # CauchyLoss 0.03 HuberLoss 0.1 or None or Square(residual size / 2)
        problem.AddResidualBlock(cost_function, loss, T_w_i[k_cam], T_i_c[k_cam], T_rgb_d[k_cam], fullcammodel[k_cam],
                                 fulldcammodel[k_cam], p3d[i])

    # 3/4/5- Adding the IMU & Biases residuals (Intrnsics and Local Scale)
    gyroscope_noise_density = 0.01303187663223313
    accelerometer_noise_density = 0.11036309879919973
    gyroscope_random_walk = 0.0007030288979477422
    accelerometer_random_walk = 0.012631195814507378
    KF = np.unique(features[:, 0].astype(int))
    t_cam = t_cam[KF]
    dt = (t_imu[1] - t_imu[0]) * 1e-9
    dt_ij = (t_cam[1] - t_cam[0]) * 1e-9
    # Delta Bias covariances (gyro, accel)
    dbias_cov_vals = (gyroscope_noise_density / dt, accelerometer_noise_density / dt)
    dbias_cov = np.eye(6)
    dbias_cov[:3, :3] *= dbias_cov_vals[0]  # delta gyro noise
    dbias_cov[3:, 3:] *= dbias_cov_vals[1]  # delta accel noise
    IMU_bias = np.linalg.cholesky(dbias_cov)
    bias_i = np.array([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])  # input constant (gyro,accel) biases  --> calibrated
    # Bias covariances (gyro, accel)
    bias_cov_vals = (gyroscope_random_walk * dt_ij, accelerometer_random_walk * dt_ij)
    Gbias_cov = np.eye(3)
    Abias_cov = np.eye(3)
    Gbias_cov *= bias_cov_vals[0]  # delta gyro noise
    Abias_cov *= bias_cov_vals[1]  # delta accel noise
    Gbias_cov_sqrt = np.linalg.cholesky(Gbias_cov)
    Abias_cov_sqrt = np.linalg.cholesky(Abias_cov)
    k_cam = 0
    imu_preint_stack = []
    gravity = np.transpose(np.array([0.0, 0.0, 9.80665]))  # in World coords (x-forwards, y-right, z-downwards) RHR
    print('Adding_IMU_residuals')
    loss = PyCeres.HuberLoss(0.1)  # set None in odometry
    for i in tqdm(range(len(imu_meas))):
        if t_cam[k_cam] <= t_imu[i] <= t_cam[k_cam + 1]:
            imu_preint_stack.append(imumeas[i])
            if t_imu[i + 1] > t_cam[k_cam + 1]:
                # print(np.array(imu_preint_stack).shape)
                cost_IMU = PyCeres.IMUFactor(np.array(imu_preint_stack), IMU_bias, dt, gravity, bias_i.T)
                cost_BIAS = PyCeres.IMUBiasFactor(Gbias_cov_sqrt, Abias_cov_sqrt)
                cost_tic = PyCeres.IMUticFactor(t_cam[k_cam + 1] * 1e-9, t_imu[i] * 1e-9)
                problem.AddResidualBlock(cost_IMU, loss, T_w_i[KF[k_cam]], T_w_i[KF[k_cam + 1]], lin_vel[KF[k_cam]],
                                         lin_vel[KF[k_cam + 1]], imustates[KF[k_cam]])
                problem.AddResidualBlock(cost_BIAS, loss, imustates[KF[k_cam]], imustates[KF[k_cam + 1]])
                problem.AddResidualBlock(cost_tic, None, imustates[KF[k_cam]])
                imu_preint_stack = []
                k_cam += 1
        if k_cam == len(t_cam) - 1:
            break
    print("Dropping remaining IMU readings, Out of RGB Frames Range!")

    # Set the optimization options
    options = PyCeres.SolverOptions()
    options.linear_solver_type = PyCeres.LinearSolverType.SPARSE_SCHUR
    options.trust_region_strategy_type = PyCeres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    options.dogleg_type = PyCeres.DoglegType.SUBSPACE_DOGLEG
    options.minimizer_progress_to_stdout = True
    options.use_explicit_schur_complement = True
    options.max_num_iterations = 1000
    options.num_threads = 16

    # Set the optimization summary format
    summary = PyCeres.Summary()

    # Here we iterate to minimize the cost function
    PyCeres.Solve(options, problem, summary)

    print(summary.FullReport() + "\n")
    print("------- Final Pose and P3D - Optimized -------------")
    print("X:     " + str(np.mean(p3d[:, 0])))
    print("Y:     " + str(np.mean(p3d[:, 1])))
    print("Z:     " + str(np.mean(p3d[:, 2])))
    print("qw:     " + str(np.mean(T_w_i[:, 3])))
    print("qx:     " + str(np.mean(T_w_i[:, 4])))
    print("qy:     " + str(np.mean(T_w_i[:, 5])))
    print("qz:     " + str(np.mean(T_w_i[:, 6])))
    print("tx:     " + str(np.mean(T_w_i[:, 0])))
    print("ty:     " + str(np.mean(T_w_i[:, 1])))
    print("tz:     " + str(np.mean(T_w_i[:, 2])))
    print("\n")
    print("------- Final RGB Left Intrinsics -------------")
    print("fx:      " + str(np.mean(fullcammodel[:, 0])) + " +- " + str(np.std(fullcammodel[:, 0])))
    print("fy:      " + str(np.mean(fullcammodel[:, 8])) + " +- " + str(np.std(fullcammodel[:, 8])))
    print("cx: " + str(np.mean(fullcammodel[:, 6])) + " +- " + str(np.std(fullcammodel[:, 6])))
    print("cy: " + str(np.mean(fullcammodel[:, 7])) + " +- " + str(np.std(fullcammodel[:, 7])))
    print("rad_k1: " + str(np.mean(fullcammodel[:, 1])) + " +- " + str(np.std(fullcammodel[:, 1])))
    print("rad_k2: " + str(np.mean(fullcammodel[:, 2])) + " +- " + str(np.std(fullcammodel[:, 2])))
    print("tan_p1: " + str(np.mean(fullcammodel[:, 4])) + " +- " + str(np.std(fullcammodel[:, 4])))
    print("tan_p2: " + str(np.mean(fullcammodel[:, 5])) + " +- " + str(np.std(fullcammodel[:, 5])))
    print("rad_k3: " + str(np.mean(fullcammodel[:, 3])) + " +- " + str(np.std(fullcammodel[:, 3])))
    print("Map Scale: " + str(np.mean(fullcammodel[:, 9])) + " +- " + str(np.std(fullcammodel[:, 9])))
    print("\n")
    print("------- Final Depth Intrinsics -------------")
    print("fx:      " + str(np.mean(fulldcammodel[:, 0])) + " +- " + str(np.std(fulldcammodel[:, 0])))
    print("fy:      " + str(np.mean(fulldcammodel[:, 8])) + " +- " + str(np.std(fulldcammodel[:, 8])))
    print("cx: " + str(np.mean(fulldcammodel[:, 6])) + " +- " + str(np.std(fulldcammodel[:, 6])))
    print("cy: " + str(np.mean(fulldcammodel[:, 7])) + " +- " + str(np.std(fulldcammodel[:, 7])))
    print("rad_k1: " + str(np.mean(fulldcammodel[:, 1])) + " +- " + str(np.std(fulldcammodel[:, 1])))
    print("rad_k2: " + str(np.mean(fulldcammodel[:, 2])) + " +- " + str(np.std(fulldcammodel[:, 2])))
    print("tan_p1: " + str(np.mean(fulldcammodel[:, 4])) + " +- " + str(np.std(fulldcammodel[:, 4])))
    print("tan_p2: " + str(np.mean(fulldcammodel[:, 5])) + " +- " + str(np.std(fulldcammodel[:, 5])))
    print("rad_k3: " + str(np.mean(fulldcammodel[:, 3])) + " +- " + str(np.std(fulldcammodel[:, 3])))
    print("\n")
    print("------- Final RGB(L)-D Extrinsic Parameters -------------")
    print("R_dc:     " + str(Rotation.from_quat(
        [np.mean(T_rgb_d[:, 4]), np.mean(T_rgb_d[:, 5]), np.mean(T_rgb_d[:, 6]), np.mean(T_rgb_d[:, 3])]).as_matrix()))
    print("q_dc:     " + str(
        [np.mean(T_rgb_d[:, 4]), np.mean(T_rgb_d[:, 5]), np.mean(T_rgb_d[:, 6]), np.mean(T_rgb_d[:, 3])]))
    print("t_dc:     " + str(
        np.array([np.mean(T_rgb_d[:, 0]), np.mean(T_rgb_d[:, 1]), np.mean(T_rgb_d[:, 2])])) + " +- " + str(
        np.array([np.std(T_rgb_d[:, 0]), np.std(T_rgb_d[:, 1]), np.std(T_rgb_d[:, 2])])))
    print("\n")
    print("------- Final RGB(L)-IMU Extrinsic Parameters -------------")
    print("R_ic:     " + str(Rotation.from_quat(
        [np.mean(T_i_c[:, 4]), np.mean(T_i_c[:, 5]), np.mean(T_i_c[:, 6]), np.mean(T_i_c[:, 3])]).as_matrix()))
    print("q_ic:     " + str([np.mean(T_i_c[:, 4]), np.mean(T_i_c[:, 5]), np.mean(T_i_c[:, 6]), np.mean(T_i_c[:, 3])]))
    print(
        "t_ic:     " + str(np.array([np.mean(T_i_c[:, 0]), np.mean(T_i_c[:, 1]), np.mean(T_i_c[:, 2])])) + " +- " + str(
            np.array([np.std(T_i_c[:, 0]), np.std(T_i_c[:, 1]), np.std(T_i_c[:, 2])])))
    print("\n")
    print("------- Final IMU Intrinsic Parameters -------------")
    print("tic:     " + str(np.mean(imustates[:, 6])) + " +- " + str(np.std(imustates[:, 6])))
    print("bwx:     " + str(np.mean(imustates[:, 0])) + " +- " + str(np.std(imustates[:, 0])))
    print("bwy:     " + str(np.mean(imustates[:, 1])) + " +- " + str(np.std(imustates[:, 1])))
    print("bwz:     " + str(np.mean(imustates[:, 2])) + " +- " + str(np.std(imustates[:, 2])))
    print("bax:     " + str(np.mean(imustates[:, 3])) + " +- " + str(np.std(imustates[:, 3])))
    print("bay:     " + str(np.mean(imustates[:, 4])) + " +- " + str(np.std(imustates[:, 4])))
    print("baz:     " + str(np.mean(imustates[:, 5])) + " +- " + str(np.std(imustates[:, 5])))

    return T_w_i, lin_vel, imustates, T_i_c, fullcammodel, T_rgb_d, fulldcammodel, p3d


def BALBackendRGB(full_cam_model, numpy_points, observations, bal_problem):
    print("------- PyCeres Initial for last X,Y,Z readings-------------\n")
    print("Initial X:     " + str(numpy_points[-1][0]))
    print("Initial Y:     " + str(numpy_points[-1][1]))
    print("Initial Z:     " + str(numpy_points[-1][2]))
    print("Initial qw:     " + str(full_cam_model[-1][0]))
    print("Initial qx:     " + str(full_cam_model[-1][1]))
    print("Initial qy:     " + str(full_cam_model[-1][2]))
    print("Initial qz:     " + str(full_cam_model[-1][3]))
    print("Initial tx:     " + str(full_cam_model[-1][4]))
    print("Initial ty:     " + str(full_cam_model[-1][5]))
    print("Initial tz:     " + str(full_cam_model[-1][6]))
    print("Initial f:      " + str(np.mean(full_cam_model[:, 7])))
    print("Initial rad_k1: " + str(np.mean(full_cam_model[:, 8])))
    print("Initial rad_k2: " + str(np.mean(full_cam_model[:, 9])))
    print("Initial rad_k3: " + str(np.mean(full_cam_model[:, 10])))
    print("Initial tan_p1: " + str(np.mean(full_cam_model[:, 11])))
    print("Initial tan_p1: " + str(np.mean(full_cam_model[:, 12])))

    # Define the problem
    problem = PyCeres.Problem()

    # Adding the mono_RGB cameras residuals
    for i in range(0, bal_problem.num_observations()):
        feature_obsrvd = np.array([observations[2 * i + 0], observations[2 * i + 1]])
        cost_function = PyCeres.HCALIBmonoRGBCostFunctor(feature_obsrvd[0], feature_obsrvd[1])
        cam_index = bal_problem.camera_index(i)
        point_index = bal_problem.point_index(i)
        loss = PyCeres.HuberLoss(0.1)
        problem.AddResidualBlock(cost_function, loss, full_cam_model[cam_index], numpy_points[point_index])

    # Set the optimization options
    options = PyCeres.SolverOptions()
    options.linear_solver_type = PyCeres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.minimizer_progress_to_stdout = True
    options.max_num_iterations = 50
    options.num_threads = 8;

    # Set the optimization summary format
    summary = PyCeres.Summary()

    # Here we iterate to minimize the cost function
    PyCeres.Solve(options, problem, summary)

    print(summary.FullReport() + "\n")
    print("------- PyCeres Final for last X,Y,Z,R,t readings-------------\n")
    print("Final X:     " + str(numpy_points[-1][0]))
    print("Final Y:     " + str(numpy_points[-1][1]))
    print("Final Z:     " + str(numpy_points[-1][2]))
    print("Final qw:     " + str(full_cam_model[-1][0]))
    print("Final qx:     " + str(full_cam_model[-1][1]))
    print("Final qy:     " + str(full_cam_model[-1][2]))
    print("Final qz:     " + str(full_cam_model[-1][3]))
    print("Final tx:     " + str(full_cam_model[-1][4]))
    print("Final ty:     " + str(full_cam_model[-1][5]))
    print("Final tz:     " + str(full_cam_model[-1][6]))
    print("Final f:      " + str(np.mean(full_cam_model[:, 7])))
    print("Final rad_k1: " + str(np.mean(full_cam_model[:, 8])))
    print("Final rad_k2: " + str(np.mean(full_cam_model[:, 9])))
    print("Final rad_k3: " + str(np.mean(full_cam_model[:, 10])))
    print("Final tan_p1: " + str(np.mean(full_cam_model[:, 11])))
    print("Final tan_p1: " + str(np.mean(full_cam_model[:, 12])))

    return full_cam_model, numpy_points
