/*
    This code is developed by Abanob Soliman, PhD Candidate, IBISC Laboratory
    As part of PhD Thesis, 2022
*/

#ifndef OPTIMIZATION_HELPER_H
#define OPTIMIZATION_HELPER_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include "glog/logging.h"
#include "Eigen/Dense"
#include "Eigen/LU"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include <sophus/se3.hpp>
#include <sophus/se2.hpp>
#include <sophus/so2.hpp>
#include <sophus/so3.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <SO3.h>
#include <SE3.h>

namespace OptimHelp {

// Pinhole Camera Model with Radial-Tangential distortions (k1,k2,k3,p1,p2)
template <typename T>
inline void ProjectP3DtoP2D(const T _T_w_i[7], const T _T_i_c[7], const T camera[10], const T point[3], Eigen::Vector<T, 2>& result) {
    // Transform P3D from World to RGB Frame using RGB Extrinsics [R,t]
    const T &focalx = camera[0];
    const T &k1 = camera[1];
    const T &k2 = camera[2];
    const T &k3 = camera[3];
    const T &p1 = camera[4];
    const T &p2 = camera[5];
    const T &cx = camera[6];
    const T &cy = camera[7];
    const T &focaly = camera[8];
    const T &scale = camera[9];
    Eigen::Matrix<T, 3, 1> xc; // Result -> P3D in RGB coordinates
    Eigen::Matrix<T, 3, 1>const xw(point[0],point[1],point[2]); // P3D in WORLD coordinates
    SE3<T>const T_w_i(_T_w_i);
    SE3<T>const T_i_c(_T_i_c);
    xc = T_i_c.inverse() * (T_w_i.inverse() * xw);
    // Add the second, fourth and sixth order radial and tangential distortion coefficients.
    // Compute the center of distortion. The sign change comes from the camera model that Jean-Yves Bouguet's Bundler assumes, whereby the camera coordinate system has a negative z axis.
    T xn = T(xc[0]/xc[2]);
    T yn = T(xc[1]/xc[2]);
    T xp = T(xn + cx / focalx);
    T yp = T(yn + cy / focaly);
    T r2 = T(xp * xp + yp * yp);
    T xpdist = T(xp*(T(1)+k1*r2+k2*r2*r2+k3*r2*r2*r2+T(2)*p1*yp)+p2*(r2+T(2)*xp*xp));
    T ypdist = T(yp*(T(1)+k1*r2+k2*r2*r2+k3*r2*r2*r2+T(2)*p2*xp)+p1*(r2+T(2)*yp*yp));
    // Compute final projected point position.
    result[0] = T(xpdist * focalx);
    result[1] = T(ypdist * focaly);
}

// Pinhole Camera Model with Radial-Tangential distortions (k1,k2,k3,p1,p2)
template <typename T>
inline void ProjectP3DepthDist(const T _T_w_i[7], const T _T_i_c[7], const T _T_d_c[7], const T dcamera[9], const T point[3], Eigen::Matrix<T, 2, 1>& pred) {
    // Transform P3D from RGB to D Frame using RGB-D Extrinsics [R,t]   
    const T &dfocalx = dcamera[0];
    const T &dk1 = dcamera[1];
    const T &dk2 = dcamera[2];
    const T &dk3 = dcamera[3];
    const T &dp1 = dcamera[4];
    const T &dp2 = dcamera[5];
    const T &dcx = dcamera[6];
    const T &dcy = dcamera[7];
    const T &dfocaly = dcamera[8];
    Eigen::Matrix<T, 3, 1> xd; // Result -> P3D in Depth coordinates
    Eigen::Matrix<T, 3, 1>const xw(point[0],point[1],point[2]); // P3D in WORLD coordinates
    SE3<T>const T_w_i(_T_w_i);
    SE3<T>const T_i_c(_T_i_c);
    SE3<T>const T_d_c(_T_d_c);
    xd = T_d_c * (T_i_c.inverse() * (T_w_i.inverse() * xw));
    // Add the second, fourth and sixth order radial and tangential distortion coefficients.
    // Compute the center of distortion. The sign change comes from the camera model that Jean-Yves Bouguet's Bundler assumes, whereby the camera coordinate system has a negative z axis.
    T xn = T(xd[0]/xd[2]);
    T yn = T(xd[1]/xd[2]);
    T xp = T(xn + dcx / dfocalx);  //normalized
    T yp = T(yn + dcy / dfocaly);  //normalized
    T r2 = T(xp * xp + yp * yp);
    T xpdist = T(xp*(T(1)+dk1*r2+dk2*r2*r2+dk3*r2*r2*r2+T(2)*dp1*yp)+dp2*(r2+T(2)*xp*xp)); 
    T ypdist = T(yp*(T(1)+dk1*r2+dk2*r2*r2+dk3*r2*r2*r2+T(2)*dp2*xp)+dp1*(r2+T(2)*yp*yp)); 
    // Compute final projected point position.
    pred[0] = T(xpdist * dfocalx);
    pred[1] = T(ypdist * dfocaly); 
}

// Pinhole Camera Model with Radial-Tangential distortions (k1,k2,k3,p1,p2)
template <typename T>
inline void ProjectP2DCamDist(const T _T_w_i[7], const T _T_i_c[7], const T _T_d_c[7], const T camera[10], const T dcamera[9], const Eigen::Matrix<T, 2, 1>& corner, Eigen::Matrix<T, 2, 1>& pred) {
    // Transform P3D from RGB to D Frame using RGB-D Extrinsics [R,t]   
    const T &focalx = camera[0];
    const T &k1 = camera[1];
    const T &k2 = camera[2];
    const T &k3 = camera[3];
    const T &p1 = camera[4];
    const T &p2 = camera[5];
    const T &cx = camera[6];
    const T &cy = camera[7];
    const T &focaly = camera[8];
    const T &scale = camera[9];
    const T &dfocalx = dcamera[0];
    const T &dk1 = dcamera[1];
    const T &dk2 = dcamera[2];
    const T &dk3 = dcamera[3];
    const T &dp1 = dcamera[4];
    const T &dp2 = dcamera[5];
    const T &dcx = dcamera[6];
    const T &dcy = dcamera[7];
    const T &dfocaly = dcamera[8];
    Eigen::Matrix<T, 3, 1> xw, xd; // Result -> P3D in world coordinates
    SE3<T>const T_w_i(_T_w_i);
    SE3<T>const T_i_c(_T_i_c);
    SE3<T>const T_d_c(_T_d_c);
    SE3<T>const T_w_c(T_w_i * T_i_c);
    Eigen::Matrix<T,3,3>const Rwc(T_w_c.q_.R());      // Rwc here
    Eigen::Map<Matrix<T,3,1>>const twc(T_w_c.t_);     // twc here
    Eigen::Matrix<T,3,3>const Kc{
    		{focalx, T(0.0), cx},
    		{T(0.0), focaly, cy},
    		{T(0.0), T(0.0), T(1.0)},
    };      // Cam intrinsics here
    Eigen::Matrix<T,3,1>const p2d(corner[0], corner[1], T(1.0));      // uv1 here
    xw = Rwc * (scale * Kc.inverse() * p2d) + twc;
    // Add the second, fourth and sixth order radial and tangential distortion coefficients.
    // Compute the center of distortion. The sign change comes from the camera model that Jean-Yves Bouguet's Bundler assumes, whereby the camera coordinate system has a negative z axis.
    xd = T_d_c * (T_i_c.inverse() * (T_w_i.inverse() * xw));
    T xn = T(xd[0]/xd[2]);
    T yn = T(xd[1]/xd[2]);
    T xp = T(xn + dcx / dfocalx);  //normalized
    T yp = T(yn + dcy / dfocaly);  //normalized
    T r2 = T(xp * xp + yp * yp);
    T xpdist = T(xp*(T(1)+dk1*r2+dk2*r2*r2+dk3*r2*r2*r2+T(2)*dp1*yp)+dp2*(r2+T(2)*xp*xp)); 
    T ypdist = T(yp*(T(1)+dk1*r2+dk2*r2*r2+dk3*r2*r2*r2+T(2)*dp2*xp)+dp1*(r2+T(2)*yp*yp)); 
    // Compute final projected point position.
    pred[0] = T(xpdist * dfocalx);
    pred[1] = T(ypdist * dfocaly); 
}

// Calculate IMU omega right jacobian (Jr)
template <typename T>
inline void rightJacobian(Eigen::Matrix<T, 3, 1>& w, Eigen::Matrix<T, 3, 3>& Jr) {
    T    th = w.norm();
    Eigen::Matrix<T, 3, 3> W = SO3<T>::hat(w);
    if (th > (T)1e-4)
    {
      T a = ((T)1. - cos(th))/(th*th);
      T b = (th - sin(th))/(th*th*th);
      Jr = Eigen::Matrix<T, 3, 3>::Identity() - a*W + b*(W*W);
    }
    else
    {
      Jr = Eigen::Matrix<T, 3, 3>::Identity();
    }
}

}  // namespace OptimHelp

#endif  // OPTIMIZATION_HELPER_H
