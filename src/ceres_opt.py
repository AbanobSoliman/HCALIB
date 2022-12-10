import sys
import numpy as np
import matplotlib.pyplot as mpplot
import matplotlib.image as mpimg
import cv2 as cv
import glob
import os
pyceres_location="/home/abanodsoliman/ceres-solver/ceres-bin/lib"
sys.path.insert(0, pyceres_location)
import PyCeres
from jax import grad
import jax.numpy as jnp
import argparse
from scipy.spatial.transform import Rotation

# Testing: Create your own function
# Cost Function f(x) definition
def residual_calc(rx, ry, rz, tx, ty, tz, f, k1, k2, x, y, z, f_obs, flag):
    angle_axis = [rx, ry, rz]
    pt = [x, y, z]
    theta2 = rx * rx + ry * ry + rz * rz
    if theta2 > sys.float_info.epsilon:
        theta = jnp.sqrt(theta2)
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        theta_inverse = 1.0 / theta
        w = angle_axis[0] * theta_inverse, angle_axis[1] * theta_inverse, angle_axis[2] * theta_inverse
        w_cross_pt = w[1] * pt[2] - w[2] * pt[1], w[2] * pt[0] - w[0] * pt[2], w[0] * pt[1] - w[1] * pt[0]
        tmp = (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (1.0 - costheta)
        cpx = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp
        cpy = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp
        cpz = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp
    else:
        w_cross_pt = angle_axis[1] * pt[2] - angle_axis[2] * pt[1], angle_axis[2] * pt[0] - angle_axis[0] * pt[2], angle_axis[0] * pt[1] - angle_axis[1] * pt[0]
        cpx = pt[0] + w_cross_pt[0]
        cpy = pt[1] + w_cross_pt[1]
        cpz = pt[2] + w_cross_pt[2]
    px = tx + cpx
    py = ty + cpy
    pz = tz + cpz
    if (flag):
        return (f * (1.0 + ((- px / pz) * (- px / pz) + (- py / pz) * (- py / pz)) * (
                    k1 + k2 * ((- px / pz) * (- px / pz) + (- py / pz) * (- py / pz)))) * (- py / pz)) - f_obs
    else:
        return (f * (1.0 + ((- px / pz) * (- px / pz) + (- py / pz) * (- py / pz)) * (k1 + k2 * ((- px / pz) * (- px / pz) + (- py / pz) * (- py / pz)))) * (- px / pz)) - f_obs

class PYmonoRGBCostFunction(PyCeres.CostFunction):
    def __init__(self, feature_obsrvd):
        # MUST BE CALLED. Initializes the Ceres::CostFunction class
        super().__init__()
        # MUST BE CALLED. Sets the size of the residuals and parameters
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([9,3])
        self.feature_obsrvd = feature_obsrvd

    # The CostFunction::Evaluate(...) virtual function implementation
    def Evaluate(self, parameters, residuals, jacobians):
        rx = parameters[0][0]
        ry = parameters[0][1]
        rz = parameters[0][2]
        tx = parameters[0][3]
        ty = parameters[0][4]
        tz = parameters[0][5]
        f = parameters[0][6]
        k1 = parameters[0][7]
        k2 = parameters[0][8]

        x = parameters[1][0]
        y = parameters[1][1]
        z = parameters[1][2]

        residuals[0] = residual_calc(rx, ry, rz, tx, ty, tz, f, k1, k2, x, y, z, self.feature_obsrvd[0], 0)
        residuals[1] = residual_calc(rx, ry, rz, tx, ty, tz, f, k1, k2, x, y, z, self.feature_obsrvd[1], 1)

        if (jacobians != None):  # check for Null
            # partial derivatives of the residuals wrt parameters
            jacobian = jacobians[0]
            jacobian[0], jacobian[2], jacobian[4], jacobian[6], jacobian[8], \
            jacobian[10], jacobian[12], jacobian[14], jacobian[16] \
                = grad(residual_calc, (0, 1, 2, 3, 4, 5, 6, 7, 8))(rx, ry, rz, tx, ty, tz, f, k1, k2, x, y, z, self.feature_obsrvd[0],0)
            jacobian[1], jacobian[3], jacobian[5], jacobian[7], jacobian[9], \
            jacobian[11], jacobian[13], jacobian[15], jacobian[17] \
                = grad(residual_calc, (0, 1, 2, 3, 4, 5, 6, 7, 8))(rx, ry, rz, tx, ty, tz, f, k1, k2, x, y, z, self.feature_obsrvd[1],1)
            jacobian = jacobians[1]
            jacobian[0], jacobian[2], jacobian[4] \
                = grad(residual_calc, (9, 10, 11))(rx, ry, rz, tx, ty, tz, f, k1, k2, x, y, z, self.feature_obsrvd[0],0)
            jacobian[1], jacobian[3], jacobian[5] \
                = grad(residual_calc, (9, 10, 11))(rx, ry, rz, tx, ty, tz, f, k1, k2, x, y, z, self.feature_obsrvd[1],1)
        return True