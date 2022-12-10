"""
This script computes the relative pose error and absolute trajectory errors from the ground truth trajectory
and the estimated trajectory.
** Inspired and reconfigured for IBISCape sequences from TUM rgbd_benchmark_tools **
"""

import sys
sys.path.append('.')
from backend_HCALIB import *

_EPS = np.finfo(float).eps * 4.0


def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[0]
    q = np.array(l[1], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
            (1.0, 0.0, 0.0, t[0]),
            (0.0, 1.0, 0.0, t[1]),
            (0.0, 0.0, 1.0, t[2]),
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], t[0]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], t[1]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], t[2]),
        (0.0, 0.0, 0.0, 1.0)
    ), dtype=np.float64)


def read_trajectory(filename):
    """
    Read a trajectory from a np array.

    Input:
    array -- qw, qx, qy, qz, tx, ty ,tz

    Output:
    dictionary of stamped 3D poses
    """
    list = filename.tolist()
    list_ok = []
    for i, l in enumerate(list):
        if l[:4] == [0, 0, 0, 0]:
            continue
        isnan = False
        for v in l:
            if np.isnan(v):
                isnan = True
                break
        if isnan:
            sys.stderr.write("Warning: line %d of file '%s' has NaNs, skipping line\n" % (i, filename))
            continue
        d = l[1:4]
        d.append(l[0])
        list_ok.append([i,l[4:7],d])
    traj = dict([(l[0], l[1:8]) for l in list_ok])
    return traj


def find_closest_index(L, t):
    """
    Find the index of the closest value in a list.

    Input:
    L -- the list
    t -- value to be found

    Output:
    index of the closest element
    """
    beginning = 0
    difference = abs(L[0] - t)
    best = 0
    end = len(L)
    while beginning < end:
        middle = int((end + beginning) / 2)
        if abs(L[middle] - t) < difference:
            difference = abs(L[middle] - t)
            best = middle
        if t == L[middle]:
            return middle
        elif L[middle] > t:
            end = middle
        else:
            beginning = middle + 1
    return best


def ominus(a, b):
    """
    Compute the relative 3D transformation between a and b.

    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)

    Output:
    Relative 3D transformation from a to b.
    """
    return np.dot(np.linalg.inv(transform44(a)), transform44(b))


def scale(a, scalar):
    """
    Scale the translational components of a 4x4 homogeneous matrix by a scale factor.
    """
    return np.array(
        [[a[0, 0], a[0, 1], a[0, 2], a[0, 3] * scalar],
         [a[1, 0], a[1, 1], a[1, 2], a[1, 3] * scalar],
         [a[2, 0], a[2, 1], a[2, 2], a[2, 3] * scalar],
         [a[3, 0], a[3, 1], a[3, 2], a[3, 3]]]
    )


def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return np.linalg.norm(transform[0:3, 3])


def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return np.arccos(min(1, max(-1, (np.trace(transform[0:3, 0:3]) - 1) / 2)))


def distances_along_trajectory(traj):
    """
    Compute the translational distances along a trajectory.
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i + 1]], traj[keys[i]]) for i in range(len(keys) - 1)]
    distances = [0]
    sum = 0
    for t in motion:
        sum += compute_distance(t)
        distances.append(sum)
    return distances


def rotations_along_trajectory(traj, scale):
    """
    Compute the angular rotations along a trajectory.
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i + 1]], traj[keys[i]]) for i in range(len(keys) - 1)]
    distances = [0]
    sum = 0
    for t in motion:
        sum += compute_angle(t) * scale
        distances.append(sum)
    return distances


def evaluate_trajectory(traj_gt, traj_est, param_max_pairs=10000, param_fixed_delta=False, param_delta=1.00,
                        param_delta_unit="f", param_offset=0.00, param_scale=1.00):
    """
    Compute the relative pose error between two trajectories.

    Input:
    traj_gt -- the first trajectory (ground truth)
    traj_est -- the second trajectory (estimated trajectory)
    param_max_pairs -- number of relative poses to be evaluated
    param_fixed_delta -- false: evaluate over all possible pairs
                         true: only evaluate over pairs with a given distance (delta)
    param_delta -- distance between the evaluated pairs
    param_delta_unit -- unit for comparison:
                        "s": seconds
                        "m": meters
                        "rad": radians
                        "deg": degrees
                        "f": frames
    param_offset -- time offset between two trajectories (to model the delay)
    param_scale -- scale to be applied to the second trajectory

    Output:
    list of compared poses and the resulting translation and rotation error
    """
    stamps_gt = list(traj_gt.keys())
    stamps_est = list(traj_est.keys())
    stamps_gt.sort()
    stamps_est.sort()

    stamps_est_return = []
    for t_est in stamps_est:
        t_gt = stamps_gt[find_closest_index(stamps_gt, t_est + param_offset)]
        t_est_return = stamps_est[find_closest_index(stamps_est, t_gt - param_offset)]
        t_gt_return = stamps_gt[find_closest_index(stamps_gt, t_est_return + param_offset)]
        if not t_est_return in stamps_est_return:
            stamps_est_return.append(t_est_return)
    if (len(stamps_est_return) < 2):
        raise Exception(
            "Number of overlap in the timestamps is too small. Did you run the evaluation on the right files?")

    if param_delta_unit == "s":
        index_est = list(traj_est.keys())
        index_est.sort()
    elif param_delta_unit == "m":
        index_est = distances_along_trajectory(traj_est)
    elif param_delta_unit == "rad":
        index_est = rotations_along_trajectory(traj_est, 1)
    elif param_delta_unit == "deg":
        index_est = rotations_along_trajectory(traj_est, 180 / np.pi)
    elif param_delta_unit == "f":
        index_est = range(len(traj_est))
    else:
        raise Exception("Unknown unit for delta: '%s'" % param_delta_unit)

    if not param_fixed_delta:
        if (param_max_pairs == 0 or len(traj_est) < np.sqrt(param_max_pairs)):
            pairs = [(i, j) for i in range(len(traj_est)) for j in range(len(traj_est))]
        else:
            pairs = [(random.randint(0, len(traj_est) - 1), random.randint(0, len(traj_est) - 1)) for i in
                     range(param_max_pairs)]
    else:
        pairs = []
        for i in range(len(traj_est)):
            j = find_closest_index(index_est, index_est[i] + param_delta)
            if j != len(traj_est) - 1:
                pairs.append((i, j))
        if (param_max_pairs != 0 and len(pairs) > param_max_pairs):
            pairs = random.sample(pairs, param_max_pairs)

    gt_interval = np.median([s - t for s, t in zip(stamps_gt[1:], stamps_gt[:-1])])
    gt_max_time_difference = 2 * gt_interval

    result = []
    for i, j in pairs:
        stamp_est_0 = stamps_est[i]
        stamp_est_1 = stamps_est[j]

        stamp_gt_0 = stamps_gt[find_closest_index(stamps_gt, stamp_est_0 + param_offset)]
        stamp_gt_1 = stamps_gt[find_closest_index(stamps_gt, stamp_est_1 + param_offset)]

        if (abs(stamp_gt_0 - (stamp_est_0 + param_offset)) > gt_max_time_difference or
                abs(stamp_gt_1 - (stamp_est_1 + param_offset)) > gt_max_time_difference):
            continue

        error44 = ominus(scale(
            ominus(traj_est[stamp_est_1], traj_est[stamp_est_0]), param_scale),
            ominus(traj_gt[stamp_gt_1], traj_gt[stamp_gt_0]))

        trans = compute_distance(error44)
        rot = compute_angle(error44)

        result.append([stamp_est_0, stamp_est_1, stamp_gt_0, stamp_gt_1, trans, rot])

    if len(result) < 2:
        raise Exception("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory!")

    return result


def percentile(seq, q):
    """
    Return the q-percentile of a list
    """
    seq_sorted = list(seq)
    seq_sorted.sort()
    return seq_sorted[int((len(seq_sorted) - 1) * q)]


def RPE_calc(estimated, ground_truth):
    traj_gt = read_trajectory(ground_truth)
    traj_est = read_trajectory(estimated)

    result = evaluate_trajectory(traj_gt,
                                 traj_est,
                                 int(10000),
                                 'store_true',
                                 float(1.0),
                                 'f',
                                 float(0.0),
                                 float(1.0))

    stamps = np.array(result)[:, 0]
    trans_error = np.array(result)[:, 4]
    rot_error = np.array(result)[:, 5]

    print("\n\n--------------------Relative Pose Error Analysis--------------------\n")
    print("compared_pose_pairs %d pairs" % (len(trans_error)))
    print("translational_error.rmse %f m" % np.sqrt(np.dot(trans_error, trans_error) / len(trans_error)))
    print("translational_error.mean %f m" % np.mean(trans_error))
    print("translational_error.median %f m" % np.median(trans_error))
    print("translational_error.std %f m" % np.std(trans_error))
    print("translational_error.min %f m" % np.min(trans_error))
    print("translational_error.max %f m" % np.max(trans_error))
    print("rotational_error.rmse %f deg" % (np.sqrt(np.dot(rot_error, rot_error) / len(rot_error)) * 180.0 / np.pi))
    print("rotational_error.mean %f deg" % (np.mean(rot_error) * 180.0 / np.pi))
    print("rotational_error.median %f deg" % (np.median(rot_error) * 180.0 / np.pi))
    print("rotational_error.std %f deg" % (np.std(rot_error) * 180.0 / np.pi))
    print("rotational_error.min %f deg" % (np.min(rot_error) * 180.0 / np.pi))
    print("rotational_error.max %f deg" % (np.max(rot_error) * 180.0 / np.pi))

    return stamps, trans_error, rot_error

