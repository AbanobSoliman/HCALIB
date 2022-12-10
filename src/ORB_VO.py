import sys
sys.path.append('.')
sys.path.append('./evaluate')
from pose_evaluation_utils import *

Pose = []
def mono_VO_ORB(img_data_dir, K_rgb, image_w, image_h):
    width = image_w
    height = image_h
    fx, fy, cx, cy = [K_rgb[0,0], K_rgb[1,1], K_rgb[0,2], K_rgb[1,2]]

    # get the image list in the directory
    img_list = glob(img_data_dir + '/*.png')
    img_list.sort()
    num_frames = len(img_list)

    for i in range(num_frames):
        curr_img = cv.imread(img_list[i], 0)

        if i == 0:
            curr_R = np.eye(3)
            curr_t = np.array([0, 0, 0])
        else:
            prev_img = cv.imread(img_list[i - 1], 0)

            # ====================== Use ORB Feature to do feature matching =====================#
            # create ORB features
            orb = cv.ORB_create(nfeatures=6000)

            # find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(prev_img, None)
            kp2, des2 = orb.detectAndCompute(curr_img, None)

            # use brute-force matcher
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

            # Match ORB descriptors
            matches = bf.match(des1, des2)

            # Sort the matched keypoints in the order of matching distance
            # so the best matches came to the front
            matches = sorted(matches, key=lambda x: x.distance)

            img_matching = cv.drawMatches(prev_img, kp1, curr_img, kp2, matches[0:100], None)
            cv.imshow('feature matching', img_matching)

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            # compute essential matrix
            E, mask = cv.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999,
                                           threshold=1)
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            _, R, t, mask = cv.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))

            # get camera motion
            R = R.transpose()
            t = -np.matmul(R, t)

            if i == 1:
                curr_R = R
                curr_t = t
            else:
                curr_R = np.matmul(prev_R, R)
                curr_t = np.matmul(prev_R, t) + prev_t

            # draw the current image with keypoints
            curr_img_kp = cv.drawKeypoints(curr_img, kp2, None, color=(0, 255, 0), flags=0)
            cv.imshow('keypoints from current image', curr_img_kp)

        # save current pose
        [tx, ty, tz] = [curr_t[0], curr_t[1], curr_t[2]]
        qw, qx, qy, qz = rot2quat(curr_R)

        Pose.append([tx, ty, tz, qw, qx, qy, qz])

        prev_R = curr_R
        prev_t = curr_t
cv.destroyAllWindows()