"""
Copy from [calculate_homography](https://github.com/kevin-george/cv_tools/blob/master/calculate_homography.py)

The reference can be found at [](http://igt.ip.uca.fr/~ab/Classes/DIKU-3DCV2/Handouts/Lecture16.pdf)
"""

import numpy as np
import cv2

def check_solutions(fp, sp, K, R1, R2, t):
    """Using cv2.triangulatePoints to calculate position
    in real-world co-ordinate system. If z value for both
    points are positive, then that is our solution.
    More details can be found in the paper,
    David Nister, An efficient solution to the five-point relative pose problem.
    TODO: Need to take care of points that are far away
    Args:
        fp: Set of matched points from first image
        sp: Set of matched points from second image
        K: Calibration matrix
        R1: First rotation matrix
        R2: Second rotation matrix
        t: Translation vector
    Returns:
        R: Correct rotation matrix
    """
    dist = 50.0
    P0 = np.eye(3,4)
    P1 = np.dot(K, np.concatenate((R1, t.reshape(3,1)), axis=1))
    Q = cv2.triangulatePoints(P0, P1, fp.T, sp.T)
    mask1 = (Q[2, :] * Q[3, :]) > 0
    Q[0, :] /= Q[3, :]
    Q[1, :] /= Q[3, :]
    Q[2, :] /= Q[3, :]
    Q[3, :] /= Q[3, :]
    mask1 &= (Q[2,:] < dist)
    Q = np.dot(P1, Q)
    mask1 &= (Q[2,:] > 0)
    mask1 &= (Q[2,:] < dist)

    P2 = np.dot(K, np.concatenate((R2, t.reshape(3, 1)), axis=1))
    Q = cv2.triangulatePoints(P0, P2, fp.T, sp.T)
    mask2 = (Q[2, :] * Q[3, :]) > 0
    Q[0, :] /= Q[3, :]
    Q[1, :] /= Q[3, :]
    Q[2, :] /= Q[3, :]
    Q[3, :] /= Q[3, :]
    mask2 &= (Q[2, :] < dist)
    Q = np.dot(P1, Q)
    mask2 &= (Q[2, :] > 0)
    mask2 &= (Q[2, :] < dist)

    P3 = np.dot(K, np.concatenate((R1, -t.reshape(3, 1)), axis=1))
    Q = cv2.triangulatePoints(P0, P3, fp.T, sp.T)
    mask3 = (Q[2, :] * Q[3, :]) > 0
    Q[0, :] /= Q[3, :]
    Q[1, :] /= Q[3, :]
    Q[2, :] /= Q[3, :]
    Q[3, :] /= Q[3, :]
    mask3 &= (Q[2, :] < dist)
    Q = np.dot(P1, Q)
    mask3 &= (Q[2, :] > 0)
    mask3 &= (Q[2, :] < dist)

    P4 = np.dot(K, np.concatenate((R2, -t.reshape(3, 1)), axis=1))
    Q = cv2.triangulatePoints(P0, P4, fp.T, sp.T)
    mask4 = (Q[2, :] * Q[3, :]) > 0
    Q[0, :] /= Q[3, :]
    Q[1, :] /= Q[3, :]
    Q[2, :] /= Q[3, :]
    Q[3, :] /= Q[3, :]
    mask4 &= (Q[2, :] < dist)
    Q = np.dot(P1, Q)
    mask4 &= (Q[2, :] > 0)
    mask4 &= (Q[2, :] < dist)

    good1 = np.count_nonzero(mask1)
    good2 = np.count_nonzero(mask2)
    good3 = np.count_nonzero(mask3)
    good4 = np.count_nonzero(mask4)

    print("The four good:{},{},{},{}".format(good1, good2, good3, good4))

    max_count = max(good1, good2, good3, good4)
    if max_count == good1:
        return R1, t
    elif max_count == good2:
        return R2, t
    elif max_count == good3:
        return R1, -t
    elif max_count == good4:
        return R2, -t

    return None


def rotate_angle(R):
    """
    http://www.cnblogs.com/singlex/p/RotateMatrix2Euler.html
    :param R:
    :return: thetaz, thetay, thetax
    """
    r11 = R[0][0]
    r21 = R[1][0]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]

    from math import pi, atan2, sqrt
    z = atan2(r21, r11)/pi*180
    y = atan2(-r31, sqrt(r32*r32 + r33*r33))/pi*180
    x = atan2(r32, r33)/pi*180

    print("rotate_angle:\nz:{}\ny:{}\nx:{}".format(z, y, x))