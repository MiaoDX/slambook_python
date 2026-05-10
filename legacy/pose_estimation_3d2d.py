from pose_estimation_2d2d import *

def pixel2cam(pt, K):
    """
    // 像素坐标转相机归一化坐标
    [1、像素坐标与像平面坐标系之间的关系 ](http://blog.csdn.net/waeceo/article/details/50580607)
    :param pt: point position in pixel coordinate
    :param K:
    :return: point position in camera coordinate
    """
    """
    return Point2f
    (
        (p.x - K.at < double > (0, 2)) / K.at < double > (0, 0),
        (p.y - K.at < double > (1, 2)) / K.at < double > (1, 1)
    );
    """
    return np.array([ (pt[0]-K[0,2])/K[0,0],  (pt[1]-K[1,2])/K[1,1] ])



def triangulation(R, t, pts1, pts2, K):
    """
    https://pythonpath.wordpress.com/2012/08/29/cv2-triangulatepoints/
    :param R:
    :param t:
    :param pts1:
    :param pts2:
    :param K:
    :return: pts3xN
    """

    projM1 = np.eye(4)

    projM2 = np.eye(4)

    print("R.type:{}, R.shape:{}".format(type(R), R.shape))
    print("t.type:{}, t.shape:{}".format(type(t), t.shape))
    projM2[:3, :3] = R
    projM2[:3, -1] = t.T

    assert len(pts1) == len(pts2)
    pts1_cam_Nx2 = np.array([pixel2cam(x, K) for x in pts1])
    pts2_cam_Nx2 = np.array([pixel2cam(x, K) for x in pts2])

    """ triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2[, points4D]) -> points4D """
    pts4d = cv2.triangulatePoints(projM1[:3], projM2[:3], pts1_cam_Nx2.T, pts2_cam_Nx2.T)


    # convert from homogeneous coordinates to 3D
    pts4D = pts4d.T
    pts3D = pts4D[:, :3] / np.repeat(pts4D[:, 3], 3).reshape(-1, 3)

    # plot with matplotlib
    # Ys = pts3D[:, 0]
    # Zs = pts3D[:, 1]
    # Xs = pts3D[:, 2]
    Xs = pts3D[:, 0]
    Ys = pts3D[:, 1]
    Zs = pts3D[:, 2]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xs, Ys, Zs, c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D point cloud: Use pan axes button below to inspect')
    plt.show()

    # 转换成非齐次坐标
    pts1_cam_3xN = pts4d[:3] / pts4d[-1]

    if DEBUG:

        # 验证三角化点与特征点的重投影关系
        pts1_cam_3xN_norm = pts1_cam_3xN / pts1_cam_3xN[-1] # normalization

        print("Points in first camera frame:\n{}".format(pts1_cam_Nx2))
        print("Point projected from 3D:\n{}".format(pts1_cam_3xN_norm.T))

        # -second
        #pts2_trans_3xN = np.array([R.dot(x)+t for x in pts3xN.T])
        pts2_trans_3xN = R.dot(pts1_cam_3xN) + t
        pts2_trans_3xN_norm = pts2_trans_3xN/pts2_trans_3xN[-1]

        print("Points in second camera frame:\n{}".format(pts2_cam_Nx2))
        print("Point reprojected from second frame:\n{}".format(pts2_trans_3xN_norm.T))

    pts1_cam_Nx3 = pts1_cam_3xN.T
    return pts1_cam_Nx3


def PNPSolver_img2_points_and_3DPoints(pts1_cam_Nx3, pts2_pixel_Nx2, K):

    assert len(pts1_cam_Nx3) == len(pts2_pixel_Nx2)

    print("In PNPSolver points num:{}".format(len(pts1_cam_Nx3)))

    """solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, flags]]]]) -> retval, rvec, tvec"""
    _, r, t = cv2.solvePnP(pts1_cam_Nx3, pts2_pixel_Nx2, K, None, useExtrinsicGuess=False)

    print("R t from solvePnP")
    R, _ = cv2.Rodrigues(r)
    print("R:\n{}".format(R))
    print("r:\n{}".format(r))
    rotate_angle(R)
    print("t:\n{}".format(t))

    return R, t


if __name__ == '__main__':
    base_dir = "H:/projects/SLAM/python_code/dataset/our/trajs2/"
    base_dir2 = "H:/projects/SLAM/python_code/dataset/our/trajs_bright/"
    base_dir3 = "H:/projects/SLAM/python_code/dataset/our/trajs_r/"

    im1_file = base_dir + '1.jpg'
    im2_file = base_dir + '4.jpg'

    # DEBUG = False

    if DEBUG:
        print("HHHHH")


    im1 = cv2.imread(im1_file, 0)
    im2 = cv2.imread(im2_file, 0)

    # im1 = cv2.imread(im1_file)
    # im2 = cv2.imread(im2_file)

    kp1, des1 = find_keypoints_and_description(im1)
    kp2, des2 = find_keypoints_and_description(im2)

    matches = find_matches_from_descriptors(des1, des2)

    DEBUG_Matches(im1, kp1, im2, kp2, matches, "Matches first")


    K = np.array([[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935], [0, 0, 1]]) # Canon5DMarkIII-EF50mm

    F, E, R, t, pts1_F, pts2_F, pts1_E, pts2_E = find_F_E_R_t(kp1, kp2, matches, K)

    draw_epilines_from_F(im1, im2, pts1_F, pts2_F, F)

    t = t/(t[0]/-40.0) #1-4
    #t = t / (t[-1] / -15.0) #1-2
    #t = t/(t[0]/40.0) #1-7 # it will be totally wrong if we are using 7a
    print("Scaled t:{}".format(t))
    pts1_cam_Nx3 = triangulation(R, t, pts1_E, pts2_E, K)
    for i in range(10):
       print(pts1_cam_Nx3[i])

    import pylab
    pylab.hist(pts1_cam_Nx3[:,-1], bins=10)
    pylab.show()

    """
    # just do it, without all funcy and forward and backward calculation
    R = np.eye(3)

    #t = np.array([0.0, 0.0, -15.0])  # 1-2
    t = np.array([-40.0, 0, 0])  # 1-4
    # t = np.array([40.0, 0, 0])  # 1-7a # we can still use 7a without any problem

    pts1 = []
    pts2 = []
    for m in matches:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
    pts1_cam_Nx3_another = triangulation(R, t, pts1, pts2, K)
    #pts1_cam_Nx3_another = triangulation(R, t, pts1_E, pts2_E, K) # it is a liitle worse than all points, why?
    for i in range(10):
       print(pts1_cam_Nx3_another[i])

    #pylab.hist(data, normed=1)
    pylab.hist(pts1_cam_Nx3_another[:,-1], bins=10)
    pylab.show()
    """


    """
    # prune out some points and re-calc the R, t from the 3d points
    pts1_cam_Nx3_half = pts1_cam_Nx3[:len(pts1_cam_Nx3)//2]
    pts2_E_half = pts2_E[:len(pts2_E)//2]

    PNPSolver_img2_points_and_3DPoints(pts1_cam_Nx3, pts2_E, K)
    PNPSolver_img2_points_and_3DPoints(pts1_cam_Nx3_half, pts2_E_half, K)
    """
