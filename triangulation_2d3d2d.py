"""
# Transport of [triangulation_2d3d2d](https://github.com/MiaoDX-fork-and-pruning/slambook/blob/windows/ch7/triangulation_2d3d2d.cpp)

1.find feature matches, [ref](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html)
2.estimate the relative R,t (eight-points, five-points). In real projects, we get the `exact` relative motion ahead
3.triangulation to find the 3d points
4.prune out some points
5.use solvePnP to re-calc the R,t

"""
import cv2
from matplotlib import pyplot as plt
import numpy as np

from utils import rotate_angle


def find_keypoints_and_description(img):
    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # kp = kp[:5]
    # compute the descriptors with ORB
    # NOTE: it will output another key-points, but the are presenting the same thing
    kpp, des = orb.compute(img, kp)

    print("KP points:{}".format(len(kpp)))

    if DEBUG:
        print("first 10 key points:")
        for kpi in kp[:10]:
            print("{}".format(cv2.KeyPoint_convert([kpi])))

    """
    # Make sure the key-points are the same
    
    kp1, des1 = orb.detectAndCompute(img, None)    
    for kpi, kppi, kp1i in zip(kp, kpp, kp1):
        # print("{},{},{}\n".format(kpi, kppi,cv2.KeyPoint_overlap(kpi, kppi)))
        print("{},{},{}".format(cv2.KeyPoint_convert([kpi]), cv2.KeyPoint_convert([kppi]),  cv2.KeyPoint_convert([kp1i])))
        assert cv2.KeyPoint_overlap(kpi, kppi) == 1.0
        assert cv2.KeyPoint_overlap(kpi, kp1i) == 1.0
        assert cv2.KeyPoint_overlap(kppi, kp1i) == 1.0
        
    for d1, d2 in zip(des, des1):
        print("des1:{},des2:{},equal{}\n".format(d1, d2, d1==d2))
    """

    if DEBUG:
        # draw only keypoints location,not size and orientation
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        plt.imshow(img2),plt.show()
    return kpp, des




def find_feature_matches_from_keypoints_and_descriptors(img1, kp1, des1, img2, kp2, des2):

    # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING) # to make sure we are the same as the C++ version

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches_sorted = sorted(matches, key=lambda x: x.distance)

    print("Max distance:{}, min distance:{}".format(matches_sorted[-1].distance, matches_sorted[0].distance))
    # 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    prune_dis = max(matches_sorted[0].distance, 30.0)

    matches = list(filter(lambda x: x.distance <= prune_dis, matches))

    print("We found {} pairs of points in total".format(len(matches)))

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    plt.imshow(img3), plt.show()



    return matches


def essentialFromFundamental(F, K1, K2):
    """
    Normally, we only need to set K1 and K2 to the same one
    """
    return K2.T.dot(F).dot(K1)

def find_F_E_R_t(kp1, kp2, matches, K):
    """
    [ref](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html)
    :param kp1:
    :param kp2:
    :param matches:
    :return:
    """

    pts1 = []
    pts2 = []
    for m in matches:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

    #  Let’s find the Fundamental Matrix
    #pts1 = np.int32(pts1)
    #pts2 = np.int32(pts2)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    """
    for i in range(len(pts1)):
        print("i:{}".format(i))
        print("p1:{}".format(pts1[i]))
        print("p2:{}".format(pts2[i]))
    """

    # F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    F, mask_F = cv2.findFundamentalMat(pts1, pts2, method = cv2.FM_8POINT)

    # We select only inlier points
    pts1_F = pts1[mask_F.ravel() == 1]
    pts2_F = pts2[mask_F.ravel() == 1]
    print("inner points in findFundamentalMat num:{}".format(len(pts1_F)))
    print("F:\n{}".format(F))


    # [ref](https://github.com/kevin-george/cv_tools/blob/master/calculate_fmatrix.py#L195)
    # Pick a random match, homogenize it and check if Fundamental
    # matrix calculated is good enough x' * F * x = 0
    import random
    import sys
    index = random.randint(0, len(pts1))
    pt1 = np.array([[pts1[index][0], pts1[index][1], 1]])
    pt2 = np.array([[pts2[index][0], pts2[index][1], 1]])
    error = np.dot(pt1, np.dot(F, pt2.T))
    if error >= 1:
        print("Fundamental matrix calculation Failed! " + str(error))
        # sys.exit()
        input("Press Enter to continue ...")


    E, mask_E = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K)
    # We select only inlier points
    pts1_E = pts1[mask_E.ravel() == 1]
    pts2_E = pts2[mask_E.ravel() == 1]
    print("inner points in findEssentialMat num:{}".format(len(pts1_E)))
    print("E:\n{}".format(E))

    RE1, RE2, tE = cv2.decomposeEssentialMat(E) # there will be two answers for R, and four answers in total: R1 t, R1 -t, R2 t, R2 -t
    print("R t from decomposeEssentialMat")
    print("RE1:{}".format(RE1))
    r1, _ = cv2.Rodrigues(RE1)
    print("r:\n{}".format(r1))
    print("RE2:{}".format(RE2))
    r2, _ = cv2.Rodrigues(RE2)
    print("r:\n{}".format(r2))
    print("tE:{}".format(tE))

    # it should be the same as the recoverPose, however, it is not. It seem that the check_solutions is somewhat wrong
    import utils
    R_checked, t_checked = utils.check_solutions(pts1_E, pts2_E, K, RE1, RE2, tE)
    print("R t from decomposeEssentialMat and checked")
    print("R_checked:{}".format(R_checked))
    r_checked, _ = cv2.Rodrigues(R_checked)
    print("r_checked:\n{}".format(r_checked))
    print("t_checked:{}".format(t_checked))

    E_FF = essentialFromFundamental(F, K, K) # it should be the same, however, it is not :(
    print("E from fundamental:\n{}".format(E_FF))
    """
    RE1_FF, RE2_FF, tE_FF = cv2.decomposeEssentialMat(E_FF)  # there will be two answers for R
    print("R t from decomposeEssentialMat, which E from F")
    print("RE1_FF:{}".format(RE1_FF))
    r1_FF, _ = cv2.Rodrigues(RE1_FF)
    print("r:\n{}".format(r1_FF))
    print("RE2_FF:{}".format(RE2_FF))
    r2_FF, _ = cv2.Rodrigues(RE2_FF)
    print("r_FF:\n{}".format(r2_FF))
    print("tE_FF:{}".format(tE_FF))
    """


    H, mask_H = cv2.findHomography(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3)
    # We select only inlier points
    pts1_H = pts1[mask_H.ravel() == 1]
    pts2_H = pts2[mask_H.ravel() == 1]
    print("inner points in findHomography num:{}".format(len(pts1_H)))
    print("H:\n{}".format(H))

    _, RH, tH, _ = cv2.decomposeHomographyMat(H, K) # it will have four answers, we need to find which is better
    print("R t from decomposeHomographyMat, there can be up many R")
    print("RH:{}".format(RH))
    for RHi in RH:
        rH, _ = cv2.Rodrigues(RHi)
        print("rH:{}".format(rH))
    print("tH:{}".format(tH))

    assert len(RH) == 4
    RH_checked, th_checked = utils.check_solutions(pts1_H, pts2_H, K, RH[0], RH[3], tH[0])
    print("R t from decomposeHomographyMat and checked")
    print("RH_checked:{}".format(RH_checked))
    rH_checked, _ = cv2.Rodrigues(RH_checked)
    print("rH_checked:\n{}".format(rH_checked))
    print("tH_checked:{}".format(th_checked))



    _, R, t, mask_rc = cv2.recoverPose(E, pts1, pts2, K) # this can have the determiate results, so we choose it
    # We select only inlier points
    print("inner points in recoverPose num:{}".format(len(mask_rc)))
    pts1_rc = pts1[mask_rc.ravel() == 1]
    pts2_rc = pts2[mask_rc.ravel() == 1]

    print("R t from recoverPose")
    r, _ = cv2.Rodrigues(R)
    print("R:\n{}".format(R))
    print("r:\n{}".format(r))
    rotate_angle(R)
    print("t:\n{}".format(t))

    # find_F_E_R_t
    return F, E, R, t, pts1_F, pts2_F, pts1_E, pts2_E


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
    print("t:\n{}".format(t))

    return R, t

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    print("img1.shape:{}".format(img1.shape))
    r,c = img1.shape[:2] # it can be gray or RGB
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def draw_epilines_from_F(img1, img2, pts1, pts2, F):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image

    # make them to int so that we can draw on them
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()

if __name__ == '__main__':
    # base_dir = 'H:/projects/SLAM/slambook/ch7/'
    # base_dir = "H:/projects/SLAM/python_code/dataset/RGBD_BENCHMARK/rgbd_dataset_freiburg2_xyz_chosen/simple/"
    # im1_file = base_dir+'1.png'
    # im2_file = base_dir + '2.png'


    # base_dir = "H:/projects/SLAM/python_code/dataset/our/trajs/"
    # im1_file = base_dir+'0_b.jpg'
    # im2_file = base_dir + '1_b.jpg'
    base_dir = "H:/projects/SLAM/python_code/dataset/our/trajs2/"
    base_dir2 = "H:/projects/SLAM/python_code/dataset/our/trajs_bright/"
    base_dir3 = "H:/projects/SLAM/python_code/dataset/our/trajs_r/"

    im1_file = base_dir + '1.jpg'
    # im2_file = base_dir + '3.jpg'
    # im2_file = base_dir2 + '1.jpg'
    im2_file = base_dir + '7a.jpg'

    DEBUG = False

    if DEBUG:
        print("HHHHH")


    # im1 = cv2.imread(im1_file, 0)
    # im2 = cv2.imread(im2_file, 0)

    im1 = cv2.imread(im1_file)
    im2 = cv2.imread(im2_file)

    # find_feature_matches( im1, im2 )
    kp1, des1 = find_keypoints_and_description(im1)
    kp2, des2 = find_keypoints_and_description(im2)

    matches = find_feature_matches_from_keypoints_and_descriptors(im1, kp1, des1, im2, kp2, des2)

    # K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])
    # K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
    K = np.array([[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935], [0, 0, 1]]) # Canon5DMarkIII-EF50mm

    F, E, R, t, pts1_F, pts2_F, pts1_E, pts2_E = find_F_E_R_t(kp1, kp2, matches, K)

    draw_epilines_from_F(im1, im2, pts1_F, pts2_F, F)

    #t = t/(t[0]/-40.0) #1-4
    #t = t / (t[-1] / -15.0) #1-2
    t = t/(t[0]/40.0) #1-7 # it will be totally wrong if we are using 7a
    print("Scaled t:{}".format(t))
    pts1_cam_Nx3 = triangulation(R, t, pts1_E, pts2_E, K)
    for i in range(10):
       print(pts1_cam_Nx3[i])

    import pylab
    pylab.hist(pts1_cam_Nx3[:,-1], bins=10)
    pylab.show()


    # just do it, without all funcy and forward and backward calculation
    R = np.eye(3)

    #t = np.array([0.0, 0.0, -15.0])  # 1-2
    #t = np.array([-40.0, 0, 0])  # 1-4
    t = np.array([40.0, 0, 0])  # 1-7a # we can still use 7a without any problem

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


    # prune out some points and re-calc the R, t from the 3d points
    pts1_cam_Nx3_half = pts1_cam_Nx3[:len(pts1_cam_Nx3)//2]
    pts2_E_half = pts2_E[:len(pts2_E)//2]

    # PNPSolver_img2_points_and_3DPoints(pts1_cam_Nx3, pts2_E, K)
    # PNPSolver_img2_points_and_3DPoints(pts1_cam_Nx3_half, pts2_E_half, K)

"""
SOME NOTES:
1. image color and corresponding difference, `im1 = cv2.imread(im1_file)` and `im1 = cv2.imread(im1_file, 0)`
2. inner points in findFundamentalMat, when it is not the same as the input num?
3. when we calc Fundamental Matrix
    pts1 = np.int32(pts1) and pts1 = np.array(pts1) will lead to very different answers, so, which is better??
4. Matrix manipulation in C++ and python
    [Opencv中Mat矩阵相乘——点乘、dot、mul运算详解](http://blog.csdn.net/dcrmg/article/details/52404580
        
    arr = [[1,2,3],[4,5,6]]
    a = np.array(arr)
    b = np.array(arr)
    
    C++     python
    *       dot
    a*b.t()     a.dot(b.T) or np.dot(a,b.T)
    
    dot     * -> sum
    a.dot(b)    (a*b).sum()
    
    mul     *
    a.mul(b)    a*b
    
    So, C++: E = K2.t () * F * K1;
    python:  E = K2.T.dot(F).dot(K1)
    
5. The corresponding relationship between F,E and H

6. [recoverPose](http://docs.opencv.org/3.2.0/d9/d0c/group__calib3d.html)

    This function decomposes an essential matrix using decomposeEssentialMat and then verifies possible pose hypotheses by doing cheirality check. 
    The cheirality check basically means that the triangulated 3D points should have positive depth. Some details can be found in [119] .
    
7. check_solutions(fp, sp, K, R1, R2, t):
    should have same result as recoverPose, but without good luck    
    
8. some note online
    * [import cv2 #Notes](https://pythonpath.wordpress.com/import-cv2/)
    
9. The usage of 3xN and Nx3

10. findFundamentalMat, findEssentialMat all have `mask` return value and means `inner points`, we should take care of them

11. [相机位姿求解问题？](https://www.zhihu.com/collection/128407057)
"""