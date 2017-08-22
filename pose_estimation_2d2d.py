"""

Something interesting, but not so pleasing.
IN our recoverPose, seems the output mask in None, a.k.a inliners are none. As said in [recoverPose](http://docs.opencv.org/3.2.0/d9/d0c/group__calib3d.html),
In the output mask only inliers which pass the cheirality check, so, we did a wrong calculation?
Update: after try print out the values, we find that, the mask is make up by 0 and 255, not like others with 0 and 1.

"""

"""
# Transport of [triangulation_2d3d2d](https://github.com/MiaoDX-fork-and-pruning/slambook/blob/windows/ch7/triangulation_2d3d2d.cpp)

1.find feature matches, [ref](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html)
2.estimate the relative R,t (eight-points, five-points). In real projects, we get the `exact` relative motion ahead
3.triangulation to find the 3d points
4.prune out some points
5.use solvePnP to re-calc the R,t

"""
import cv2
from mpl_toolkits.mplot3d import Axes3D #projection='3d'
from matplotlib import pyplot as plt
import numpy as np

from utils import *
import imutils


DEBUG = False

def find_keypoints_and_description(img):
    # Initiate STAR detector
    orb = cv2.ORB_create(1000)

    """
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    
    # compute the descriptors with ORB
    # NOTE: it will output another key-points, but the are presenting the same thing
    kpp, des = orb.compute(img, kp)
    """

    kpp, des = orb.detectAndCompute(img, None)
    if DEBUG:
        print("KP points:{}".format(len(kpp)))



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
        img2 = cv2.drawKeypoints(img, kpp, None, color=(0, 255, 0), flags=0)
        plt.imshow(img2),plt.show()
    return kpp, des



def DEBUG_Matches(img1, kp1, img2, kp2, matches, name):
    """ drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]]) -> outImg """
    im = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    plt.imshow(cv2plt(im)), plt.title(name), plt.show()

def find_matches_from_descriptors(des1, des2):

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches_sorted = sorted(matches, key=lambda x: x.distance)

    if DEBUG:
        print("Max distance:{}, min distance:{}".format(matches_sorted[-1].distance, matches_sorted[0].distance))
    # 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    prune_dis = max(matches_sorted[0].distance, 30.0)

    matches = list(filter(lambda x: x.distance <= prune_dis, matches))

    if DEBUG:
        print("We found {} pairs of points in total".format(len(matches)))

    return matches


def kps2pts(kps1, kps2, matches):
    pts1 = []
    pts2 = []
    dis = []
    for m in matches:
        pts2.append(kps2[m.trainIdx].pt)
        pts1.append(kps1[m.queryIdx].pt)
        dis.append(m.distance)
    return np.float32(pts1), np.float32(pts2), np.float32(dis)


def print_kps(kps1, kps2, matches, start, end):
    assert len(kps1) == len(kps2)
    assert start>=0 and end > start and len(matches) >= end

    print("All matches:{}, Key points {}-{}:".format(len(matches), start, end))

    pts1, pts2, dis = kps2pts(kps1, kps2, matches)
    print_pts(pts1, pts2, dis, start, end)



def print_pts(pts1, pts2, dis, start, end):
    assert len(pts1) == len(pts2)
    for i in range(start, end):
        print("i:{},p1:{},p2:{},d:{}".format(i, pts1[i], pts2[i], dis[i]))

def find_H_and_refineMatches(kps1, kps2, matches):

    pts1, pts2, _ = kps2pts(kps1, kps2, matches)
    H, mask_H = cv2.findHomography(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3)

    # We select only inlier points
    pts1_H = pts1[mask_H.ravel() == 1]
    pts2_H = pts2[mask_H.ravel() == 1]
    #new_matches = matches[mask_H.ravel() == 1]
    matches_H = []
    matches_H_bad = []
    for i in range(len(matches)):
        if mask_H[i] == 1:
            matches_H.append(matches[i])
        else:
            matches_H_bad.append(matches[i])

    print("In find_H_and_refineMatches, matches:{} -> {}".format(len(matches), len(matches_H)))
    print("H:\n{}".format(H))

    return H, matches_H, matches_H_bad, pts1_H, pts2_H


def find_F_and_refineMatches(kps1, kps2, matches):

    pts1, pts2, _ = kps2pts(kps1, kps2, matches)
    #F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99) # F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # We select only inlier points
    pts1_F = pts1[mask_F.ravel() == 1]
    pts2_F = pts2[mask_F.ravel() == 1]
    #new_matches = matches[mask_H.ravel() == 1]
    matches_F = []
    matches_F_bad = []
    for i in range(len(matches)):
        if mask_F[i] == 1:
            matches_F.append(matches[i])
        else:
            #print("bad in F refine:{}".format(i))
            matches_F_bad.append(matches[i])

    print("In find_F_and_refineMatches, matches:{} -> {}".format(len(matches), len(matches_F)))
    print("F:\n{}".format(F))

    random_check_F(pts1_F, pts2_F, F)

    return F, matches_F, matches_F_bad, pts1_F, pts2_F



def random_check_F(pts1, pts2, F):
    # [ref](https://github.com/kevin-george/cv_tools/blob/master/calculate_fmatrix.py#L195)
    # Pick a random match, homogenize it and check if Fundamental
    # matrix calculated is good enough x' * F * x = 0
    import random
    import sys
    index = random.randint(0, len(pts1)-1)
    pt1 = np.array([[pts1[index][0], pts1[index][1], 1]])
    pt2 = np.array([[pts2[index][0], pts2[index][1], 1]])
    error = np.dot(pt1, np.dot(F, pt2.T))
    if error >= 1:
        print("Fundamental matrix calculation Failed! " + str(error))
        # sys.exit()
        input("Press Enter to continue ...")


def DEBUG_Rt(R, t, name):
    print("DEBUG RT:{}".format(name))

    r, _ = cv2.Rodrigues(R)
    print("R:\n{}".format(R))
    print("r:{}".format(r.T))
    rotate_angle(R)
    print("t:{}".format(t.T))

def DEBUG_Rt_simple(R, t, name):
    print("DEBUG RT:{}".format(name))
    rotate_angle(R)
    print("t:{}".format(t.T))


def scaled_E(E):
    return E/E[-1,-1]


def DEBUG_E(E):
    print("E:\n{}".format(E))
    print("Scaled E:\n{}".format(scaled_E(E)))


def recoverPoseFromE_cv3_with_kps(E, kps1, kps2, matches, K):
    pts1, pts2, _ = kps2pts(kps1, kps2, matches)

    _, R, t, mask_rp = cv2.recoverPose(E, pts1, pts2,
                                       K)  # this can have the determiate results, so we choose it


    # We select only inlier points
    pts1_rp = pts1[mask_rp.ravel() != 0]
    pts2_rp = pts2[mask_rp.ravel() != 0]
    matches_rp = []
    matches_rp_bad = []
    for i in range(len(matches)):
        if mask_rp[i] != 0:
            matches_rp.append(matches[i])
        else:
            #print("bad in F refine:{}".format(i))
            matches_rp_bad.append(matches[i])



    #print("In recoverPoseFromE_cv3, points:{} -> inliner:{}".format(len(pts1), len(pts1_rp)))

    return R, t, matches_rp, matches_rp_bad, pts1_rp, pts2_rp


def recoverPoseFromE_cv3(E, pts1, pts2, K):

    assert len(pts1) == len(pts2)

    """
    RE1, RE2, tE = cv2.decomposeEssentialMat(
        E)  # there will be two answers for R, and four answers in total: R1 t, R1 -t, R2 t, R2 -t
    print("R t from decomposeEssentialMat, note that there are four potential answers")
    DEBUG_Rt(RE1, tE, "RE1, tE")
    DEBUG_Rt(RE2, tE, "RE2, tE")
    """
    assert imutils.is_cv3() == True

    _, R, t, mask_rc = cv2.recoverPose(E, pts1, pts2,
                                       K)  # this can have the determiate results, so we choose it

    # for m in mask_rc:
    #     print(m.ravel())

    # We select only inlier points
    pts1_rc = pts1[mask_rc.ravel() != 0]
    #pts2_rc = pts2[mask_rc.ravel() != 0]
    print("In recoverPoseFromE_cv3, points:{} -> inliner:{}".format(len(pts1), len(pts1_rc)))

    return R, t

def find_E_from_F(F, K1, K2):
    """
    Normally, we only need to set K1 and K2 to the same one
    """
    return K2.T.dot(F).dot(K1)


def find_E_cv3(kp1, kp2, matches, K):

    pts1, pts2, _ = kps2pts(kp1, kp2, matches)

    """ findEssentialMat(points1, points2, cameraMatrix[, method[, prob[, threshold[, mask]]]]) -> retval, mask """
    #E, mask_E = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=0.2)
    E, mask_E = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K)

    # We select only inlier points
    pts1_E = pts1[mask_E.ravel() == 1]
    pts2_E = pts2[mask_E.ravel() == 1]

    #new_matches = matches[mask_H.ravel() == 1]
    matches_E = []
    for i in range(len(matches)):
        if mask_E[i] == 1:
            matches_E.append(matches[i])


    # print("In find_E_cv3, matches:{} -> {}".format(len(matches), len(matches_E)))

    return E, matches_E, pts1_E, pts2_E


def find_F_E_R_t(kp1, kp2, matches, K):
    """
    [ref](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html)
    :param kp1:
    :param kp2:
    :param matches:
    :return:
    """

    E_cv3, _,pts1_E_cv3, pts2_E_cv3 = find_E_cv3(kp1, kp2, matches, K)
    DEBUG_E(E_cv3)
    print("Try to use E from findEssentialMat ot recover pose")
    R_cv3, t_cv3 = recoverPoseFromE_cv3(E_cv3, pts1_E_cv3, pts2_E_cv3, K)
    DEBUG_Rt(R_cv3, t_cv3, "E from findEssential")

    F, _, _, pts1_F, pts2_F = find_F_and_refineMatches(kp1, kp2, matches)

    return F, E_cv3, R_cv3, t_cv3, pts1_F, pts2_F, pts1_E_cv3, pts2_E_cv3


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

    #draw_epilines_from_F(im1, im2, pts1_F, pts2_F, F)
