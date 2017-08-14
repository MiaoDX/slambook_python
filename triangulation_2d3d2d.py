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

    """
    kp1, des1 = orb.detectAndCompute(img, None)
    # Make sure the key-points are the same
    
    for kpi, kppi, kp1i in zip(kp, kpp, kp1):
        # print("{},{},{}\n".format(kpi, kppi,cv2.KeyPoint_overlap(kpi, kppi)))
        print("{},{},{}".format(cv2.KeyPoint_convert([kpi]), cv2.KeyPoint_convert([kppi]),  cv2.KeyPoint_convert([kp1i])))
        assert cv2.KeyPoint_overlap(kpi, kppi) == 1.0
        assert cv2.KeyPoint_overlap(kpi, kp1i) == 1.0
        assert cv2.KeyPoint_overlap(kppi, kp1i) == 1.0
        
    for d1, d2 in zip(des, des1):
        print("des1:{},des2:{},equal{}\n".format(d1, d2, d1==d2))
    """

    # draw only keypoints location,not size and orientation
    # img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    # plt.imshow(img2),plt.show()
    return kpp, des





def find_feature_matches_from_keypoints_and_descriptors(img1, kp1, des1, img2, kp2, des2):

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    for m in matches[:5]:
        print("distance in match:{}".format(m.distance))

    # 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    prune_dis = max(matches[0].distance, 30.0)
    matches = list(filter(lambda x: x.distance <= prune_dis, matches))

    print("We found {} pairs of points in total".format(len(matches)))

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

    plt.imshow(img3), plt.show()
    pass


if __name__ == '__main__':
    base_dir = 'H:/projects/SLAM/slambook/ch7/'
    im1_file = base_dir+'1.png'
    im2_file = base_dir + '2.png'

    im1 = cv2.imread(im1_file)
    im2 = cv2.imread(im2_file)

    # find_feature_matches( im1, im2 )
    kp1, des1 = find_keypoints_and_description(im1)
    kp2, des2 = find_keypoints_and_description(im2)

    find_feature_matches_from_keypoints_and_descriptors(im1, kp1, des1, im2, kp2, des2)