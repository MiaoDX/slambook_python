"""
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
"""

if __name__ == "__main__":
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt


    base_dir = "H:/projects/SLAM/python_code/dataset/our/trajs2/"

    im1_file = base_dir + '1.jpg'
    im2_file = base_dir + '4.jpg'

    img1 = cv2.imread(im1_file, 0)  # queryImage
    img2 = cv2.imread(im2_file, 0)  # trainImage

    # Initiate SIFT detector
    # sift = cv2.SIFT()
    # fd_de = cv2.xfeatures2d.SIFT_create()
    fd_de = cv2.xfeatures2d.SURF_create()
    # fd_de = cv2.ORB_create(nfeatures=5000)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = fd_de.detectAndCompute(img1, None)
    kp2, des2 = fd_de.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    index_params_orb = dict(algorithm=6, # FLANN_INDEX_LSH <-> 6, not so sure why I can not use it
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2

    index_params_orb2 = dict(algorithm=6, # FLANN_INDEX_LSH <-> 6, not so sure why I can not use it
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2

    """
    # FAILED
    index_params_auto = dict(algorithm=255,
                             target_precision=0.9,
                             build_weight=0.01,
                             memory_weight=0,
                             sample_fraction=0.1) # FLANN_INDEX_AUTOTUNED <-> 255
    # flann = cv2.FlannBasedMatcher(index_params_auto, search_params)
    """

    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    #flann = cv2.FlannBasedMatcher(index_params_orb, search_params)
    #flann = cv2.FlannBasedMatcher(index_params_orb2, search_params)
    #matches_HAMMING = flann.match(des1, des2) # for ORB like

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]



    matches_good = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            matches_good.append(m)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    print("Matches with ratio test:{}->{}".format(len(matches), len(matches_good)))

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    #img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches_good, None, flags=2)


    plt.imshow(img3, ), plt.show()