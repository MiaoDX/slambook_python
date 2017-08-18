"""
The results are not so ideal. Potential errors are here!!!

We are going to simulate a stereo system:
The first image can be treat as the left camera, the second (move x=40mm) can be treated as the second image

[ref:使用OpenCV/python进行双目测距](http://www.cnblogs.com/zhiyishou/p/5767592.html)
[ref:机器视觉学习笔记（8）——基于OpenCV的Bouguet立体校正](http://blog.csdn.net/xuelabizp/article/details/50476639)
"""

from triangulation_2d3d2d import *



def stereoRectifyCalc(K, size, R, t):
    """
    :param K:
    :param size:
    :param R:
    :param t:
    :return:
    """

    """ stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T[, R1[, R2[, P1[, P2[, Q[, flags[, alpha[, newImageSize]]]]]]]]) -> R1, R2, P1, P2, Q, validPixROI1, validPixROI2 """

    dist = np.zeros(5) # since we are using the same camera, this dose not matter at all!!
    #R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K, dist, K, dist, size, R, t)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K, dist, K, dist, size, R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)


    print("R1:{}\n,R2:{}\n,P1:{}\n,P2:{}\n,Q:{}".format(R1, R2, P1, P2, Q))
    left_map1, left_map2 = cv2.initUndistortRectifyMap(K, dist, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(K, dist, R2, P2, size, cv2.CV_16SC2)

    return left_map1, left_map2, right_map1, right_map2, Q



def testDis2(left_map1, left_map2, right_map1, right_map2, Q, frame1 , frame2):
    import utils
    import imutils

    """
    [4.3检验效果](http://blog.csdn.net/xuelabizp/article/details/50476639)

    Mat img1 = imread("left01.jpg"), img1r;
    Mat img2 = imread("right01.jpg"), img2r;

    Mat img(imageSize.height, imageSize.width * 2, CV_8UC3);//高度一样，宽度双倍
    imshow("rectified", img);
    remap(img1, img1r, rmap[0][0], rmap[0][1], CV_INTER_AREA);//左校正
    remap(img2, img2r, rmap[1][0], rmap[1][1], CV_INTER_AREA);//右校正

    Mat imgPart1 = img( Rect(0, 0, imageSize.width, imageSize.height) );//浅拷贝
    Mat imgPart2 = img( Rect(imageSize.width, 0, imageSize.width, imageSize.height) );//浅拷贝
    resize(img1r, imgPart1, imgPart1.size(), 0, 0, CV_INTER_AREA);
    resize(img2r, imgPart2, imgPart2.size(), 0, 0, CV_INTER_AREA);

    //画横线
    for( int i = 0; i < img.rows; i += 32 )
        line(img, Point(0, i), Point(img.cols, i), Scalar(0, 255, 0), 1, 8);

    //显示行对准的图形
    Mat smallImg;//由于我的分辨率1:1显示太大，所以缩小显示
    resize(img, smallImg, Size(), 0.5, 0.5, CV_INTER_AREA);
    imshow("rectified", smallImg);
    :return:
    """

    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)

    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)


    height, width = img1_rectified.shape[:2]
    cv2.imshow("imgL", imutils.resize(imgL, width//10))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("img1_rectified.shape:{}".format(img1_rectified.shape))
    print("imgL.shape:{}".format(imgL.shape))

    vis = np.concatenate((img1_rectified, img2_rectified), axis=1)

    plt.imshow(utils.cv2plt(vis)), plt.show()


def testDis(left_map1, left_map2, right_map1, right_map2, Q, frame1 , frame2):
    import numpy as np
    import cv2
    import imutils
    import pylab

    height, width = frame1.shape[:2]
    RESIZE = 10

    cv2.namedWindow("left")
    cv2.namedWindow("right")
    cv2.namedWindow("depth")
    cv2.moveWindow("left", 0, 0)
    cv2.moveWindow("right", width//RESIZE, 0)
    cv2.createTrackbar("num", "depth", 0, 10, lambda x: None)
    cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)
    # camera1 = cv2.VideoCapture(0)
    # camera2 = cv2.VideoCapture(1)

    # 添加点击事件，打印当前点的距离
    def callbackFunc(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            print(threeD[y][x]) # the coordinate seems a little wrong

    cv2.setMouseCallback("depth", callbackFunc, None)

    while True:
        # ret1, frame1 = camera1.read()
        # ret2, frame2 = camera2.read()
        #
        # if not ret1 or not ret2:
        #     break


        # 根据更正map对图片进行重构
        img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)

        # 将图片置为灰度图，为StereoBM作准备
        imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

        # 两个trackbar用来调节不同的参数查看效果
        num = cv2.getTrackbarPos("num", "depth")
        blockSize = cv2.getTrackbarPos("blockSize", "depth")
        if blockSize % 2 == 0:
            blockSize += 1
        if blockSize < 5:
            blockSize = 5

        # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
        stereo = cv2.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)
        disparity = stereo.compute(imgL, imgR)

        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 将图片扩展至3d空间中，其z方向的值则为当前的距离
        threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., Q)

        # for i in range(10):
        #     print(threeD[i])

        # cv2.imshow("left", img1_rectified)
        # cv2.imshow("right", img2_rectified)
        # cv2.imshow("depth", disp)

        cv2.imshow("left", imutils.resize(img1_rectified, width//RESIZE))
        cv2.imshow("right", imutils.resize(img2_rectified, width//RESIZE))
        cv2.imshow("depth", imutils.resize(disp, width//RESIZE))

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            print("Going to write to disk")
            cv2.imwrite("./snapshot/BM_left.jpg", imgL)
            cv2.imwrite("./snapshot/BM_right.jpg", imgR)
            cv2.imwrite("./snapshot/BM_depth.jpg", disp)
            print("Wirte done")
        elif key == ord("c"):
            print("show the hist, block operation")
            pylab.hist(threeD[:, -1], bins=100)
            pylab.show()

    # camera1.release()
    # camera2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    base_dir = "H:/projects/SLAM/python_code/dataset/our/trajs2/"


    im1_file = base_dir + '1.jpg'
    im2_file = base_dir + '4.jpg'

    K = np.array([[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935], [0, 0, 1]]) # Canon5DMarkIII-EF50mm

    #DEBUG = False

    if DEBUG:
        print("HHHHH")


    # im1 = cv2.imread(im1_file, 0)
    # im2 = cv2.imread(im2_file, 0)

    im1 = cv2.imread(im1_file)
    im2 = cv2.imread(im2_file)

    """
    # find_feature_matches( im1, im2 )
    kp1, des1 = find_keypoints_and_description(im1)
    kp2, des2 = find_keypoints_and_description(im2)

    matches = find_feature_matches_from_keypoints_and_descriptors(im1, kp1, des1, im2, kp2, des2)
    F, E, R, t, pts1_F, pts2_F, pts1_E, pts2_E = find_F_E_R_t(kp1, kp2, matches, K)
    """

    """
    R = np.array([[ 0.99995568,  0.00584152,  0.00738401],
     [-0.00577213,  0.99993931, -0.00938406],
     [-0.00743838,  0.00934102,  0.99992871]]
    )

    t = np.array([[-0.99457732], [ 0.06874528], [ 0.07803867]])
    t = t/(t[0]/-40.0) #1-4

    height, width = im1.shape[:2]
    size = (width, height)
    left_map1, left_map2, right_map1, right_map2, Q = stereoRectifyCalc(K, size, R, t)

    testDis(left_map1, left_map2, right_map1, right_map2, Q, im1, im2)
    #testDis2(left_map1, left_map2, right_map1, right_map2, Q, im1, im2)    
    """


    # just do it, without all funcy and forward and backward calculation
    R = np.eye(3)
    t = np.array([-40.0, 0, 0]) # 1-4
    height, width = im1.shape[:2]
    size = (width, height)
    left_map1, left_map2, right_map1, right_map2, Q = stereoRectifyCalc(K, size, R, t)
    testDis(left_map1, left_map2, right_map1, right_map2, Q, im1, im2)
    #testDis2(left_map1, left_map2, right_map1, right_map2, Q, im1, im2)

