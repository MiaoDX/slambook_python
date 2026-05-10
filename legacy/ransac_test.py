"""
When we have many many points pairs, the findEssentialMat and recoverPose will give us only one answer anyway,
which will definitely loose the benefit of numbers and we can do some `RANSAC` or `sliding-window` approaches.

Let's say, we have 1000 pairs of matched points, there can be ways to do it:

===========
First:
1. Calculate every 100 points and get 10 answers
2. Remove the answers with less confidence, aka, the less percentage of points passing the cheirality check (in recoverPose)
3. Get the main centroid of the remaining answers
===========

===========
Second:
1. Calculate every 100 points and get 10 answers
2. For each answer, calc the reprojection error (or the epipolar equation) on all points and filter out some answers
3. Get the main centroid of the remaining answers
===========

"""
from pose_estimation_2d2d import *


DEBUG = True


def plot_the_Rt_pairs():
    pass


def split_the_matches(matches, step):
    for i in range(len(matches)//step + 1):
        yield matches[i*step:(i+1)*step]

def remove_less_confidence(kps1, kps2, matches, K, splitnum, threshold):

    Rs = []
    ts = []
    confidences = []

    splited_matches = split_the_matches(matches, splitnum)

    for m in splited_matches:
        if len(m) < 5:
            print("Less than 5 points, no need")
            continue

        #print(len(m))
        E_cv3, matches_E, _, _ = find_E_cv3(kps1, kps2, m, K)
        R, t, matches_rp, matches_rp_bad, _, _ = recoverPoseFromE_cv3_with_kps(E_cv3, kps1, kps2, m, K)

        conf = len(matches_rp)/len(m)
        if conf >= threshold:
            Rs.append(R)
            ts.append(t)
            confidences.append(conf)

    return Rs, ts, confidences

def mean_Rt(Rs, ts):
    rs = []
    for R in Rs:
        rs.append(rotate_angle_no_output(R))

    rs = np.array(rs)
    ts = np.array(ts)

    assert rs[0].shape == (3, 1) and ts[0].shape == (3, 1)

    #print("Rs:{}, ts:{}".format(rs, ts))
    #print("rs.shape:{}, ts.shape:{}".format(rs.shape, ts.shape))
    return rs.mean(axis=0), ts.mean(axis=0)



def test_remove_less_confidence(im1_file, im2_file):


    K = np.array([[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935], [0, 0, 1]])  # Canon5DMarkIII-EF50mm

    #im1_file = '1.jpg'
    #im2_file = '3.jpg'

    #print(im1_file, im2_file)

    im1 = cv2.imread(im1_file, 0)
    im2 = cv2.imread(im2_file, 0)

    kps1, des1 = find_keypoints_and_description(im1)
    kps2, des2 = find_keypoints_and_description(im2)

    matches = find_matches_from_descriptors(des1, des2)


    splitnum = 42

    Rs, ts, confidences = remove_less_confidence(kps1, kps2, matches, K, splitnum, 0.9)
    #test_remove_less_confidence_only_time(kps1, kps2, matches, K, splitnum, 0.6)

    if DEBUG:
        print("The answers num:{} of total:{}".format(len(Rs), len(matches)//splitnum))
        for R, t, con in zip(Rs, ts, confidences):
            print("=============")
            print("Confidence:{}".format(con))
            DEBUG_Rt_simple(R, t, str(con))
            print("=============")


        rs_mean, ts_mean = mean_Rt(Rs, ts)
        print("rs_mean:{}\nts_mean:{}".format(rs_mean.T, ts_mean.T))


def test_remove_less_confidence_only_time(kps1, kps2, matches, K, splitnum, thres):
    start_lsd = cv2.getTickCount()
    freq = cv2.getTickFrequency()

    test_num = 10
    for i in range(test_num):
        Rs, ts, confidences = remove_less_confidence(kps1, kps2, matches, K, splitnum, thres)

    duration_ms_lsd = (cv2.getTickCount() - start_lsd) * 1000 / freq / test_num
    print("Elapsed time for remove less confidence: {} ms".format(duration_ms_lsd ))

def test_remove_less_confidence_time(im1_file, im2_file):
    start_lsd = cv2.getTickCount()
    freq = cv2.getTickFrequency()

    test_num = 10
    for i in range(test_num):
        test_remove_less_confidence(im1_file, im2_file)

    duration_ms_lsd = (cv2.getTickCount() - start_lsd) * 1000 / freq / test_num
    print("Elapsed time for remove less confidence: {} ms".format(duration_ms_lsd ))

if __name__ == "__main__":
    import argparse
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--image1", required=True,
                    help="path to input image1")
    ap.add_argument("-r", "--image2", required=True,
                    help="path to input image2")
    args = vars(ap.parse_args())

    base_dir = "H:/projects/SLAM/python_code/dataset/our/trajs2/"

    im1_file = base_dir + args["image1"]
    im2_file = base_dir + args["image2"]

    #print("{}\n{}".format(im1_file, im2_file))
    print("{}\n{}".format(args["image1"], args["image2"]))

    # test_remove_less_confidence_time()
    test_remove_less_confidence(im1_file, im2_file)


