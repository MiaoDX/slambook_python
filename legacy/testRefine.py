"""
It seems that, with Homograhpy refine is no big harm (:<).
While with Fundamental is not so ideal.
However, when draw out the BAD matched points, I have not see why they are bad -.-
"""


from pose_estimation_2d2d import *

#@profile
def testRefine():
    base_dir = "H:/projects/SLAM/python_code/dataset/our/trajs2/"

    im1_file = base_dir + '1.jpg'
    im2_file = base_dir + '7b.jpg'

    im1 = cv2.imread(im1_file, 0)
    im2 = cv2.imread(im2_file, 0)

    # im1 = cv2.imread(im1_file)
    # im2 = cv2.imread(im2_file)

    kp1, des1 = find_keypoints_and_description(im1)
    kp2, des2 = find_keypoints_and_description(im2)

    matches = find_matches_from_descriptors(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance) # we do the sorting
    # matches = matches[:len(matches)//2] # we crop it
    # np.random.shuffle(matches) # Shuffle it


    K = np.array([[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935], [0, 0, 1]])  # Canon5DMarkIII-EF50mm

    print_kps(kp1, kp2, matches, 0, 10)

    # DEBUG_Matches(im1, kp1, im2, kp2, matches, "Matches first")
    E_cv3, matches_E, pts1_E_cv3, pts2_E_cv3 = find_E_cv3(kp1, kp2, matches, K)

    # for i in range(10):
    #     E_cv3, matches_E, pts1_E_cv3, pts2_E_cv3 = find_E_cv3(kp1, kp2, matches_E, K)  # this should have no changes



    # DEBUG_E(E_cv3)
    print("Try to use E from findEssentialMat to recover pose")
    R_cv3, t_cv3 = recoverPoseFromE_cv3(E_cv3, pts1_E_cv3, pts2_E_cv3, K)
    DEBUG_Rt(R_cv3, t_cv3, "E from findEssential")

    """
    F, matches_F, matches_F_bad, pts1_F, pts2_F = find_F_and_refineMatches(kp1, kp2, matches)

    print_kps(kp1, kp2, matches_F, 0, 10)
    print("==========")
    print_kps(kp1, kp2, matches_F_bad, 0, 10)


    #DEBUG_Matches(im1, kp1, im2, kp2, matches_F, "Matches first")
    #DEBUG_Matches(im1, kp1, im2, kp2, matches_F_bad, "Matches find_H_and_refineMatches BAD")

    E_F_refine, _, pts1_E_F_refine, pts2_E_F_refine = find_E_cv3(kp1, kp2, matches_F, K)
    DEBUG_E(E_F_refine)
    print("Try to use E from findEssentialMat then refined with F to recover pose")
    R_F_refine, t_F_refine = recoverPoseFromE_cv3(E_F_refine, pts1_E_F_refine, pts2_E_F_refine, K)
    DEBUG_Rt(R_F_refine, t_F_refine, "E from findEssentialMat then refined with F")
    """

    """
    H, matches_H, matches_H_bad, pts1_H, pts2_H = find_H_and_refineMatches(kp1, kp2, matches)
    print_kps(kp1, kp2, matches_H, 0, 10)
    print("==========")
    print_kps(kp1, kp2, matches_H_bad, 0, 10)
    #DEBUG_Matches(im1, kp1, im2, kp2, matches_H, "Matches find_H_and_refineMatches")
    #DEBUG_Matches(im1, kp1, im2, kp2, matches_H_bad, "Matches find_H_and_refineMatches BAD")

    E_H_refine, _, pts1_H_refine, pts2_H_refine = find_E_cv3(kp1, kp2, matches_H, K)
    DEBUG_E(E_H_refine)
    print("Try to use E from findEssentialMat then refined with F to recover pose")
    R_H_refine, t_H_refine = recoverPoseFromE_cv3(E_H_refine, pts1_H_refine, pts2_H_refine, K)
    DEBUG_Rt(R_H_refine, t_H_refine, "E from findEssentialMat then refined with H")
    """

if __name__ == '__main__':
    testRefine()