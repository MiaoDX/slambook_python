Python implements of some of the projects/files in [slambook](https://github.com/gaoxiang12/slambook).


## SOME NOTES:

* image color and corresponding difference, `im1 = cv2.imread(im1_file)` and `im1 = cv2.imread(im1_file, 0)`. Or the color space (RGB, BGR) and the consequences.

* inner points in findFundamentalMat, when it is not the same as the input num?

* when we calc Fundamental Matrix

`pts1 = np.int32(pts1)` and `pts1 = np.array(pts1)` will lead to very different answers, so, which is better??

Use `np.array(pts1)` or `np.float32(pts1)`


* Matrix manipulation in C++ and python

[Opencv中Mat矩阵相乘——点乘、dot、mul运算详解](http://blog.csdn.net/dcrmg/article/details/52404580

```        
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
```

* The corresponding relationship between F,E and H

* [recoverPose](http://docs.opencv.org/3.2.0/d9/d0c/group__calib3d.html)

``` vi
    This function decomposes an essential matrix using decomposeEssentialMat and then verifies possible pose hypotheses by doing cheirality check. The cheirality check basically means that the triangulated 3D points should have positive depth. Some details can be found in [119] .
```

* check_solutions(fp, sp, K, R1, R2, t):
    should have same result as recoverPose, but without good luck    
    
* some notes online
  
    - [import cv2 #Notes](https://pythonpath.wordpress.com/import-cv2/)
    
* The usage of 3xN and Nx3

* findFundamentalMat, findEssentialMat all have `mask` return value and means `inner points`, we should take care of them

* [相机位姿求解问题？](https://www.zhihu.com/question/51510464)

