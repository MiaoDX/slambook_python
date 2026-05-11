# Implementation Notes

These notes preserve the practical lessons from the legacy scripts in a more
structured form. They are mostly about OpenCV and NumPy behavior that can change
SLAM results in subtle ways.

## Image Loading And Color

OpenCV defaults to BGR color when loading images:

```python
color_bgr = cv2.imread(image_path)
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
```

Feature extraction, photometric residuals, visualization, and saved outputs can
all change when an example silently switches between BGR, RGB, and grayscale.
Examples should choose the intended mode explicitly and convert color spaces
before plotting or comparing against RGB libraries.

## Inlier Masks

OpenCV functions such as `findFundamentalMat`, `findEssentialMat`, and
`recoverPose` return masks for inlier correspondences. The mask values may be
`1` or `255` depending on the function and OpenCV version, so normalize masks
before indexing:

```python
inliers = mask.ravel() != 0
```

The number of inliers can be smaller than the number of input matches. Downstream
code should apply the mask consistently to both image-point arrays.

## Point Types And Shapes

Point dtype matters for epipolar geometry. Integer conversion can change numeric
results:

```python
pts = np.asarray(pts, dtype=np.float32)
```

Use `Nx2` for image points and `Nx3` for 3D points at public Python boundaries.
Convert only at API edges when a library expects a transposed `3xN` or `4xN`
layout.

## Matrix Multiplication

The common C++ OpenCV and Python NumPy equivalents are:

```text
C++              Python
*                @ or np.dot(...)
a * b.t()        a @ b.T
a.dot(b)         (a * b).sum()
a.mul(b)         a * b
```

For example:

```python
E = K2.T @ F @ K1
```

Reference: [Opencv中Mat矩阵相乘——点乘、dot、mul运算详解](http://blog.csdn.net/dcrmg/article/details/52404580).

## Epipolar Geometry

Keep the relationship between the fundamental matrix `F`, essential matrix `E`,
and homography `H` explicit in examples. A calibrated pose pipeline should use
the calibrated intrinsics to move from `F` to `E`, while planar-scene examples
may need `H` instead.

## Pose Recovery And Cheirality

`recoverPose` decomposes an essential matrix and selects the pose using a
cheirality check: triangulated points should have positive depth in front of the
cameras. A hand-written `check_solutions(...)` helper should be expected to
match that behavior only when it uses the same inlier set, point normalization,
and positive-depth convention.

Reference: [OpenCV recoverPose documentation](http://docs.opencv.org/3.2.0/d9/d0c/group__calib3d.html).

## Useful References

- [import cv2 notes](https://pythonpath.wordpress.com/import-cv2/)
- [相机位姿求解问题？](https://www.zhihu.com/question/51510464)
