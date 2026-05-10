Python implementations of selected projects and examples from
[slambook](https://github.com/gaoxiang12/slambook).

This repository contains a teaching-first Python package for the core slambook
concepts. The legacy root scripts remain in place while the migrated importable
modules and examples live under `slam/` and `examples/`.

## Install

Core educational dependencies:

```bash
pip install -e .[core]
```

Core dependencies plus tests:

```bash
pip install -e .[core,test]
python -m pytest
```

With `uv`, use the declared extras directly:

```bash
uv sync --extra core --extra test
uv run --extra core --extra test python -m pytest
```

In mainland China, a PyPI mirror can be used with the checked-in lockfile:

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --extra core --extra test --frozen
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv run --extra core --extra test --frozen python -m pytest
```

Optional dependency groups are defined for 3D/evaluation tools, modern matchers,
and reference backends:

```bash
pip install -e .[3d]
pip install -e .[modern]
pip install -e .[backend]
pip install -e .[all]
```

Importing `slam` does not require optional modern backends.

## Migration Status

See `docs/status.md` for the chapter-by-chapter status table.
See `docs/datasets.md` for expected local dataset layouts.
See `docs/validation.md` for the latest local validation notes.

Representative migrated examples:

```bash
python examples/ch7_feature_vo/pose_estimation_2d2d.py \
  --image0 data/slambook/ch7/1.png \
  --image1 data/slambook/ch7/2.png \
  --matcher orb

python examples/ch10_bundle_adjustment/scipy_bal.py \
  --bal examples/ch10_bundle_adjustment/tiny_bal.txt \
  --fix-cameras

python examples/ch13_dense_mapping/rgbd_fusion.py \
  --color-dir data/slambook/ch13/color \
  --depth-dir data/slambook/ch13/depth \
  --pose-file data/slambook/ch13/pose.txt \
  --intrinsics FX FY CX CY \
  --output outputs/ch13_cloud.ply
```

## Legacy Scripts

The original root-level scripts are kept under `legacy/` as the historical
baseline during the migration:

- `legacy/pose_estimation_2d2d.py`
- `legacy/pose_estimation_3d2d.py`
- `legacy/flann_based_matcher.py`
- `legacy/ransac_test.py`
- `legacy/simStereoCamera.py`
- `legacy/testRefine.py`
- `legacy/utils.py`

New code should use importable modules under `slam/` and runnable examples under
`examples/`.


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
