# Coordinate Conventions

This project uses column vectors and homogeneous transforms that left-multiply
points:

```text
p_a = R_ab p_b + t_ab
P_a = T_ab P_b
```

`T_ab` maps coordinates from frame `b` into frame `a`. The first subscript is
the destination frame; the second subscript is the source frame.

## World And Camera Poses

- `T_wc` maps camera-frame coordinates into world-frame coordinates.
- `T_cw` maps world-frame coordinates into camera-frame coordinates.
- `T_cw = inverse(T_wc)`.

For a world point `P_w`, projection into a camera uses:

```text
P_c = T_cw P_w
```

## Two-View Relative Poses

Camera indices use zero-based names in code and examples.

- `T_10` maps points from camera 0 into camera 1.
- `T_01` maps points from camera 1 into camera 0.
- `T_01 = inverse(T_10)`.

OpenCV's `recoverPose` result is interpreted as `T_10`:

```text
p_1 = R_10 p_0 + t_10
```

The translation from essential matrix decomposition is unit length up to an
unknown scene scale. Triangulation examples that need metric coordinates must
use a scaled or externally known baseline.

## Array Shapes

Public Python APIs prefer row-major point arrays:

- 2D image points: `Nx2`
- 3D points: `Nx3`
- Rotations: `3x3`
- Translations: `3` or `3x1`, normalized internally to `3x1`
- Homogeneous transforms: `4x4`

OpenCV calls that require `2xN`, `3xN`, or `4xN` arrays should convert at the
boundary and return to the public `Nx*` convention.
