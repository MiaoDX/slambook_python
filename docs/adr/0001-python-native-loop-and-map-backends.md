# ADR 0001: Use Python-Native Loop Closure And Mapping Artifacts

Date: 2026-05-10

## Status

Accepted

## Context

Upstream `slambook` uses DBoW3 for loop closure vocabulary/scoring and Octomap
for occupancy mapping. Exact binary or API compatibility would require adding
C++-native dependencies that are harder to install and less useful for the
teaching-first Python baseline.

The Python port already provides:

- OpenCV local descriptors.
- `VisualVocabulary` and BoW histograms for loop closure.
- NumPy and FAISS nearest-neighbor retrieval.
- RGB-D point clouds, known-pose fusion, voxel downsampling, normals, and
  occupancy voxel-grid export.

## Decision

Do not require exact DBoW3 or Octomap compatibility for the baseline Python
port. Keep the maintained artifacts Python-native:

- BoW vocabulary: compressed NumPy `.npz`.
- Image descriptors: NumPy `.npy`.
- Point clouds: ASCII PLY.
- Occupancy grid: compressed NumPy `.npz` with voxel size, voxel indices,
  centers, and counts.

DBoW3 and Octomap may be added later only as optional interoperability tools,
not as required dependencies and not as replacements for the Python-native
teaching path.

## Consequences

- Core examples remain easy to install and test with `uv sync --extra core
  --extra test`.
- Loop closure and occupancy mapping remain inspectable with standard Python
  tooling.
- Users needing exact DBoW3 vocabulary files or Octomap `.bt` files must use
  upstream tools or add a separate optional conversion path.
