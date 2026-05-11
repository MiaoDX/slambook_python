# Chapter 12 Loop Closure

Build simple mean-pooled OpenCV descriptors:

```bash
uv run --frozen python examples/ch12_loop_closure/build_descriptors.py \
  --images data/slambook/ch12 \
  --output outputs/ch12_descriptors.npy
```

Train a small bag-of-visual-words vocabulary and encode BoW histograms:

```bash
uv run --frozen python examples/ch12_loop_closure/train_vocabulary.py \
  --images data/slambook/ch12 \
  --words 64 \
  --output outputs/ch12_vocabulary.npz

uv run --frozen python examples/ch12_loop_closure/build_descriptors.py \
  --images data/slambook/ch12 \
  --vocabulary outputs/ch12_vocabulary.npz \
  --output outputs/ch12_bow.npy
```

Retrieve loop candidates while excluding immediate temporal neighbors:

```bash
uv run --frozen python examples/ch12_loop_closure/retrieve_candidates.py \
  --descriptors outputs/ch12_bow.npy \
  --current-index 8 \
  --temporal-window 2
```
