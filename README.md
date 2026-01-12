# Neural Triangle Culling (BVH Traversal) — NAS + Dataset Pipeline

This repo is a research pipeline for **testing whether tiny neural networks can “cull” ray–triangle intersection tests** during BVH traversal by predicting *misses* cheaply enough to matter.

It includes:
- **Dataset generation** from BVH leaf-sample logs → `triangles.bin` (a compact binary dataset)
- **Optional canonicalization** (pre-rotation) to reduce rotational degrees of freedom
- **Dataset balancing + dedup** tools
- **A small architecture search space** generator (FLOP-limited MLPs)
- **An exhaustive NAS loop** (architectures × feature subsets) with a strict compute-cost constraint and clean VAL→TEST separation

If you want the experimental narrative/results context, read **`Notes 2.txt`**.

---

## Project layout

- `bvh_to_binary.py`  
  Parses BVH leaf-sample text logs and writes `triangles.bin` (57 float features + label).

- `bvh_to_binary_prerotate.py`  
  Same as above, but applies a **coordinate system normalization**:
  - translate ray origin to (0,0,0)
  - rotate coordinates so the BVH face center (orthogonal to Z) lies on the Z-axis  
  This canonicalizes orientation while keeping ray-direction information.

- `datasetcleaner.py`  
  Rebalances a dataset to a target hit/miss ratio (default **40% hits / 60% misses**) and can deduplicate.

- `architecturesearch.cpp`  
  Enumerates FLOP-limited MLP architectures and outputs `nn_flops_tables.txt` (usable as `nas.txt`).

- `main2.py` (**recommended**)  
  Runs the exhaustive NAS experiment, saves graphs + top checkpoints, writes `nas_exhaustive_report.txt`.

- `main.py`  
  Older/alternate version of the training + search script.

- `test.py`  
  Quick sanity check / analysis plots for a `triangles.bin` dataset.

- `nas.txt`  
  Example architecture list (parsed by `main*.py`).

- `nas_exhaustive_report.txt`  
  Example output report file.

---

## Binary dataset format (`triangles.bin`)

Each record is **232 bytes**:

- `57 × float32` features (little-endian)
- `1 × uint8` label (`1` = hit, `0` = miss)
- `3 × pad bytes` (zeros)

So each sample is:
57f + B + 3x == 232 bytes


The training scripts load this using a NumPy structured dtype and expect the file size to be a multiple of 232.

---

## Features (the 57-float base vector)

`bvh_to_binary*.py` builds a 57D base feature vector per (ray, triangle) sample. The layout is documented in the code; highlights include:
- ray direction, triangle vertices
- triangle area, unit normal, aspect ratio, orientation sign
- centroid / orthocenter / Fermat point / circumcenter + circumradius / incenter + inradius
- AABB (6)
- OBB (10)

> Note: some indices are intentionally placeholders/unmapped (kept as zeros) to preserve an expected layout.

`main*.py` then expands this to **64D** by adding derived features:
- `ray_normal_dot`
- MT-ish derived scalars: `determinant`, `u_raw`, `v_raw`, and `dot(ray_dir, v0/v1/v2)`
  - `u_raw` / `v_raw` exist but are not selectable in the feature search space.

---

## Installation

### Python
You need:
- Python 3.9+ recommended
- `numpy`, `torch`, `scikit-learn`, `matplotlib`

Example:
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install numpy torch scikit-learn matplotlib
