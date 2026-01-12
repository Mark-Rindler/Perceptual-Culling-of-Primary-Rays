#usage: 

# Basic usage - random sample 10,000 triangles
#python bvh_to_binary.py tris.txt -o triangles.bin -n 10000

# Use all triangles (no random sampling)
#python bvh_to_binary.py tris.txt -o triangles.bin --all

# Reproducible sampling with seed
#python bvh_to_binary.py tris.txt -o triangles.bin -n 50000 --seed 42

# Sample multiple leaves/triangles per hierarchy level
#python bvh_to_binary.py tris.txt -o triangles.bin -n 5000 --leaves-per-sample 3 --tris-per-leaf 2=

#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Parses BVH leaf sample text files and converts to the binary format
expected by the NAS triangle culling script.

Binary format per sample: 57 float32 + 1 uint8 label + 3 pad bytes = 232 bytes

COORDINATE SYSTEM NORMALIZATION:
- Ray origin is translated to (0, 0, 0)
- Coordinate system is rotated so the center of the BVH face (orthogonal to Z-axis)
  lies on the Z-axis at (0, 0, z) for some z
- Ray direction is rotated accordingly (NOT forced to any particular direction)
- This canonicalizes the BVH position while preserving ray direction information
"""

import re
import numpy as np
import random
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set


@dataclass
class Triangle:
    face_id: int
    indices: Tuple[int, int, int]
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    # Unique identifier for deduplication: (sample_idx, leaf_idx, tri_idx)
    unique_id: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class Leaf:
    node_id: int
    prim_start: int
    prim_count: int
    tri_offset: int
    tri_count: int
    ray_origin: np.ndarray
    ray_dir: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    triangles: List[Triangle] = field(default_factory=list)


@dataclass
class LeafSample:
    header_count: int
    tri_count: int
    leaves: List[Leaf] = field(default_factory=list)


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------

def parse_vec3(s: str) -> np.ndarray:
    """Parse a string like '(-0.676949,-0.319162,-2.57479)' into a numpy array."""
    s = s.strip().strip('()')
    parts = s.split(',')
    return np.array([float(p) for p in parts], dtype=np.float32)


def parse_indices(s: str) -> Tuple[int, int, int]:
    """Parse a string like '(27507,27257,27623)' into a tuple of ints."""
    s = s.strip().strip('()')
    parts = s.split(',')
    return tuple(int(p) for p in parts)


def parse_bvh_file(filepath: str) -> List[LeafSample]:
    """
    Parse the BVH leaf sample text file.

    Expected format:
    ==== BVH LEAF SAMPLE (headers=N, tris=M) ====
    LEAF X node=... primStart=... primCount=... triOffset=... triCount=...
      ray origin=(...) dir=(...)
      boundsLocal bmin=(...) bmax=(...)
        TRI face=... idx=(...)
          v0=(...) v1=(...) v2=(...)
        ...
    """
    print(f"Parsing BVH file: {filepath}")

    samples = []
    current_sample = None
    current_leaf = None
    current_tri_data = {}
    sample_idx = -1
    leaf_idx = -1
    tri_idx = -1

    # Regex patterns - allow leading whitespace
    sample_header_re = re.compile(r'\s*={4}\s*BVH LEAF SAMPLE \(headers=(\d+), tris=(\d+)\)\s*={4}')
    leaf_header_re = re.compile(
        r'\s*LEAF\s+(\d+)\s+node=(\d+)\s+primStart=(\d+)\s+primCount=(\d+)\s+'
        r'triOffset=(\d+)\s+triCount=(\d+)'
    )
    ray_re = re.compile(r'ray origin=(\([^)]+\))\s+dir=(\([^)]+\))')
    bounds_re = re.compile(r'boundsLocal bmin=(\([^)]+\))\s+bmax=(\([^)]+\))')
    tri_header_re = re.compile(r'TRI face=(\d+)\s+idx=(\([^)]+\))')
    vertex_re = re.compile(r'v0=(\([^)]+\))\s+v1=(\([^)]+\))\s+v2=(\([^)]+\))')

    def finalize_triangle(line_num: int) -> bool:
        """Finalize current triangle, return True if successful."""
        nonlocal current_tri_data, tri_idx
        if not current_tri_data:
            return True

        if current_tri_data.get('v0') is None or \
           current_tri_data.get('v1') is None or \
           current_tri_data.get('v2') is None:
            print(f"Warning: Triangle at line ~{line_num} has missing vertices, skipping. "
                  f"face_id={current_tri_data.get('face_id')}", file=sys.stderr)
            current_tri_data = {}
            return False

        current_tri_data['unique_id'] = (sample_idx, leaf_idx, tri_idx)
        current_leaf.triangles.append(Triangle(**current_tri_data))
        current_tri_data = {}
        return True

    def finalize_leaf(line_num: int):
        """Finalize current leaf."""
        nonlocal current_leaf, leaf_idx
        if current_leaf is None:
            return

        finalize_triangle(line_num)

        # Validate triangle count
        expected = current_leaf.tri_count
        actual = len(current_leaf.triangles)
        if expected != actual:
            print(f"Warning: Leaf {current_leaf.node_id} expected {expected} triangles, "
                  f"parsed {actual}", file=sys.stderr)

        current_sample.leaves.append(current_leaf)
        current_leaf = None

    def finalize_sample(line_num: int):
        """Finalize current sample."""
        nonlocal current_sample
        if current_sample is None:
            return

        finalize_leaf(line_num)

        # Validate total triangle count
        expected = current_sample.tri_count
        actual = sum(len(leaf.triangles) for leaf in current_sample.leaves)
        if expected != actual:
            print(f"Warning: Sample expected {expected} total triangles, "
                  f"parsed {actual}", file=sys.stderr)

        # Validate header count (number of leaves)
        expected_leaves = current_sample.header_count
        actual_leaves = len(current_sample.leaves)
        if expected_leaves != actual_leaves:
            print(f"Warning: Sample expected {expected_leaves} leaves, "
                  f"parsed {actual_leaves}", file=sys.stderr)

        samples.append(current_sample)
        current_sample = None

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line_stripped = line.strip()

            # Check for sample header
            m = sample_header_re.match(line_stripped)
            if m:
                finalize_sample(line_num)

                sample_idx += 1
                leaf_idx = -1
                tri_idx = -1

                current_sample = LeafSample(
                    header_count=int(m.group(1)),
                    tri_count=int(m.group(2))
                )
                current_leaf = None
                current_tri_data = {}
                continue

            if current_sample is None:
                continue

            # Check for leaf header
            m = leaf_header_re.match(line_stripped)
            if m:
                finalize_leaf(line_num)

                leaf_idx += 1
                tri_idx = -1

                current_leaf = Leaf(
                    node_id=int(m.group(2)),
                    prim_start=int(m.group(3)),
                    prim_count=int(m.group(4)),
                    tri_offset=int(m.group(5)),
                    tri_count=int(m.group(6)),
                    ray_origin=np.zeros(3, dtype=np.float32),
                    ray_dir=np.zeros(3, dtype=np.float32),
                    bounds_min=np.zeros(3, dtype=np.float32),
                    bounds_max=np.zeros(3, dtype=np.float32),
                )
                continue

            if current_leaf is None:
                continue

            # Check for ray info
            m = ray_re.search(line_stripped)
            if m:
                current_leaf.ray_origin = parse_vec3(m.group(1))
                current_leaf.ray_dir = parse_vec3(m.group(2))
                continue

            # Check for bounds info
            m = bounds_re.search(line_stripped)
            if m:
                current_leaf.bounds_min = parse_vec3(m.group(1))
                current_leaf.bounds_max = parse_vec3(m.group(2))
                continue

            # Check for triangle header
            m = tri_header_re.search(line_stripped)
            if m:
                finalize_triangle(line_num)

                tri_idx += 1

                current_tri_data = {
                    'face_id': int(m.group(1)),
                    'indices': parse_indices(m.group(2)),
                    'v0': None,
                    'v1': None,
                    'v2': None,
                }
                continue

            # Check for vertex data
            m = vertex_re.search(line_stripped)
            if m and current_tri_data:
                current_tri_data['v0'] = parse_vec3(m.group(1))
                current_tri_data['v1'] = parse_vec3(m.group(2))
                current_tri_data['v2'] = parse_vec3(m.group(3))
                continue

    # Finalize last sample
    finalize_sample(line_num)

    # Filter out empty samples
    samples = [s for s in samples if s.leaves and any(leaf.triangles for leaf in s.leaves)]

    total_leaves = sum(len(s.leaves) for s in samples)
    total_tris = sum(len(leaf.triangles) for s in samples for leaf in s.leaves)
    print(f"Parsed {len(samples)} leaf samples, {total_leaves} leaves, {total_tris} triangles")

    return samples


# ---------------------------------------------------------------------
# Geometric computations
# ---------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector, returning zero vector if magnitude is too small."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.zeros_like(v)
    return v / norm


def compute_rotation_point_to_z_axis(point: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix that moves a point onto the Z-axis.
    Returns a 3x3 rotation matrix R such that R @ point = (0, 0, ||point||).
    
    This rotates the coordinate system so the given point lies on the Z-axis,
    canonicalizing the BVH's position relative to the ray origin.
    """
    p = point.astype(np.float64)
    p_norm = np.linalg.norm(p)
    
    if p_norm < 1e-12:
        # Point is at origin, no rotation needed
        return np.eye(3, dtype=np.float32)
    
    # Normalize to get direction from origin to point
    p_dir = p / p_norm
    
    # We want to rotate p_dir to align with +Z or -Z (preserve the sign of z)
    # Use +Z if point has positive z component, -Z otherwise
    if p_dir[2] >= 0:
        target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    
    dot = np.dot(p_dir, target)
    
    if dot > 0.999999:
        # Already aligned
        return np.eye(3, dtype=np.float32)
    
    if dot < -0.999999:
        # Opposite direction, rotate 180° around X axis
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=np.float32)
    
    # General case: use Rodrigues' rotation formula
    # Axis of rotation is p_dir × target
    axis = np.cross(p_dir, target)
    axis = axis / np.linalg.norm(axis)
    
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    
    # Skew-symmetric cross-product matrix of axis
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ], dtype=np.float64)
    
    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    R = np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    
    return R.astype(np.float32)


def ray_triangle_intersect(ray_origin: np.ndarray, ray_dir: np.ndarray,
                           v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                           epsilon: float = 1e-7) -> bool:
    """
    Möller–Trumbore ray-triangle intersection.
    Returns True if ray hits the triangle (t >= 0).
    """
    e1 = v1 - v0
    e2 = v2 - v0

    pvec = np.cross(ray_dir, e2)
    det = np.dot(e1, pvec)

    if abs(det) < epsilon:
        return False  # Ray parallel to triangle

    inv_det = 1.0 / det
    tvec = ray_origin - v0
    u = np.dot(tvec, pvec) * inv_det

    if u < 0.0 or u > 1.0:
        return False

    qvec = np.cross(tvec, e1)
    v = np.dot(ray_dir, qvec) * inv_det

    if v < 0.0 or u + v > 1.0:
        return False

    t = np.dot(e2, qvec) * inv_det

    return t >= 0.0  # Hit if t is non-negative


def compute_triangle_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute triangle area using cross product."""
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    return 0.5 * np.linalg.norm(cross)


def compute_triangle_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute unit normal of triangle."""
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    return normalize(cross)


def compute_centroid(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute triangle centroid."""
    return (v0 + v1 + v2) / 3.0


def compute_circumcenter(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute circumcenter and circumradius of triangle.
    Uses barycentric coordinates.
    """
    a = v1 - v2
    b = v2 - v0
    c = v0 - v1

    a_len_sq = np.dot(a, a)
    b_len_sq = np.dot(b, b)
    c_len_sq = np.dot(c, c)

    # Barycentric coordinates for circumcenter
    alpha = a_len_sq * np.dot(-b, c)
    beta = b_len_sq * np.dot(-c, a)
    gamma = c_len_sq * np.dot(-a, b)

    denom = alpha + beta + gamma
    if abs(denom) < 1e-12:
        # Degenerate triangle
        centroid = compute_centroid(v0, v1, v2)
        return centroid, 0.0

    alpha /= denom
    beta /= denom
    gamma /= denom

    circumcenter = alpha * v0 + beta * v1 + gamma * v2
    circumradius = np.linalg.norm(v0 - circumcenter)

    return circumcenter.astype(np.float32), float(circumradius)


def compute_incenter(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute incenter and inradius of triangle."""
    a = np.linalg.norm(v1 - v2)  # side opposite v0
    b = np.linalg.norm(v2 - v0)  # side opposite v1
    c = np.linalg.norm(v0 - v1)  # side opposite v2

    perimeter = a + b + c
    if perimeter < 1e-12:
        centroid = compute_centroid(v0, v1, v2)
        return centroid, 0.0

    incenter = (a * v0 + b * v1 + c * v2) / perimeter

    # Inradius = Area / semi-perimeter
    area = compute_triangle_area(v0, v1, v2)
    inradius = 2.0 * area / perimeter

    return incenter.astype(np.float32), float(inradius)


def compute_orthocenter(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute orthocenter of triangle: H = v0 + v1 + v2 - 2 * circumcenter."""
    circumcenter, _ = compute_circumcenter(v0, v1, v2)
    orthocenter = v0 + v1 + v2 - 2.0 * circumcenter
    return orthocenter.astype(np.float32)


def compute_fermat_point(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute Fermat point (minimizes sum of distances to vertices).
    Uses iterative Weiszfeld algorithm.
    """
    point = compute_centroid(v0, v1, v2).astype(np.float64)
    vertices = [v0.astype(np.float64), v1.astype(np.float64), v2.astype(np.float64)]

    for _ in range(50):
        weights = []
        weighted_sum = np.zeros(3, dtype=np.float64)

        for v in vertices:
            dist = np.linalg.norm(point - v)
            if dist < 1e-10:
                return v.astype(np.float32)
            w = 1.0 / dist
            weights.append(w)
            weighted_sum += w * v

        total_weight = sum(weights)
        new_point = weighted_sum / total_weight

        if np.linalg.norm(new_point - point) < 1e-8:
            break
        point = new_point

    return point.astype(np.float32)


def compute_aspect_ratio(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute aspect ratio as longest_edge / shortest_altitude."""
    edges = [
        np.linalg.norm(v1 - v0),
        np.linalg.norm(v2 - v1),
        np.linalg.norm(v0 - v2),
    ]

    longest_edge = max(edges)
    area = compute_triangle_area(v0, v1, v2)

    if longest_edge < 1e-12:
        return 1.0

    shortest_altitude = 2.0 * area / longest_edge

    if shortest_altitude < 1e-12:
        return 1e6  # Degenerate

    return longest_edge / shortest_altitude


def compute_orientation_sign(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                             ray_dir: np.ndarray) -> float:
    """
    Compute orientation sign based on whether triangle faces toward or away from ray.
    Returns +1.0 if triangle normal points against ray direction (front-facing).
    Returns -1.0 if triangle normal points with ray direction (back-facing).
    """
    normal = compute_triangle_normal(v0, v1, v2)
    dot = np.dot(normal, ray_dir)
    # Negative dot product means normal opposes ray direction (front-facing)
    return 1.0 if dot <= 0 else -1.0


def compute_aabb(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute axis-aligned bounding box: [min_x, min_y, min_z, max_x, max_y, max_z]."""
    vertices = np.stack([v0, v1, v2])
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return np.concatenate([mins, maxs]).astype(np.float32)


def compute_obb(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute oriented bounding box representation.
    Returns 10 floats: center(3) + half_extents(3) + rotation_quaternion(4)
    """
    center = compute_centroid(v0, v1, v2)

    e1 = v1 - v0
    e2 = v2 - v0
    normal = np.cross(e1, e2)

    e1_norm = np.linalg.norm(e1)
    normal_norm = np.linalg.norm(normal)

    if e1_norm < 1e-12 or normal_norm < 1e-12:
        return np.array([
            center[0], center[1], center[2],
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

    axis_x = e1 / e1_norm
    axis_z = normal / normal_norm
    axis_y = np.cross(axis_z, axis_x)

    vertices = np.stack([v0 - center, v1 - center, v2 - center])
    local_coords = np.zeros((3, 3))
    for i, v in enumerate(vertices):
        local_coords[i, 0] = np.dot(v, axis_x)
        local_coords[i, 1] = np.dot(v, axis_y)
        local_coords[i, 2] = np.dot(v, axis_z)

    half_extents = np.abs(local_coords).max(axis=0)

    R = np.column_stack([axis_x, axis_y, axis_z])
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([
        center[0], center[1], center[2],
        half_extents[0], half_extents[1], half_extents[2],
        w, x, y, z
    ], dtype=np.float32)


def build_feature_vector(ray_dir: np.ndarray, v0: np.ndarray, v1: np.ndarray,
                         v2: np.ndarray) -> np.ndarray:
    """
    Build the 57-element feature vector matching the NAS script's expected format.

    NOTE: After coordinate normalization:
    - Ray origin is at (0, 0, 0)
    - BVH face center is on the Z-axis at (0, 0, z)
    - Ray direction varies (rotated along with geometry)

    Layout (indices):
    [0-2]:   ray_dir (3)
    [3-5]:   v0 (3)
    [6-8]:   v1 (3)
    [9-11]:  v2 (3)
    [12]:    placeholder (unmapped in original)
    [13]:    area (1)
    [14-16]: unit_normal (3)
    [17-20]: placeholder (unmapped in original) (4)
    [21]:    orientation_sign (1)
    [22]:    aspect_ratio (1)
    [23]:    placeholder (unmapped in original) (1)
    [24-26]: centroid (3)
    [27-29]: orthocenter (3)
    [30-32]: fermat_point (3)
    [33-35]: circumcenter (3)
    [36]:    circumradius (1)
    [37-39]: incenter (3)
    [40]:    inradius (1)
    [41-46]: aabb (6)
    [47-56]: obb (10)

    Total: 57
    """
    features = np.zeros(57, dtype=np.float32)

    features[0:3] = ray_dir
    features[3:6] = v0
    features[6:9] = v1
    features[9:12] = v2
    features[12] = 0.0
    features[13] = compute_triangle_area(v0, v1, v2)
    features[14:17] = compute_triangle_normal(v0, v1, v2)
    features[17:21] = 0.0
    features[21] = compute_orientation_sign(v0, v1, v2, ray_dir)
    features[22] = compute_aspect_ratio(v0, v1, v2)
    features[23] = 0.0
    features[24:27] = compute_centroid(v0, v1, v2)
    features[27:30] = compute_orthocenter(v0, v1, v2)
    features[30:33] = compute_fermat_point(v0, v1, v2)

    circumcenter, circumradius = compute_circumcenter(v0, v1, v2)
    features[33:36] = circumcenter
    features[36] = circumradius

    incenter, inradius = compute_incenter(v0, v1, v2)
    features[37:40] = incenter
    features[40] = inradius

    features[41:47] = compute_aabb(v0, v1, v2)
    features[47:57] = compute_obb(v0, v1, v2)

    return features


# ---------------------------------------------------------------------
# Sampling and output
# ---------------------------------------------------------------------

def build_triangle_index(samples: List[LeafSample]) -> List[Tuple[int, int, int, Leaf, Triangle]]:
    """
    Build a flat index of all triangles with their hierarchy info.
    Returns list of (sample_idx, leaf_idx, tri_idx, leaf, triangle).
    """
    index = []
    for s_idx, sample in enumerate(samples):
        for l_idx, leaf in enumerate(sample.leaves):
            for t_idx, tri in enumerate(leaf.triangles):
                index.append((s_idx, l_idx, t_idx, leaf, tri))
    return index


def sample_triangles_no_duplicates(
    samples: List[LeafSample],
    n_triangles: int,
    seed: Optional[int] = None,
    max_retries_per_sample: int = 1000
) -> List[Tuple[np.ndarray, np.ndarray, Leaf, Triangle]]:
    """
    Sample exactly n_triangles unique triangles using hierarchical selection.

    Selection algorithm:
    1. Randomly select a leaf sample
    2. Randomly select a leaf from that sample
    3. Randomly select a triangle from that leaf
    4. If triangle already selected, restart from step 1 (new sample)
    5. Repeat until n_triangles unique triangles collected

    Returns list of (ray_origin, ray_dir, leaf, triangle).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Build index of valid samples (those with triangles)
    valid_samples = []
    for s_idx, sample in enumerate(samples):
        valid_leaves = [(l_idx, leaf) for l_idx, leaf in enumerate(sample.leaves)
                        if leaf.triangles]
        if valid_leaves:
            valid_samples.append((s_idx, sample, valid_leaves))

    if not valid_samples:
        print("Error: No valid samples with triangles found.", file=sys.stderr)
        return []

    # Count total available triangles
    total_available = sum(
        len(leaf.triangles)
        for _, sample, valid_leaves in valid_samples
        for _, leaf in valid_leaves
    )

    if n_triangles > total_available:
        print(f"Warning: Requested {n_triangles} triangles but only {total_available} "
              f"available. Will return all unique triangles.", file=sys.stderr)
        n_triangles = total_available

    selected_ids: Set[Tuple[int, int, int]] = set()
    results: List[Tuple[np.ndarray, np.ndarray, Leaf, Triangle]] = []
    total_retries = 0

    while len(results) < n_triangles:
        retries = 0

        while retries < max_retries_per_sample:
            # Step 1: Random sample
            s_idx, sample, valid_leaves = random.choice(valid_samples)

            # Step 2: Random leaf from sample
            l_idx, leaf = random.choice(valid_leaves)

            # Step 3: Random triangle from leaf
            t_idx = random.randrange(len(leaf.triangles))
            tri = leaf.triangles[t_idx]

            # Step 4: Check for duplicate
            unique_id = (s_idx, l_idx, t_idx)
            if unique_id not in selected_ids:
                selected_ids.add(unique_id)
                results.append((
                    leaf.ray_origin.copy(),
                    leaf.ray_dir.copy(),
                    leaf,
                    tri
                ))
                break

            # Duplicate found - restart from step 1 (new sample)
            retries += 1
            total_retries += 1

        if retries >= max_retries_per_sample:
            # Extremely unlikely unless we're near saturation
            print(f"Warning: Hit max retries ({max_retries_per_sample}) at "
                  f"{len(results)}/{n_triangles} samples. Pool may be nearly exhausted.",
                  file=sys.stderr)

            # Fallback: directly sample from remaining triangles
            remaining = []
            for vs_idx, _, valid_leaves in valid_samples:
                for vl_idx, leaf in valid_leaves:
                    for vt_idx, tri in enumerate(leaf.triangles):
                        uid = (vs_idx, vl_idx, vt_idx)
                        if uid not in selected_ids:
                            remaining.append((uid, leaf, tri))

            if not remaining:
                print(f"Error: No more unique triangles available.", file=sys.stderr)
                break

            uid, leaf, tri = random.choice(remaining)
            selected_ids.add(uid)
            results.append((leaf.ray_origin.copy(), leaf.ray_dir.copy(), leaf, tri))

    if total_retries > 0:
        print(f"Note: {total_retries} total retries due to duplicate selections")

    return results


def transform_to_canonical_coordinates(ray_origin: np.ndarray, ray_dir: np.ndarray,
                                        bounds_min: np.ndarray, bounds_max: np.ndarray,
                                        v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
                                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform triangle vertices to canonical coordinate system:
    1. Translate so ray origin is at (0, 0, 0)
    2. Rotate so the center of the BVH face (orthogonal to Z-axis) lies on the Z-axis at (0, 0, z)
    
    The BVH face used is determined by the ray direction:
    - If ray goes in +Z direction, use the -Z face (entry face)
    - If ray goes in -Z direction, use the +Z face (entry face)
    
    Returns: (ray_dir_transformed, v0_transformed, v1_transformed, v2_transformed)
    """
    # Step 1: Center at ray origin
    v0_centered = v0 - ray_origin
    v1_centered = v1 - ray_origin
    v2_centered = v2 - ray_origin
    bounds_min_centered = bounds_min - ray_origin
    bounds_max_centered = bounds_max - ray_origin
    
    # Step 2: Compute the center of the BVH face orthogonal to Z-axis
    # The face center has x,y at the center of the box, z at either min or max
    face_center_x = (bounds_min_centered[0] + bounds_max_centered[0]) / 2.0
    face_center_y = (bounds_min_centered[1] + bounds_max_centered[1]) / 2.0
    
    # Choose which Z-face based on ray direction (use the entry face)
    if ray_dir[2] >= 0:
        # Ray going in +Z direction, enters through -Z face (min z)
        face_center_z = bounds_min_centered[2]
    else:
        # Ray going in -Z direction, enters through +Z face (max z)
        face_center_z = bounds_max_centered[2]
    
    face_center = np.array([face_center_x, face_center_y, face_center_z], dtype=np.float32)
    
    # Step 3: Compute rotation to move face_center onto the Z-axis
    R = compute_rotation_point_to_z_axis(face_center)
    
    # Step 4: Apply rotation to vertices and ray direction
    v0_rotated = (R @ v0_centered).astype(np.float32)
    v1_rotated = (R @ v1_centered).astype(np.float32)
    v2_rotated = (R @ v2_centered).astype(np.float32)
    ray_dir_rotated = (R @ ray_dir).astype(np.float32)
    
    return ray_dir_rotated, v0_rotated, v1_rotated, v2_rotated


def process_and_write_binary(samples: List[LeafSample],
                             output_path: str,
                             n_output_triangles: int,
                             seed: Optional[int] = None):
    """
    Sample triangles and write to binary format.
    
    Applies coordinate system normalization:
    - Ray origin at (0, 0, 0)
    - BVH face center (orthogonal to Z) on the Z-axis at (0, 0, z)
    """
    print(f"\nSampling {n_output_triangles} unique triangles...")

    sampled = sample_triangles_no_duplicates(
        samples,
        n_triangles=n_output_triangles,
        seed=seed
    )

    if not sampled:
        print("Error: No samples generated.", file=sys.stderr)
        return

    print(f"Processing {len(sampled)} triangle samples...")
    print("Applying coordinate normalization: ray origin at (0,0,0), BVH face center on Z-axis")

    X_all = []
    y_all = []
    hits = 0
    misses = 0

    for ray_origin, ray_dir, leaf, tri in sampled:
        # Transform to canonical coordinates (origin at 0, BVH face center on Z-axis)
        ray_dir_transformed, v0_transformed, v1_transformed, v2_transformed = \
            transform_to_canonical_coordinates(ray_origin, ray_dir, 
                                               leaf.bounds_min, leaf.bounds_max,
                                               tri.v0, tri.v1, tri.v2)

        # Compute hit/miss using transformed coordinates
        is_hit = ray_triangle_intersect(
            np.zeros(3, dtype=np.float32),
            ray_dir_transformed,
            v0_transformed,
            v1_transformed,
            v2_transformed
        )

        if is_hit:
            hits += 1
        else:
            misses += 1

        features = build_feature_vector(ray_dir_transformed, v0_transformed, 
                                        v1_transformed, v2_transformed)
        X_all.append(features)
        y_all.append(1 if is_hit else 0)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.uint8)

    print(f"Class distribution: {hits} hits ({100*hits/len(y_all):.1f}%), "
          f"{misses} misses ({100*misses/len(y_all):.1f}%)")

    print(f"Writing to {output_path}...")

    with open(output_path, 'wb') as f:
        for i in range(len(X_all)):
            f.write(X_all[i].tobytes())
            f.write(y_all[i:i+1].tobytes())
            f.write(b'\x00\x00\x00')

    file_size = os.path.getsize(output_path)
    expected_size = len(X_all) * 232

    print(f"Wrote {len(X_all)} samples ({file_size} bytes)")
    assert file_size == expected_size, f"Size mismatch: {file_size} vs expected {expected_size}"
    print("Done!")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BVH leaf sample text to binary format for NAS script"
    )
    parser.add_argument("input", help="Input .txt file with BVH leaf samples")
    parser.add_argument("-o", "--output", default="triangles.bin",
                        help="Output binary file (default: triangles.bin)")
    parser.add_argument("-n", "--num-triangles", type=int, default=10000,
                        help="Exact number of unique triangles to output (default: 10000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--all", action="store_true",
                        help="Use all triangles instead of random sampling")

    args = parser.parse_args()

    samples = parse_bvh_file(args.input)

    if not samples:
        print("Error: No valid samples parsed from input file.", file=sys.stderr)
        sys.exit(1)

    if args.all:
        # Use all triangles (no sampling)
        all_data = []
        for sample in samples:
            for leaf in sample.leaves:
                for tri in leaf.triangles:
                    all_data.append((leaf.ray_origin, leaf.ray_dir, leaf, tri))

        print(f"\nProcessing all {len(all_data)} triangles...")
        print("Applying coordinate normalization: ray origin at (0,0,0), BVH face center on Z-axis")

        X_all = []
        y_all = []
        hits = 0

        for ray_origin, ray_dir, leaf, tri in all_data:
            # Transform to canonical coordinates
            ray_dir_transformed, v0_transformed, v1_transformed, v2_transformed = \
                transform_to_canonical_coordinates(ray_origin, ray_dir,
                                                   leaf.bounds_min, leaf.bounds_max,
                                                   tri.v0, tri.v1, tri.v2)

            is_hit = ray_triangle_intersect(
                np.zeros(3, dtype=np.float32),
                ray_dir_transformed,
                v0_transformed,
                v1_transformed,
                v2_transformed
            )

            if is_hit:
                hits += 1

            features = build_feature_vector(ray_dir_transformed, v0_transformed,
                                            v1_transformed, v2_transformed)
            X_all.append(features)
            y_all.append(1 if is_hit else 0)

        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.uint8)

        misses = len(y_all) - hits
        print(f"Class distribution: {hits} hits ({100*hits/len(y_all):.1f}%), "
              f"{misses} misses ({100*misses/len(y_all):.1f}%)")

        print(f"Writing to {args.output}...")
        with open(args.output, 'wb') as f:
            for i in range(len(X_all)):
                f.write(X_all[i].tobytes())
                f.write(y_all[i:i+1].tobytes())
                f.write(b'\x00\x00\x00')

        print(f"Wrote {len(X_all)} samples")
    else:
        process_and_write_binary(
            samples,
            args.output,
            n_output_triangles=args.num_triangles,
            seed=args.seed
        )
