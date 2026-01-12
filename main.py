import os
import sys
import math
import gc
from collections import defaultdict
from datetime import datetime

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torch.nn.utils import clip_grad_norm_
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: PyTorch, NumPy, Scikit-Learn, and Matplotlib are required for this script.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------
# Binary format: 57 float32 + uint8 label + 3 pad bytes
# Use dtype.itemsize to avoid silent mismatches if dtype changes.
# ---------------------------------------------------------------------
triangle_dtype = np.dtype([
    ("x", "<f4", (57,)),  # 57 float32
    ("y", "u1"),          # uint8 label
    ("pad", "u1", (3,)),  # 3 bytes padding
])

SAMPLE_SIZE = triangle_dtype.itemsize
assert SAMPLE_SIZE == 232, f"Unexpected record size: {SAMPLE_SIZE} (expected 232)"


def read_triangle_samples_fast(filepath):
    """
    Efficiently reads triangles.bin into contiguous NumPy memory.

    Returns:
      X_base: (N,57) float32
      y_raw:  (N,)  uint8
    """
    print(f"Reading triangle data from '{filepath}' (fast NumPy path)...")
    if not os.path.exists(filepath):
        print(f"Error: Binary data file not found at '{filepath}'", file=sys.stderr)
        return None, None

    file_size = os.path.getsize(filepath)
    if file_size % SAMPLE_SIZE != 0:
        print(
            f"Warning: File size {file_size} is not a multiple of record size {SAMPLE_SIZE}. "
            f"Trailing bytes will be ignored.",
            file=sys.stderr
        )

    try:
        data = np.fromfile(filepath, dtype=triangle_dtype)
    except Exception as e:
        print(f"Error: Failed to np.fromfile('{filepath}'): {e}", file=sys.stderr)
        return None, None

    X_base = data["x"].astype(np.float32, copy=False)
    y_raw = data["y"]
    print(f"Successfully read {X_base.shape[0]} triangle samples.")
    return X_base, y_raw


def parse_nas_file(filepath):
    """Parses nas.txt and extracts architecture rows: n L H flops."""
    print(f"\nParsing NAS file from '{filepath}'...")
    if not os.path.exists(filepath):
        print(f"Error: NAS file not found at '{filepath}'", file=sys.stderr)
        return []

    all_architectures = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Table") or line.startswith("---"):
                    continue
                try:
                    arch_data = [int(val) for val in line.split()]
                    if len(arch_data) >= 4:
                        if len(arch_data) > 4:
                            print(f"Warning: NAS line has more than 4 integers, trimming: '{line}'", file=sys.stderr)
                        all_architectures.append(arch_data[:4])
                except ValueError:
                    print(f"Warning: Skipping unparsable line: '{line}'", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred while parsing NAS file: {e}", file=sys.stderr)
        return []

    print(f"Found {len(all_architectures)} total neural architectures.")
    return all_architectures


def build_feature_combos_by_sum(min_sum=7, max_sum=64):
    """
    Enumerate feature index combos. Assumes final X_full will be 64-dim:
      0..56 base floats
      57 ray_normal_dot
      58 determinant
      59 u_raw (not selectable)
      60 v_raw (not selectable)
      61 dot_v0
      62 dot_v1
      63 dot_v2

    Note: Indices 12, 17-20, and 23 from X_base_57 are intentionally unmapped
    and excluded from the feature search space. These correspond to features
    that were found to be unhelpful or redundant in preliminary analysis.
    """
    FEATURE_SLICES = {
        "ray_dir":            list(range(0, 3)),

        "vertex_positions":   list(range(3, 12)),
        "area":               [13],
        "unit_normal":        list(range(14, 17)),
        "aspect_ratio":       [22],
        "obb":                list(range(47, 57)),
        "ray_normal_dot":     [57],

        "centroid":           list(range(24, 27)),
        "orthocenter":        list(range(27, 30)),
        "fermat_point":       list(range(30, 33)),
        "circumcenter":       list(range(33, 36)),
        "circumradius":       [36],
        "incenter":           list(range(37, 40)),
        "inradius":           [40],

        "orientation_sign":   [21],
        "aabb":               list(range(41, 47)),

        "determinant":        [58],
        "u_raw":              [59],  # not selectable
        "v_raw":              [60],  # not selectable
        "dot_v0":             [61],
        "dot_v1":             [62],
        "dot_v2":             [63],
    }

    base_features = [
        "determinant", "dot_v0", "dot_v1", "dot_v2",
        "orientation_sign",
        "aabb",
    ]

    MT_FEATURES = {"determinant", "dot_v0", "dot_v1", "dot_v2"}

    groupA = ["vertex_positions", "area", "unit_normal", "aspect_ratio", "obb", "ray_normal_dot"]
    groupB = ["centroid", "orthocenter", "fermat_point", "circumcenter", "circumradius", "incenter", "inradius"]

    combos_by_sum = defaultdict(list)

    n_base = len(base_features)
    for mask in range(1 << n_base):
        chosen_base = []
        base_indices = []

        for i in range(n_base):
            if mask & (1 << i):
                name = base_features[i]
                chosen_base.append(name)
                base_indices.extend(FEATURE_SLICES[name])

        mt_selected = [name for name in chosen_base if name in MT_FEATURES]
        if len(mt_selected) > 2:
            continue

        for a in [None] + groupA:
            chosen_A = list(chosen_base)
            indices_A = list(base_indices)
            if a is not None:
                chosen_A.append(a)
                indices_A.extend(FEATURE_SLICES[a])

            for b in [None] + groupB:
                chosen = list(chosen_A)
                indices = list(indices_A)
                if b is not None:
                    chosen.append(b)
                    indices.extend(FEATURE_SLICES[b])

                final_indices = sorted(set(FEATURE_SLICES["ray_dir"] + indices))
                final_dim = len(final_indices)
                final_features = sorted(set(["ray_dir"] + chosen))

                if min_sum <= final_dim <= max_sum:
                    combos_by_sum[final_dim].append({
                        "sum": final_dim,
                        "features": final_features,
                        "indices": final_indices,
                    })

    return combos_by_sum


def validate_feature_combos(combos_by_sum, expected_dim=64):
    max_idx = -1
    for _, combos in combos_by_sum.items():
        for c in combos:
            if not c["indices"]:
                raise ValueError("Found an empty feature combo indices list.")
            m = max(c["indices"])
            if m > max_idx:
                max_idx = m

    if max_idx >= expected_dim:
        raise ValueError(
            f"Feature combos reference index {max_idx}, but expected_dim={expected_dim}. "
            "Your FEATURE_SLICES mapping is inconsistent with X_full."
        )


def create_model(n_inputs, n_layers, n_hidden):
    """
    MLP with LayerNorm instead of BatchNorm for robustness with small/variable batch sizes.
    LayerNorm normalizes across features rather than batch dimension, avoiding issues
    when batch size is 1 or very small.
    """
    layers = []
    layers.append(nn.Linear(n_inputs, n_hidden))
    layers.append(nn.LayerNorm(n_hidden))
    layers.append(nn.LeakyReLU(0.01))

    for _ in range(n_layers - 1):
        layers.append(nn.Linear(n_hidden, n_hidden))
        layers.append(nn.LayerNorm(n_hidden))
        layers.append(nn.LeakyReLU(0.01))

    layers.append(nn.Linear(n_hidden, 1))
    model = nn.Sequential(*layers)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_in")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return model


def calculate_calibration_bins(probabilities, true_labels, n_bins=10):
    """
    Returns per-bin positive rate (not accuracy).
    """
    if len(probabilities) == 0:
        return []

    probabilities = np.asarray(probabilities, dtype=np.float64)
    true_labels = np.asarray(true_labels, dtype=np.int32)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_totals = np.zeros(n_bins, dtype=np.int64)
    bin_positives = np.zeros(n_bins, dtype=np.int64)

    for prob, label in zip(probabilities, true_labels):
        if prob >= 1.0:
            idx = n_bins - 1
        elif prob <= 0.0:
            idx = 0
        else:
            idx = int(prob * n_bins)
            if idx >= n_bins:
                idx = n_bins - 1

        bin_totals[idx] += 1
        if label == 1:
            bin_positives[idx] += 1

    out = []
    for i in range(n_bins):
        total = int(bin_totals[i])
        pos = int(bin_positives[i])
        pos_rate = (pos / total) if total > 0 else np.nan
        out.append({
            "bin_start": float(bins[i]),
            "bin_end": float(bins[i + 1]),
            "total_samples": total,
            "positives": pos,
            "positive_rate": float(pos_rate) if not np.isnan(pos_rate) else np.nan
        })
    return out


def confusion_counts(y_true, pred_hit):
    y_true = y_true.astype(np.int32)
    pred_hit = pred_hit.astype(np.int32)
    tp = int(((y_true == 1) & (pred_hit == 1)).sum())
    tn = int(((y_true == 0) & (pred_hit == 0)).sum())
    fp = int(((y_true == 0) & (pred_hit == 1)).sum())
    fn = int(((y_true == 1) & (pred_hit == 0)).sum())
    return tp, tn, fp, fn


def culling_metrics_from_probs(probs_hit, y_true, threshold):
    """
    Predict HIT if prob >= threshold else MISS (culled).

    Cull accuracy = P(true miss | predicted miss) = TN / (TN + FN)
    Culled fraction = (TN + FN) / N
    """
    probs_hit = np.asarray(probs_hit, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)

    pred_hit = (probs_hit >= threshold).astype(np.int32)
    tp, tn, fp, fn = confusion_counts(y_true, pred_hit)

    n = len(y_true)
    culled = tn + fn

    overall_acc = (tp + tn) / n if n else 0.0
    hit_precision = tp / (tp + fp) if (tp + fp) else 0.0
    hit_recall = tp / (tp + fn) if (tp + fn) else 0.0

    cull_acc = (tn / culled) if culled else 1.0
    culled_frac = culled / n if n else 0.0
    true_culled_frac = tn / n if n else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "overall_accuracy": float(overall_acc),
        "hit_precision": float(hit_precision),
        "hit_recall": float(hit_recall),
        "cull_accuracy": float(cull_acc),
        "culled_fraction": float(culled_frac),
        "true_culled_fraction": float(true_culled_frac),
        "culled_count": int(culled),
    }


def select_threshold_max_cull_at_accuracy(probs_hit, y_true, min_cull_accuracy=0.95):
    """
    Fast threshold selection (no O(n_grid*N) scan).

    pred_hit = (prob >= threshold)
    predicted_miss (culled) = (prob < threshold)

    If no threshold yields cull_accuracy >= min_cull_accuracy with culled_count > 0,
    fallback is "cull nothing" (predict hit always) => threshold = 0.0.
    """
    probs = np.asarray(probs_hit, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int32)

    n = int(len(y))
    if n == 0:
        return culling_metrics_from_probs([], [], 0.0)

    order = np.argsort(probs, kind="mergesort")
    p_sorted = probs[order]
    y_sorted = y[order]  # 1=hit, 0=miss

    total_hits = int(y_sorted.sum())
    total_misses = n - total_hits

    cum_hits = np.cumsum(y_sorted, dtype=np.int64)  # hits in prefix

    def metrics_from_culled_k_and_thr(k, thr):
        # culled are first k samples in ascending prob (p < thr)
        if k <= 0:
            tn = 0
            fn = 0
            cull_acc = 1.0
        else:
            fn = int(cum_hits[k - 1])
            tn = int(k - fn)
            cull_acc = (tn / k) if k else 1.0

        tp = total_hits - fn
        fp = total_misses - tn

        overall_acc = (tp + tn) / n if n else 0.0
        hit_precision = tp / (tp + fp) if (tp + fp) else 0.0
        hit_recall = tp / (tp + fn) if (tp + fn) else 0.0

        culled = tn + fn
        return {
            "threshold": float(thr),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "overall_accuracy": float(overall_acc),
            "hit_precision": float(hit_precision),
            "hit_recall": float(hit_recall),
            "cull_accuracy": float(cull_acc),
            "culled_fraction": float(culled / n) if n else 0.0,
            "true_culled_fraction": float(tn / n) if n else 0.0,
            "culled_count": int(culled),
        }

    # Always allow "cull nothing"
    best = metrics_from_culled_k_and_thr(k=0, thr=0.0)

    unique_vals = np.unique(p_sorted)

    for v in unique_vals:
        if v >= 1.0:
            thr = 1.0
            k = int(np.searchsorted(p_sorted, 1.0, side="left"))  # p < 1.0
        else:
            thr = float(np.nextafter(v, 1.0))  # just above v
            k = int(np.searchsorted(p_sorted, thr, side="left"))

        m = metrics_from_culled_k_and_thr(k=k, thr=thr)

        if m["culled_count"] > 0 and m["cull_accuracy"] >= min_cull_accuracy:
            if (m["culled_count"] > best["culled_count"] or
                (m["culled_count"] == best["culled_count"] and m["cull_accuracy"] > best["cull_accuracy"]) or
                (m["culled_count"] == best["culled_count"] and m["cull_accuracy"] == best["cull_accuracy"] and m["threshold"] < best["threshold"])):
                best = m

    return best


def build_full_X_and_y(X_base_57, y_raw_u8, scale_ray_dirs=True):
    """
    Produces X_full with 64 float features and y in {0,1} float32.

    If scaling ray_dir, the model sees scaled ray_dir in X[:,0:3],
    but ray_normal_dot is computed using the ORIGINAL normalized ray direction
    so it remains an angle-like feature (not magnitude-weighted).
    """
    X = np.array(X_base_57, dtype=np.float32, copy=True)
    y = (y_raw_u8.astype(np.int32) > 0).astype(np.float32)

    # Preserve original ray direction for ray_normal_dot
    ray_dir_orig = X[:, 0:3].copy()
    ray_dir_orig_norm = np.linalg.norm(ray_dir_orig, axis=1, keepdims=True).astype(np.float32)
    ray_dir_orig_unit = ray_dir_orig / np.where(ray_dir_orig_norm > 1e-12, ray_dir_orig_norm, 1.0)

    if scale_ray_dirs:
        v0 = X[:, 3:6]
        lengths = np.linalg.norm(v0, axis=1, keepdims=True).astype(np.float32)
        # FIX: Prevent zeroing out ray direction when v0 is at origin
        lengths = np.where(lengths > 1e-12, lengths, 1.0)
        X[:, 0:3] *= lengths
        print("Scaled ray directions using distance to v0 (model sees scaled ray_dir).")
    else:
        print("Ray direction scaling is DISABLED (SCALE_RAY_DIR=False).")

    # Derived feature: ray_normal_dot -> idx 57 (angle-like)
    unit_normal = X[:, 14:17]
    ray_normal_dot = np.sum(ray_dir_orig_unit * unit_normal, axis=1, keepdims=True).astype(np.float32)
    X = np.hstack((X, ray_normal_dot))

    # MT-ish derived features -> idx 58..63 (use model-visible ray_dir)
    v0 = X[:, 3:6]
    v1 = X[:, 6:9]
    v2 = X[:, 9:12]
    ray_dir = X[:, 0:3]

    e1 = v1 - v0
    e2 = v2 - v0
    tvec = -v0

    pvec = np.cross(ray_dir, e2, axis=1)
    qvec = np.cross(tvec, e1, axis=1)

    determinant = np.sum(e1 * pvec, axis=1, keepdims=True).astype(np.float32)
    u_raw = np.sum(tvec * pvec, axis=1, keepdims=True).astype(np.float32)
    v_raw = np.sum(ray_dir * qvec, axis=1, keepdims=True).astype(np.float32)

    dot_v0 = np.sum(ray_dir * v0, axis=1, keepdims=True).astype(np.float32)
    dot_v1 = np.sum(ray_dir * v1, axis=1, keepdims=True).astype(np.float32)
    dot_v2 = np.sum(ray_dir * v2, axis=1, keepdims=True).astype(np.float32)

    X = np.hstack((X, determinant, u_raw, v_raw, dot_v0, dot_v1, dot_v2)).astype(np.float32)
    print(f"Built X_full with derived features. Shape: {X.shape} (expected N x 64)")
    return X, y


def precompute_train_scaler_stats(X_full, train_idx, eps=1e-8):
    """
    Equivalent to StandardScaler per-feature, computed ONCE on train split.
    Then for any feature subset, use mean/std slices.
    """
    X_train_full = X_full[train_idx]
    mean = X_train_full.mean(axis=0).astype(np.float32)
    std = X_train_full.std(axis=0).astype(np.float32)
    std = np.where(std > eps, std, 1.0).astype(np.float32)
    inv_std = (1.0 / std).astype(np.float32)
    return mean, inv_std


def scale_with_stats(X, mean, inv_std):
    """
    Apply standardization using precomputed mean and inverse std.
    Warns if non-finite values are encountered and replaced.
    """
    Xs = (X - mean) * inv_std
    if not np.isfinite(Xs).all():
        bad_count = int((~np.isfinite(Xs)).sum())
        print(f"Warning: {bad_count} non-finite values detected in scaled data, replacing with 0.0", file=sys.stderr)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    return Xs


def save_topk_checkpoints(top_k_models, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    saved_paths = []
    for rank, m in enumerate(top_k_models, start=1):
        n, L, H, flops = m["arch"]

        # Status based on TEST (available after final evaluation)
        if m.get("test_cull_accuracy") is None:
            status = "NOTEST"
        else:
            status = "PASS" if m["test_cull_accuracy"] >= m["min_cull_accuracy"] else "FAIL"

        total_cost = m.get("total_cost", "NA")
        thr = m.get("threshold", 0.0)
        test_cull = m.get("test_culled_fraction", None)
        test_acc = m.get("test_cull_accuracy", None)

        if test_cull is None or test_acc is None:
            fname = f"rank_{rank:02d}_{status}_n{n}_L{L}_H{H}_cost{total_cost}_thr{thr:.6f}.pt"
        else:
            fname = (
                f"rank_{rank:02d}_{status}_n{n}_L{L}_H{H}_cost{total_cost}_"
                f"cull{test_cull:.4f}_acc{test_acc:.4f}.pt"
            )

        path = os.path.join(checkpoint_dir, fname)

        payload = {
            "arch": m["arch"],
            "feature_names": m["feature_names"],
            "feature_indices": m["feature_indices"],
            "threshold": float(m["threshold"]),
            "total_cost": m["total_cost"],
            "min_cull_accuracy": m["min_cull_accuracy"],
            "scale_ray_dirs": m["scale_ray_dirs"],

            # Selection metrics (VAL)
            "val_metrics": {
                "val_culled_fraction": m.get("val_culled_fraction"),
                "val_cull_accuracy": m.get("val_cull_accuracy"),
                "val_overall_accuracy": m.get("val_overall_accuracy"),
                "val_hit_recall": m.get("val_hit_recall"),
                "val_hit_precision": m.get("val_hit_precision"),
            },

            # Final evaluation metrics (TEST), filled for selected models
            "test_metrics": {
                "test_culled_fraction": m.get("test_culled_fraction"),
                "test_cull_accuracy": m.get("test_cull_accuracy"),
                "test_overall_accuracy": m.get("test_overall_accuracy"),
                "test_hit_recall": m.get("test_hit_recall"),
                "test_hit_precision": m.get("test_hit_precision"),
            },

            # Inference essentials
            "model_state": m["model_state_cpu"],       # CPU tensors
            "scaler_mean_full": m["scaler_mean_full"], # length 64
            "scaler_invstd_full": m["scaler_invstd_full"],

            # Optional calibration info (TEST)
            "calibration": m.get("calibration", None),

            "created": datetime.now().isoformat(),
        }

        torch.save(payload, path)
        saved_paths.append(path)
    return saved_paths


def log_results_to_file(results, attempted_runs, trained_runs, planned_trains, filepath, min_cull_accuracy, topk_models, checkpoint_paths=None):
    """
    Writes a report that is correct under:
      - selection/ranking on VAL
      - TEST evaluation only for selected top-k models
    """
    print(f"\nLogging results to '{filepath}'...")

    trained_only = [r for r in results if r.get("trained", False)]
    passing_val = [r for r in trained_only if (r.get("val_cull_accuracy") is not None and r["val_cull_accuracy"] >= min_cull_accuracy)]
    passing_val_sorted = sorted(passing_val, key=lambda r: r["val_culled_fraction"], reverse=True)

    try:
        with open(filepath, "w") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write("--- NAS Culling Report ---\n")
            f.write(f"Generated: {now}\n")
            f.write(f"Attempted combos:        {attempted_runs}\n")
            f.write(f"Planned training runs:   {planned_trains}\n")
            f.write(f"Completed training runs: {trained_runs}\n")
            f.write(f"Selection metric (VAL): maximize culled fraction subject to cull accuracy >= {min_cull_accuracy*100:.1f}%\n")
            f.write("\n")

            f.write(f"VAL-passing models: {len(passing_val)} / {len(trained_only)}\n\n")

            if passing_val_sorted:
                best = passing_val_sorted[0]
                n, L, H, flops = best["arch"]
                f.write("BEST MODEL BY VAL (meets constraint on VAL):\n")
                f.write(f"  Arch(n={n}, L={L}, H={H}, flops={flops})\n")
                f.write(f"  Total cost: {best['total_cost']}\n")
                f.write(f"  Threshold (chosen on VAL): {best['threshold']:.6f}\n")
                f.write(f"  VAL culled fraction: {best['val_culled_fraction']*100:.2f}%\n")
                f.write(f"  VAL cull accuracy:   {best['val_cull_accuracy']*100:.2f}%\n")
                f.write(f"  VAL overall acc:     {best['val_overall_accuracy']*100:.2f}%\n")
                f.write(f"  VAL hit recall:      {best['val_hit_recall']*100:.2f}%\n")
                f.write(f"  Features: {', '.join(best['feature_names'])}\n\n")
            else:
                f.write("No models met the cull-accuracy requirement on VAL.\n\n")

            f.write("TOP 20 VAL-PASSING MODELS (by VAL culled fraction):\n")
            f.write("=" * 70 + "\n")
            for i, r in enumerate(passing_val_sorted[:20], start=1):
                n, L, H, flops = r["arch"]
                f.write(f"Rank {i:02d}: n={n} L={L} H={H} cost={r['total_cost']} thr={r['threshold']:.6f}\n")
                f.write(
                    f"  VAL culled={r['val_culled_fraction']*100:.2f}% | cull_acc={r['val_cull_accuracy']*100:.2f}% | "
                    f"overall={r['val_overall_accuracy']*100:.2f}% | hit_recall={r['val_hit_recall']*100:.2f}%\n"
                )
                f.write(f"  Features: {', '.join(r['feature_names'])}\n\n")

            f.write("\n")
            f.write("SELECTED TOP-K MODELS: TEST EVALUATION (not used for selection)\n")
            f.write("=" * 70 + "\n")

            if checkpoint_paths:
                f.write("Saved checkpoints:\n")
                for p in checkpoint_paths:
                    f.write(f"  {p}\n")
                f.write("\n")

            for i, m in enumerate(topk_models, start=1):
                n, L, H, flops = m["arch"]
                f.write(f"TopK {i:02d}: Arch(n={n}, L={L}, H={H}, flops={flops}) cost={m['total_cost']} thr={m['threshold']:.6f}\n")
                f.write(
                    f"  VAL culled={m.get('val_culled_fraction', 0.0)*100:.2f}% | VAL cull_acc={m.get('val_cull_accuracy', 0.0)*100:.2f}%\n"
                )
                if m.get("test_culled_fraction") is not None:
                    status = "PASS" if m["test_cull_accuracy"] >= min_cull_accuracy else "FAIL"
                    f.write(
                        f"  TEST [{status}] culled={m['test_culled_fraction']*100:.2f}% | cull_acc={m['test_cull_accuracy']*100:.2f}% | "
                        f"overall={m['test_overall_accuracy']*100:.2f}% | hit_recall={m['test_hit_recall']*100:.2f}%\n"
                    )
                else:
                    f.write("  TEST: not evaluated\n")
                f.write(f"  Features: {', '.join(m['feature_names'])}\n\n")

        print("     Successfully wrote report.")
    except Exception as e:
        print(f"Error: Could not write report: {e}", file=sys.stderr)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    DATA_FILE = "triangles.bin"
    NAS_FILE = "nas.txt"
    GRAPH_DIR = "graphs"
    CHECKPOINT_DIR = "checkpoints"

    TEST_SPLIT_SIZE = 0.2
    VALIDATION_SPLIT_SIZE = 0.2

    EPOCHS_PER_MODEL = 30
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP_VALUE = 1.0

    # Early stopping configuration
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 1e-5

    MIN_CULL_ACCURACY = 0.95
    TOP_K = 20

    SCALE_RAY_DIR = True

    # Cost constants
    # Note: u_raw and v_raw are not selectable features, so their costs are
    # listed here only for documentation/completeness but will never be used.
    MT_FEATURE_COSTS = {
        "determinant": 5,
        "dot_v0": 8,
        "dot_v1": 8,
        "dot_v2": 8,
    }
    TOTAL_COST_LIMIT = 45

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark is enabled.")

    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Graphs will be saved to:       {GRAPH_DIR}/")
    print(f"Checkpoints will be saved to:  {CHECKPOINT_DIR}/")

    # ---- Load data fast ----
    X_base_57, y_raw_u8 = read_triangle_samples_fast(DATA_FILE)
    if X_base_57 is None:
        sys.exit(1)

    assert X_base_57.ndim == 2 and X_base_57.shape[1] == 57, f"Expected X_base_57 to be Nx57, got {X_base_57.shape}"

    all_architectures = parse_nas_file(NAS_FILE)
    if not all_architectures:
        sys.exit(1)

    # ---- Build features ----
    combos_by_sum = build_feature_combos_by_sum(min_sum=7, max_sum=64)
    validate_feature_combos(combos_by_sum, expected_dim=64)

    total_combos = sum(len(v) for v in combos_by_sum.values())
    print(f"\nFound {len(all_architectures)} architectures.")
    print(f"Found {total_combos} total feature combos.")

    # ---- Build X_full, y ----
    X_full, y = build_full_X_and_y(X_base_57, y_raw_u8, scale_ray_dirs=SCALE_RAY_DIR)
    assert X_full.shape[1] == 64, f"Expected X_full to be Nx64, got {X_full.shape}"
    assert len(y) == len(X_full), "X/y length mismatch"

    # ---- Splits ----
    all_indices = np.arange(len(y))
    y_int = y.astype(np.int32)

    train_full_idx, test_idx = train_test_split(
        all_indices, test_size=TEST_SPLIT_SIZE, random_state=42, stratify=y_int
    )
    train_idx, val_idx = train_test_split(
        train_full_idx, test_size=VALIDATION_SPLIT_SIZE, random_state=42, stratify=y_int[train_full_idx]
    )

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    miss_count = int((y_train == 0).sum())
    hit_count = int((y_train == 1).sum())
    print(f"Class distribution (train): {miss_count} Misses / {hit_count} Hits.")

    if hit_count > 0:
        pos_weight_value = miss_count / hit_count
    else:
        pos_weight_value = 1.0
        print("Warning: no hits in train split; pos_weight forced to 1.0")

    #pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    # ---- Precompute scaling stats ONCE ----
    scaler_mean_full, scaler_invstd_full = precompute_train_scaler_stats(X_full, train_idx)

    # ---- Precompute planned training runs (after cost filtering) ----
    planned_trains = 0
    attempted_runs = 0
    for arch in all_architectures:
        n, L, H, flops = arch
        for combo in combos_by_sum.get(n, []):
            attempted_runs += 1
            mt_cost = sum(MT_FEATURE_COSTS.get(fname, 0) for fname in combo["features"])
            if flops + mt_cost <= TOTAL_COST_LIMIT:
                planned_trains += 1

    print(f"Attempted runs (before cost filtering): {attempted_runs}")
    print(f"Planned training runs (after cost filtering): {planned_trains}\n")

    # ---- Training bookkeeping ----
    results = []
    trained_runs = 0
    top_k_models = []

    def rank_score(m):
        # Rank ONLY by VAL to avoid TEST leakage
        if m.get("val_cull_accuracy", 0.0) >= MIN_CULL_ACCURACY:
            return float(m.get("val_culled_fraction", -1.0))
        return -1.0

    def consider_for_topk(candidate):
        if len(top_k_models) < TOP_K:
            top_k_models.append(candidate)
            return

        worst_i = 0
        worst_score = rank_score(top_k_models[0])
        for i in range(1, len(top_k_models)):
            sc = rank_score(top_k_models[i])
            if sc < worst_score:
                worst_score = sc
                worst_i = i

        cand_score = rank_score(candidate)
        if cand_score > worst_score:
            top_k_models[worst_i] = candidate

    # -----------------------------------------------------------------
    # Main NAS loop (selection on VAL only)
    # -----------------------------------------------------------------
    attempted_counter = 0
    for model_idx, arch in enumerate(all_architectures):
        n, L, H, flops = arch
        feature_sets = combos_by_sum.get(n, [])
        if not feature_sets:
            continue

        print("\n" + "=" * 80)
        print(f"Arch {model_idx + 1}/{len(all_architectures)}: n={n} L={L} H={H} (feature sets: {len(feature_sets)})")
        print("=" * 80)

        for feat_set_idx, combo in enumerate(feature_sets):
            attempted_counter += 1

            current_indices = combo["indices"]
            current_feature_names = combo["features"]

            mt_cost = sum(MT_FEATURE_COSTS.get(fname, 0) for fname in current_feature_names)
            total_cost = flops + mt_cost
            if total_cost > TOTAL_COST_LIMIT:
                continue

            # ---- Build scaled subsets using precomputed mean/std ----
            X_subset = X_full[:, current_indices]
            mean_sub = scaler_mean_full[current_indices]
            invstd_sub = scaler_invstd_full[current_indices]

            X_train_s = scale_with_stats(X_subset[train_idx], mean_sub, invstd_sub)
            X_val_s = scale_with_stats(X_subset[val_idx], mean_sub, invstd_sub)

            X_train_s = np.ascontiguousarray(X_train_s, dtype=np.float32)
            X_val_s = np.ascontiguousarray(X_val_s, dtype=np.float32)

            y_train_np = np.ascontiguousarray(y_train, dtype=np.float32)
            y_val_np = np.ascontiguousarray(y_val, dtype=np.float32)

            # Prefer from_numpy to avoid extra copies
            X_train_tensor = torch.from_numpy(X_train_s)
            y_train_tensor = torch.from_numpy(y_train_np)

            X_val_tensor = torch.from_numpy(X_val_s).to(device, non_blocking=True)
            y_val_tensor = torch.from_numpy(y_val_np).to(device, non_blocking=True)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

            # FIX: Ensure minimum batch size of 2 to avoid issues with normalization layers
            # even though we switched to LayerNorm (which handles batch size 1), this is
            # still good practice for stable training dynamics
            effective_batch_size = min(BATCH_SIZE, len(train_dataset))
            if effective_batch_size < 2:
                effective_batch_size = len(train_dataset)

            drop_last_flag = (len(train_dataset) > effective_batch_size)
            train_loader = DataLoader(
                train_dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
                drop_last=drop_last_flag,
            )

            if len(train_loader) == 0:
                print(f"Warning: Empty train loader for feature set {feat_set_idx}, skipping", file=sys.stderr)
                continue

            print("\n" + "-" * 60)
            print(f"TrainRun {trained_runs + 1}/{planned_trains} | Arch(n={n},L={L},H={H}) | Cost={total_cost}")
            print(f"FeatureSet {feat_set_idx + 1}/{len(feature_sets)}: {', '.join(current_feature_names)}")
            print("-" * 60)

            model = create_model(n, L, H).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5], dtype=torch.float32, device=device))
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.2, patience=3, verbose=False
            )

            best_val_loss = float("inf")
            best_model_state_cpu = None
            epochs_without_improvement = 0

            for epoch in range(EPOCHS_PER_MODEL):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(batch_X).squeeze(1)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val_tensor).squeeze(1)
                    val_loss = float(criterion(val_logits, y_val_tensor).item())

                scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
                    best_val_loss = val_loss
                    best_model_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                    break

            if best_model_state_cpu is None:
                best_model_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            trained_runs += 1

            # ---- Threshold selection on VAL (maximize culling with >= MIN_CULL_ACCURACY) ----
            model.load_state_dict(best_model_state_cpu)
            model.to(device)
            model.eval()

            with torch.no_grad():
                val_probs = torch.sigmoid(model(X_val_tensor).squeeze(1)).cpu().numpy()

            val_choice = select_threshold_max_cull_at_accuracy(
                probs_hit=val_probs,
                y_true=y_val,
                min_cull_accuracy=MIN_CULL_ACCURACY
            )

            thr = float(val_choice["threshold"])

            print(f"VAL @ thr={thr:.6f} | Culled={val_choice['culled_fraction']*100:.2f}% | "
                  f"CullAcc={val_choice['cull_accuracy']*100:.2f}% | "
                  f"OverallAcc={val_choice['overall_accuracy']*100:.2f}% | "
                  f"HitRecall={val_choice['hit_recall']*100:.2f}%")

            run_entry = {
                "trained": True,
                "arch": arch,
                "feature_names": current_feature_names,
                "feature_indices": current_indices,
                "total_cost": total_cost,
                "threshold": thr,

                # Selection metrics (VAL)
                "val_culled_fraction": float(val_choice["culled_fraction"]),
                "val_cull_accuracy": float(val_choice["cull_accuracy"]),
                "val_overall_accuracy": float(val_choice["overall_accuracy"]),
                "val_hit_recall": float(val_choice["hit_recall"]),
                "val_hit_precision": float(val_choice["hit_precision"]),

                # TEST metrics filled later only for selected models
                "test_culled_fraction": None,
                "test_cull_accuracy": None,
                "test_overall_accuracy": None,
                "test_hit_recall": None,
                "test_hit_precision": None,
                "calibration": None,
            }
            results.append(run_entry)

            # Candidate for top-k selection (heavy artifacts stored CPU-side)
            candidate = dict(run_entry)
            candidate.update({
                "model_state_cpu": best_model_state_cpu,
                "scaler_mean_full": scaler_mean_full.copy(),
                "scaler_invstd_full": scaler_invstd_full.copy(),
                "min_cull_accuracy": MIN_CULL_ACCURACY,
                "scale_ray_dirs": SCALE_RAY_DIR,
            })
            consider_for_topk(candidate)

            # ---- Memory cleanup ----
            del model, optimizer, scheduler, criterion
            del X_train_s, X_val_s, X_train_tensor, X_val_tensor, y_val_tensor
            del train_dataset, train_loader
            del best_model_state_cpu
            gc.collect()
            if device.type == "cuda" and (trained_runs % 50 == 0):
                torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Final: evaluate TEST only for selected top-k models
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Done. Attempted combos: {attempted_counter} | Planned trains: {planned_trains} | Trained runs: {trained_runs}")
    print(f"Constraint (selection on VAL): Cull accuracy >= {MIN_CULL_ACCURACY*100:.1f}%")
    print("=" * 70)

    top_k_models_sorted = sorted(top_k_models, key=lambda m: rank_score(m), reverse=True)

    print("\nEvaluating TEST for selected top-K models (TEST not used for selection)...")
    evaluated_topk = []

    for i, m in enumerate(top_k_models_sorted, start=1):
        if i > TOP_K:
            break

        n, L, H, flops = m["arch"]
        idxs = m["feature_indices"]

        model = create_model(n, L, H).to(device)
        model.load_state_dict(m["model_state_cpu"])
        model.eval()

        X_subset = X_full[:, idxs]
        mean_sub = m["scaler_mean_full"][idxs]
        invstd_sub = m["scaler_invstd_full"][idxs]
        X_test_s = scale_with_stats(X_subset[test_idx], mean_sub, invstd_sub)
        X_test_s = np.ascontiguousarray(X_test_s, dtype=np.float32)
        X_test_tensor = torch.from_numpy(X_test_s).to(device, non_blocking=True)

        with torch.no_grad():
            test_probs = torch.sigmoid(model(X_test_tensor).squeeze(1)).cpu().numpy()

        test_metrics = culling_metrics_from_probs(test_probs, y_test, m["threshold"])
        calibration = calculate_calibration_bins(test_probs, y_test, n_bins=10)

        m["test_culled_fraction"] = float(test_metrics["culled_fraction"])
        m["test_cull_accuracy"] = float(test_metrics["cull_accuracy"])
        m["test_overall_accuracy"] = float(test_metrics["overall_accuracy"])
        m["test_hit_recall"] = float(test_metrics["hit_recall"])
        m["test_hit_precision"] = float(test_metrics["hit_precision"])
        m["calibration"] = calibration

        status = "PASS" if m["test_cull_accuracy"] >= MIN_CULL_ACCURACY else "FAIL"
        print(f"Top {i:02d} TEST [{status}] | Culled={m['test_culled_fraction']*100:.2f}% | "
              f"CullAcc={m['test_cull_accuracy']*100:.2f}% | "
              f"OverallAcc={m['test_overall_accuracy']*100:.2f}% | "
              f"HitRecall={m['test_hit_recall']*100:.2f}%")

        evaluated_topk.append(m)

        del model, X_test_tensor, X_test_s
        gc.collect()

    # ---- Save top-K checkpoints ----
    checkpoint_paths = save_topk_checkpoints(evaluated_topk, CHECKPOINT_DIR)
    print(f"\nSaved {len(checkpoint_paths)} checkpoints to '{CHECKPOINT_DIR}/'.")

    # -----------------------------------------------------------------
    # Optional: graph generation from evaluated top models (TEST probs)
    # -----------------------------------------------------------------
    print(f"\nGenerating guess distribution graphs for up to {TOP_K} top models into '{GRAPH_DIR}/'...")
    plotted = 0

    for i, m in enumerate(evaluated_topk, start=1):
        try:
            n, L, H, flops = m["arch"]

            model = create_model(n, L, H)
            model.load_state_dict(m["model_state_cpu"])
            model = model.to(device)
            model.eval()

            idxs = m["feature_indices"]
            X_subset = X_full[:, idxs]
            mean_sub = m["scaler_mean_full"][idxs]
            invstd_sub = m["scaler_invstd_full"][idxs]
            X_test_s = scale_with_stats(X_subset[test_idx], mean_sub, invstd_sub)
            X_test_s = np.ascontiguousarray(X_test_s, dtype=np.float32)
            X_test_tensor = torch.from_numpy(X_test_s).to(device, non_blocking=True)

            with torch.no_grad():
                test_probs = torch.sigmoid(model(X_test_tensor).squeeze(1)).cpu().numpy()

            probs_hits = test_probs[y_test == 1]
            probs_misses = test_probs[y_test == 0]
            mean_hits = float(np.mean(probs_hits)) if len(probs_hits) else 0.0
            mean_misses = float(np.mean(probs_misses)) if len(probs_misses) else 0.0

            n_bins = 20
            bins = np.linspace(0, 1, n_bins + 1)
            bin_hits = np.zeros(n_bins, dtype=int)
            bin_misses = np.zeros(n_bins, dtype=int)

            for prob, label in zip(test_probs, y_test):
                j = (n_bins - 1) if prob >= 1.0 else (0 if prob <= 0.0 else int(prob * n_bins))
                if label == 1:
                    bin_hits[j] += 1
                else:
                    bin_misses[j] += 1

            # Avoid zeros on log scale
            bin_hits_plot = np.maximum(bin_hits, 1)
            bin_misses_plot = np.maximum(bin_misses, 1)

            bin_centers = (bins[:-1] + bins[1:]) / 2

            plt.figure(figsize=(12, 7))
            plt.plot(bin_centers, bin_hits_plot, label=f"Hits (1) mean={mean_hits:.3f}", marker="o", linestyle="-")
            plt.plot(bin_centers, bin_misses_plot, label=f"Miss (0) mean={mean_misses:.3f}", marker="x", linestyle="--")
            plt.yscale("log")
            plt.gca().set_ylim(bottom=1)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.xlabel("Predicted P(hit) bin")
            plt.ylabel("Count (log scale)")

            status = "PASS" if m["test_cull_accuracy"] >= MIN_CULL_ACCURACY else "FAIL"
            feat_str = ", ".join(m["feature_names"])
            if len(feat_str) > 70:
                feat_str = feat_str[:70] + "..."

            plt.title(
                f"Top {i:02d} [{status}]  n={n} L={L} H={H}  Cost={m['total_cost']}  Thr={m['threshold']:.6f}\n"
                f"VAL Culled={m['val_culled_fraction']*100:.2f}%  VAL CullAcc={m['val_cull_accuracy']*100:.2f}%\n"
                f"TEST Culled={m['test_culled_fraction']*100:.2f}%  TEST CullAcc={m['test_cull_accuracy']*100:.2f}%\n"
                f"Features: {feat_str}"
            )
            plt.legend()

            fname = f"top_{i:02d}_{status}_test_cull_{m['test_culled_fraction']:.4f}_acc_{m['test_cull_accuracy']:.4f}.png"
            plt.savefig(os.path.join(GRAPH_DIR, fname))
            plt.close()
            plotted += 1

            del model, X_test_tensor, X_test_s
            gc.collect()
        except Exception as e:
            print(f"Graph error for top model {i}: {e}", file=sys.stderr)

    print(f"Generated {plotted} graphs.")

    # ---- Log file ----
    log_results_to_file(
        results=results,
        attempted_runs=attempted_counter,
        trained_runs=trained_runs,
        planned_trains=planned_trains,
        filepath="nas_exhaustive_report.txt",
        min_cull_accuracy=MIN_CULL_ACCURACY,
        topk_models=evaluated_topk,
        checkpoint_paths=checkpoint_paths
    )

    print("\n--- Script Finished ---")
