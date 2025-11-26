import struct
import random
import os
import sys
import math
from collections import Counter, defaultdict
from itertools import combinations  # currently unused
from datetime import datetime
import copy

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from torch.nn.utils import clip_grad_norm_
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_recall_fscore_support
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: PyTorch, NumPy, Scikit-Learn, and Matplotlib are required for this script.", file=sys.stderr)
    sys.exit(1)


SAMPLE_FORMAT = '<57fB3x'
SAMPLE_SIZE = struct.calcsize(SAMPLE_FORMAT)  # should be 232


def read_triangle_samples(filepath):
    """Reads all TriangleSample structs from an existing binary file."""
    print(f"Reading triangle data from '{filepath}'...")
    if not os.path.exists(filepath):
        print(f"Error: Binary data file not found at '{filepath}'", file=sys.stderr)
        return []

    print(f"Expected struct size: {SAMPLE_SIZE} bytes (Format: {SAMPLE_FORMAT})")
    samples = []
    try:
        with open(filepath, 'rb') as f:
            while True:
                data_chunk = f.read(SAMPLE_SIZE)
                if not data_chunk:
                    break
                if len(data_chunk) < SAMPLE_SIZE:
                    print(
                        f"Warning: Partial record found at end of file. "
                        f"Expected {SAMPLE_SIZE} bytes, got {len(data_chunk)}.",
                        file=sys.stderr
                    )
                    break
                samples.append(struct.unpack(SAMPLE_FORMAT, data_chunk))
    except Exception as e:
        print(f"An error occurred while reading binary file: {e}", file=sys.stderr)
        return []

    print(f"Successfully read {len(samples)} triangle samples.")
    return samples


def parse_nas_file(filepath):
    """Parses the nas.txt file and extracts all architecture rows."""
    print(f"\nParsing NAS file from '{filepath}'...")
    if not os.path.exists(filepath):
        print(f"Error: NAS file not found at '{filepath}'", file=sys.stderr)
        return []

    all_architectures = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Table') or line.startswith('---'):
                    continue
                try:
                    arch_data = [int(val) for val in line.split()]
                    if len(arch_data) >= 4:
                        if len(arch_data) > 4:
                            print(f"Warning: NAS line has more than 4 integers, trimming: '{line}'", file=sys.stderr)
                        arch_data = arch_data[:4]
                        all_architectures.append(arch_data)
                except ValueError:
                    print(f"Warning: Skipping unparsable line: '{line}'", file=sys.stderr)

    except Exception as e:
        print(f"An error occurred while parsing NAS file: {e}", file=sys.stderr)
        return []

    print(f"Found {len(all_architectures)} total neural architectures.")
    return all_architectures


def build_feature_combos_by_sum(min_sum=7, max_sum=64):
    """
    Enumerate all possible sets of feature indices under certain structural constraints.
    
    min_sum / max_sum refer to the FINAL input dimension seen by the model,
    including the 3 ray_dir components (indices 0..2), which are always added.
    
    This version is redesigned to:
    1. Add new "cheap" features derived from Möller-Trumbore (indices 58-63).
    2. Remove "useless" features ('barycentric_condition', 'perimeter', 'plane_eq').
    3. Prioritize combinations of cheap features in the search.
    """

    FEATURE_SLICES = {
        # --- Core ---
        "ray_dir":            list(range(0, 3)),   # 3
        
        # --- Expensive Geometric (Group A) ---
        "vertex_positions":   list(range(3, 12)),  # 9
        "area":               [13],                # 1
        "unit_normal":        list(range(14, 17)), # 3
        "aspect_ratio":       [22],                # 1
        "obb":                list(range(47, 57)), # 10
        "ray_normal_dot":     [57],                # 1 (Expensive: needs unit_normal)

        # --- Triangle Centers (Group B) ---
        "centroid":           list(range(24, 27)), # 3
        "orthocenter":        list(range(27, 30)), # 3
        "fermat_point":       list(range(30, 33)), # 3
        "circumcenter":       list(range(33, 36)), # 3
        "circumradius":       [36],                # 1
        "incenter":           list(range(37, 40)), # 3
        "inradius":           [40],                # 1

        # --- Cheap Geometric (New Base) ---
        "orientation_sign":   [21],                # 1
        "aabb":               list(range(41, 47)), # 6
        
        # Möller-Trumbore Features
        "determinant":        [58],                # 1 
        "u_raw":              [59],                # 1
        "v_raw":              [60],                # 1
        "dot_v0":             [61],                # 1
        "dot_v1":             [62],                # 1
        "dot_v2":             [63],                # 1
    }

    base_features = [
        "determinant",        # Replaces ray_normal_dot / unit_normal
        "u_raw",              # Replaces barycentric_condition
        "v_raw",              # Replaces barycentric_condition
        "dot_v0",             # Replaces vertex_positions
        "dot_v1",             # 
        "dot_v2",             # 
        "orientation_sign",   # 
        "aabb",               # 
    ]

    groupA = [
        "vertex_positions",  
        "area",
        "unit_normal",        
        "aspect_ratio",
        "obb",
        "ray_normal_dot"      
    ]

    groupB = [
        "centroid", 
        "orthocenter", 
        "fermat_point",
        "circumcenter", 
        "circumradius", 
        "incenter", 
        "inradius"
    ]
    

    combos_by_sum = defaultdict(list)

    n_base = len(base_features)
    # Loop over all 2^n combinations of the cheap base_features
    for mask in range(1 << n_base):
        chosen_base = []
        base_indices = []

        for i in range(n_base):
            if mask & (1 << i):
                name = base_features[i]
                chosen_base.append(name)

                base_indices.extend(FEATURE_SLICES[name])

        groupA_choices = [None] + groupA
        for a in groupA_choices:
            chosen_A = list(chosen_base)
            indices_A = list(base_indices)
            if a is not None:
                chosen_A.append(a)
                indices_A.extend(FEATURE_SLICES[a])

            groupB_choices = [None] + groupB
            for b in groupB_choices:
                chosen = list(chosen_A)
                indices = list(indices_A)
                if b is not None:
                    chosen.append(b)
                    indices.extend(FEATURE_SLICES[b])

                indices = sorted(set(indices))

                final_indices = sorted(set(FEATURE_SLICES["ray_dir"] + indices))
                final_dim = len(final_indices)
                
                final_features = sorted(list(set(["ray_dir"] + chosen)))

                if min_sum <= final_dim <= max_sum:
                    combos_by_sum[final_dim].append({
                        "sum": final_dim,
                        "features": final_features,
                        "indices": final_indices,
                    })

    return combos_by_sum


def create_model(n_inputs, n_layers, n_hidden):
    """
    Dynamically creates a PyTorch MLP with Batch Normalization
    and Kaiming He initialization.
    """
    layers = []

    layers.append(nn.Linear(n_inputs, n_hidden))
    layers.append(nn.BatchNorm1d(n_hidden))
    layers.append(nn.LeakyReLU(0.01))

    for _ in range(n_layers - 1):
        layers.append(nn.Linear(n_hidden, n_hidden))
        layers.append(nn.BatchNorm1d(n_hidden))
        layers.append(nn.LeakyReLU(0.01))

    layers.append(nn.Linear(n_hidden, 1))

    model = nn.Sequential(*layers)

    # Kaiming He initialization
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return model

def calculate_calibration_bins(probabilities, true_labels, n_bins=10):
    """
    Calculates the accuracy (proportion of positives) for probability bins.
    
    Returns:
        A list of dictionaries, one for each bin.
    """
    if len(probabilities) == 0:
        return []
        
    bins = np.linspace(0, 1, n_bins + 1)
    
    bin_totals = np.zeros(n_bins, dtype=int)
    bin_positives = np.zeros(n_bins, dtype=int)
    
    for prob, label in zip(probabilities, true_labels):
        # Find the bin index
        if prob == 1.0:
            bin_index = n_bins - 1 # Special case for 1.0, put in last bin
        else:
            # This maps 0.0-0.099 to 0, 0.1-0.199 to 1, ..., 0.9-0.999 to 9
            bin_index = int(prob * n_bins) 
        
        bin_totals[bin_index] += 1
        if label == 1:
            bin_positives[bin_index] += 1
    
    results = []
    for i in range(n_bins):
        total = bin_totals[i]
        positives = bin_positives[i]
        
        if total > 0:
            accuracy = positives / total
        else:
            accuracy = np.nan # No samples landed in this bin
        
        results.append({
            'bin_start': bins[i],
            'bin_end': bins[i+1],
            'total_samples': total,
            'correct_positives': positives, 
            'accuracy': accuracy
        })
        
    return results

def log_results_to_file(results, filepath="nas_exhaustive_report.txt"):
    """
    Logs the summary of the training results to a text file.
    Sorts by F1-Score and includes calibration data for the Top 20 models.
    """
    print(f"\nLogging results to '{filepath}'...")

    if not results:
        print("     No results to log.")
        return

    try:
        with open(filepath, 'w') as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"--- NAS Training Report ---\n")
            f.write(f"Generated: {now}\n")
            f.write(f"Total Training Runs: {len(results)}\n")
            f.write(f"Metrics below are for the 'Hit' (positive, label 1) class.\n")
            f.write("=" * 30 + "\n\n")

            avg_accuracy = sum(res['accuracy'] for res in results) / len(results)
            avg_f1 = sum(res['f1'] for res in results) / len(results)
            avg_precision = sum(res['precision'] for res in results) / len(results)
            avg_recall = sum(res['recall'] for res in results) / len(results)

            f.write(f"Average Model Accuracy: {avg_accuracy * 100:.2f}%\n")
            f.write(f"Average Model F1-Score: {avg_f1 * 100:.2f}%\n")
            f.write(f"Average Model Precision: {avg_precision * 100:.2f}%\n")
            f.write(f"Average Model Recall: {avg_recall * 100:.2f}%\n\n")

            best_model = max(results, key=lambda x: x['f1'])
            n, L, H, flops = best_model['arch']
            f.write("Highest F1-Score Run:\n")
            f.write(f"   - Arch (n, L, H, FLOPS): ({n}, {L}, {H}, {flops})\n")
            f.write(f"   - Features: {', '.join(best_model['feature_names'])}\n")
            f.write(f"   - F1-Score: {best_model['f1'] * 100:.2f}%\n")
            f.write(f"   - Accuracy: {best_model['accuracy'] * 100:.2f}%\n")
            f.write(f"   - Precision: {best_model['precision'] * 100:.2f}%\n")
            f.write(f"   - Recall: {best_model['recall'] * 100:.2f}%\n")
            
            f.write("   - Calibration Report (Accuracy = Fraction of Positives):\n")
            if 'calibration' in best_model and best_model['calibration']:
                for bin_res in best_model['calibration']:
                    # Format bin range, handling 1.0 case
                    bin_range = f"[{bin_res['bin_start']:.2f}, {bin_res['bin_end']:.2f}]"
                    if bin_res['bin_end'] < 1.0:
                         bin_range = f"[{bin_res['bin_start']:.2f}, {bin_res['bin_end']:.2f})"
                    
                    if bin_res['total_samples'] == 0:
                        acc_str = "N/A"
                    else:
                        acc_str = f"{bin_res['accuracy'] * 100:.1f}%"
                    
                    f.write(f"     {bin_range.ljust(13)}: "
                            f"Accuracy: {acc_str.ljust(6)} "
                            f"({bin_res['correct_positives']} / {bin_res['total_samples']} samples)\n")
            else:
                f.write("     (Calibration data not available)\n")
            f.write("\n")
            

            f.write("=" * 30 + "\n")
            f.write("Top 20 Runs by F1-Score (for 'Hit' class):\n") 
            f.write("=" * 30 + "\n")
            top_models = sorted(results, key=lambda x: x['f1'], reverse=True)

            for i, res in enumerate(top_models[:20]): 
                n, L, H, flops = res['arch']
                f.write(f"   Rank {i + 1}: Arch(n={n}, L={L}, H={H})\n")
                f.write(f"     On Features: {', '.join(res['feature_names'])}\n")
                f.write(
                    f"     - F1: {res['f1'] * 100:.2f}% | Acc: {res['accuracy'] * 100:.2f}% "
                    f"| P: {res['precision'] * 100:.2f}% | R: {res['recall'] * 100:.2f}%\n"
                )

                f.write("     - Calibration (Bin: Accuracy %):\n")
                if 'calibration' in res and res['calibration']:
                    try:
                        for j in range(5):
                            bin_res1 = res['calibration'][j]
                            acc_str1 = f"{bin_res1['accuracy'] * 100:.1f}%" if bin_res1['total_samples'] > 0 else "N/A"
                            range1 = f"[{bin_res1['bin_start']:.1f}-{bin_res1['bin_end']:.1f}]"
                            
                            bin_res2 = res['calibration'][j+5]
                            acc_str2 = f"{bin_res2['accuracy'] * 100:.1f}%" if bin_res2['total_samples'] > 0 else "N/A"
                            range2 = f"[{bin_res2['bin_start']:.1f}-{bin_res2['bin_end']:.1f}]"
                            
                            f.write(f"       {range1.ljust(9)}: {acc_str1.ljust(6)} | {range2.ljust(9)}: {acc_str2}\n")
                    except Exception:
                        f.write("       (Could not format calibration table)\n") 
                f.write("\n") 

        print(f"     Successfully wrote report to '{filepath}'.")

    except Exception as e:
        print(f"Error: Could not write report to file: {e}", file=sys.stderr)


# --- Main Execution ---
if __name__ == "__main__":

    DATA_FILE = 'triangles.bin'
    NAS_FILE = 'nas.txt'
    GRAPH_DIR = 'graphs' 

    TEST_SPLIT_SIZE = 0.2
    VALIDATION_SPLIT_SIZE = 0.2
    EPOCHS_PER_MODEL = 30
    PATIENCE = 5
    BATCH_SIZE = 512

    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5

    GRADIENT_CLIP_VALUE = 1.0
    NUM_WORKERS = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark is enabled.")

    pin_memory_flag = (device.type == "cuda")
    
    os.makedirs(GRAPH_DIR, exist_ok=True)
    print(f"Graphs will be saved to '{GRAPH_DIR}/'")


    triangle_data = read_triangle_samples(DATA_FILE)
    if not triangle_data:
        sys.exit(1)

    all_architectures = parse_nas_file(NAS_FILE)
    if not all_architectures:
        sys.exit(1)

    print(f"\nBuilding all feature combinations...")

    combos_by_sum = build_feature_combos_by_sum(min_sum=7, max_sum=64)
    
    
    total_runs = 0
    for arch in all_architectures:
        n = arch[0] 
        feature_sets_for_this_n = combos_by_sum.get(n, [])
        total_runs += len(feature_sets_for_this_n)
    
    print(f"Found {len(all_architectures)} architectures.")
    print(f"Found {sum(len(v) for v in combos_by_sum.values())} total feature combinations.")
    print(f"--> Preparing for {total_runs} total training runs. <--\n")

    print("Extracting full dataset and creating data splits...")

    y = np.array([s[57] for s in triangle_data], dtype=np.float32)
    X_full = np.array([s[:57] for s in triangle_data], dtype=np.float32)
    
    
    ray_dir_orig = X_full[:, 0:3] 
    normal = X_full[:, 14:17] # unit_normal
    dot_product = np.sum(ray_dir_orig * normal, axis=1, keepdims=True)
    X_full = np.hstack((X_full, dot_product.astype(np.float32)))
    print(f"Appended 'ray_normal_dot' (idx 57). Shape: {X_full.shape}")

    
    v0 = X_full[:, 3:6]
    v1 = X_full[:, 6:9]
    v2 = X_full[:, 9:12]
    ray_dir = X_full[:, 0:3] 

    e1 = v1 - v0
    e2 = v2 - v0
    
    tvec = -v0 
    
    pvec = np.cross(ray_dir, e2, axis=1)
    
    qvec = np.cross(tvec, e1, axis=1)

    determinant = np.sum(e1 * pvec, axis=1, keepdims=True)

    u_raw = np.sum(tvec * pvec, axis=1, keepdims=True)
    
    v_raw = np.sum(ray_dir * qvec, axis=1, keepdims=True)

    
    dot_v0 = np.sum(ray_dir * v0, axis=1, keepdims=True)
    
    
    dot_v1 = np.sum(ray_dir * v1, axis=1, keepdims=True)
    
    
    dot_v2 = np.sum(ray_dir * v2, axis=1, keepdims=True)

    X_full = np.hstack((
        X_full, 
        determinant.astype(np.float32),
        u_raw.astype(np.float32),
        v_raw.astype(np.float32),
        dot_v0.astype(np.float32),
        dot_v1.astype(np.float32),
        dot_v2.astype(np.float32)
    ))
    
    print(f"Appended 6 new 'cheap' features. New X_full shape: {X_full.shape}")

    lengths = np.linalg.norm(v0, axis=1, keepdims=True)
    scaled_ray = X_full[:, 0:3] * lengths 
    X_full[:, 0:3] = scaled_ray
    print("Scaled ray directions using distance to v0.")

    all_indices = np.arange(len(y))

    train_full_idx, test_idx = train_test_split(
        all_indices, test_size=TEST_SPLIT_SIZE, random_state=42, stratify=y
    )

    train_idx, val_idx = train_test_split(
        train_full_idx, test_size=VALIDATION_SPLIT_SIZE, random_state=42, stratify=y[train_full_idx]
    )

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_numpy = y_val
    y_test_numpy = y_test
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    #print(f"Data split (fixed): {len(train_idx)} train, {len(val_idx)} validation, {len(test_idx)} test samples.")

    miss_count = (y_train == 0).sum()
    hit_count = (y_train == 1).sum()

    print(f"Class distribution (train): {miss_count} Misses / {hit_count} Hits.")

    if hit_count > 0:
        pos_weight_value = miss_count / hit_count
        print(f"Using pos_weight: {pos_weight_value:.3f}")
    else:
        print("Warning: No 'Hit' samples found in training data. Using pos_weight = 1.0")
        pos_weight_value = 1.0

    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)


    results = []
    run_counter = 0

    
    for model_idx, arch in enumerate(all_architectures):
        n, L, H, flops = arch
        
        feature_sets_to_test = combos_by_sum.get(n, [])

        if not feature_sets_to_test:
            # print(f"\nInfo: Skipping Arch(n={n}, L={L}, H={H}), no feature sets found with dim {n}.")
            continue

        print("\n" + "=" * 80)
        print(f"Testing {len(feature_sets_to_test)} feature sets for Arch {model_idx + 1}/{len(all_architectures)} (n={n}, L={L}, H={H})")
        print("=" * 80)

        for feat_set_idx, combo in enumerate(feature_sets_to_test):

            current_indices = combo["indices"]
            current_feature_names = combo["features"]

            X_subset = X_full[:, current_indices]

            X_train = X_subset[train_idx]
            X_val = X_subset[val_idx]
            X_test = X_subset[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
                print("   WARNING: NaNs/Infs detected in X_train_scaled. Fixing.")
                X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            if np.isnan(X_val_scaled).any() or np.isinf(X_val_scaled).any():
                print("   WARNING: NaNs/Infs detected in X_val_scaled. Fixing.")
                X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            if np.isnan(X_test_scaled).any() or np.isinf(X_test_scaled).any():
                print("   WARNING: NaNs/Infs detected in X_test_scaled. Fixing.")
                X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=pin_memory_flag,
            )

            run_counter += 1

            print("\n" + "-" * 60)
            print(f"Run {run_counter}/{total_runs}: Arch {model_idx + 1} (n={n}, L={L}, H={H})")
            print(f"On Feature Set {feat_set_idx + 1}/{len(feature_sets_to_test)}: {', '.join(current_feature_names)}")
            print("-" * 60)

            model = create_model(n, L, H).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.2,
                patience=3,
                verbose=False 
            )

            best_val_loss = float('inf') 
            patience_counter = 0
            best_model_state = None

            for epoch in range(EPOCHS_PER_MODEL):
                model.train()
                epoch_loss = 0.0

                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    logits = outputs.squeeze(1)  
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
                    optimizer.step()
                    epoch_loss += loss.item()

                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor).squeeze(1)
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                # avg_train_loss = epoch_loss / len(train_loader)
                # print(f"  Epoch {epoch + 1:2d}/{EPOCHS_PER_MODEL} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1

            if best_model_state:
                model.load_state_dict(best_model_state)
            else:
                print("Warning: best_model_state was not set. Using final model state.")

            model.eval()
            with torch.no_grad():
                val_outputs_final = model(X_val_tensor).squeeze(1)
                val_probs = torch.sigmoid(val_outputs_final).cpu().numpy()

            best_threshold = 0.01
            best_f1_thr = -1.0

            for thr in np.linspace(0.05, 0.95, 19):
                val_pred_thr = (val_probs >= thr).astype(np.float32)
                _, _, f1_thr, _ = precision_recall_fscore_support(
                    y_val_numpy,
                    val_pred_thr,
                    pos_label=1,
                    average='binary',
                    zero_division=0
                )
                if f1_thr > best_f1_thr:
                    best_f1_thr = f1_thr
                    best_threshold = thr

            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor).squeeze(1)
                test_probs = torch.sigmoid(test_outputs).cpu().numpy()
                predicted_numpy = (test_probs >= best_threshold).astype(np.float32)

                correct = (predicted_numpy == y_test_numpy).sum().item()
                accuracy = correct / len(y_test_numpy)

                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test_numpy,
                    predicted_numpy,
                    pos_label=1,
                    average='binary',
                    zero_division=0
                )
                
                calibration_results = calculate_calibration_bins(test_probs, y_test_numpy, n_bins=10)

                print(f"-> Run {run_counter} Test Set Metrics (using threshold {best_threshold:.3f}):")
                print(f"   - Overall Accuracy: {accuracy * 100:.2f}%")
                print(f"   - Overall F1 (Hit): {f1 * 100:.2f}%")
                print(f"   - Overall P  (Hit): {precision * 100:.2f}%")
                print(f"   - Overall R  (Hit): {recall * 100:.2f}%")
                
                results.append({
                    'arch': arch,
                    'feature_names': current_feature_names,
                    'feature_indices': current_indices,
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'calibration': calibration_results,
                    'model_state': copy.deepcopy(best_model_state),
                    'scaler': copy.deepcopy(scaler)
                })

    print("\n" + "=" * 60)
    print("Calculating Efficacy (Average F1-Score) for each Feature:")
    print("=" * 60)
    
    feature_efficacy_stats = defaultdict(lambda: {'total_f1': 0.0, 'count': 0})
    all_feature_names = set()

    if not results:
        print("No results to analyze for feature efficacy.")
    else:
        for res in results:
            f1_score = res['f1']
            for feature_name in res['feature_names']:
                all_feature_names.add(feature_name)
                feature_efficacy_stats[feature_name]['total_f1'] += f1_score
                feature_efficacy_stats[feature_name]['count'] += 1

        efficacy_scores = []
        for name in sorted(list(all_feature_names)): 
            data = feature_efficacy_stats[name]
            if data['count'] > 0:
                average_f1 = data['total_f1'] / data['count']
                efficacy_scores.append({
                    'name': name,
                    'average_f1': average_f1,
                    'count': data['count']
                })
        
        efficacy_scores_sorted = sorted(efficacy_scores, key=lambda x: x['average_f1'], reverse=True)
        
        print("Feature Name".ljust(25) + "| " + "Avg. F1-Score".ljust(15) + "| " + "Models Used In")
        print("-" * 57)
        for score in efficacy_scores_sorted:
            f1_str = f"{score['average_f1'] * 100:.2f}%"
            print(f"{score['name'].ljust(25)}| {f1_str.ljust(15)}| {score['count']}")


    print("\n" + "=" * 60)
    print(f"Training Complete. Ran {len(results)} total experiments. Top 20 Runs by F1-Score:")
    print("=" * 60)

    top_models = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    print(f"\nGenerating guess distribution graphs for Top 20 models into '{GRAPH_DIR}' folder...")
    plot_filenames = []
    
    for i, res in enumerate(top_models[:20]): 
        try:
            n, L, H, flops = res['arch']
            
            model = create_model(n, L, H).to(device)
            if res['model_state'] is None:
                print(f"Warning: Skipping graph for Top {i+1} model, no model_state was saved.")
                continue
                
            model.load_state_dict(res['model_state'])
            model.eval()

            current_indices = res['feature_indices']
            scaler = res['scaler']
            
            X_subset = X_full[:, current_indices]
            X_test = X_subset[test_idx]
            
            X_test_scaled = scaler.transform(X_test)
            
            if np.isnan(X_test_scaled).any() or np.isinf(X_test_scaled).any():
                print(f"   Warning: NaNs/Infs detected in X_test_scaled for Top {i+1} model. Fixing.")
                X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                test_outputs = model(X_test_tensor).squeeze(1)
                test_probs = torch.sigmoid(test_outputs).cpu().numpy()

            probs_hits = test_probs[y_test_numpy == 1]
            probs_misses = test_probs[y_test_numpy == 0]
            
            mean_hits = np.mean(probs_hits) if len(probs_hits) > 0 else 0
            std_hits = np.std(probs_hits) if len(probs_hits) > 0 else 0
            mean_misses = np.mean(probs_misses) if len(probs_misses) > 0 else 0
            std_misses = np.std(probs_misses) if len(probs_misses) > 0 else 0

            n_bins = 20
            bin_hits = np.zeros(n_bins, dtype=int)
            bin_misses = np.zeros(n_bins, dtype=int)
            bins = np.linspace(0, 1, n_bins + 1)
            
            for prob, label in zip(test_probs, y_test_numpy):
                if prob == 1.0:
                    bin_index = n_bins - 1
                else:
                    bin_index = int(prob * n_bins)
                
                if label == 0: # Miss
                    bin_misses[bin_index] += 1
                else: # Hit
                    bin_hits[bin_index] += 1

            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            plt.figure(figsize=(12, 7))

            plt.plot(bin_centers, bin_hits, color='green', label=f'Hits (Label=1) - $\\mu={mean_hits:.3f}$', marker='o', linestyle='-')
            plt.plot(bin_centers, bin_misses, color='red', label=f'Misses (Label=0) - $\\mu={mean_misses:.3f}$', marker='x', linestyle='--')
            
            plt.xlabel('Model Output (Predicted Probability Bin)')
            plt.ylabel('Count of Guesses (Log Scale)')

            feat_str = ", ".join(res["feature_names"])
            if len(feat_str) > 50:
                feat_str = feat_str[:50] + "..."
            plt.title(f'Top {i + 1} Model: Overlaid Guess Distributions\nArch(n={n}, L={L}, H={H}) | F1: {res["f1"] * 100:.2f}% | Features: {feat_str}')
            
            plt.legend()
            plt.xticks(np.linspace(0, 1, 11)) 
            plt.xlim(0, 1)
            
            plt.yscale('log')
            plt.gca().set_ylim(bottom=1) 
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            filename = f'top_{i+1:02d}_model_distribution_lines.png' # Pad with zero

            save_path = os.path.join(GRAPH_DIR, filename)
            plt.savefig(save_path)
            plt.close() 
            plot_filenames.append(filename)

        except Exception as e:
            print(f"Error: Could not generate graph for Top {i+1} model: {e}", file=sys.stderr)
            
    if plot_filenames:
        print(f"Successfully generated {len(plot_filenames)} graphs in '{GRAPH_DIR}/'.")
    
    for i, res in enumerate(top_models[:20]): 
        n, L, H, flops = res['arch']
        print(f"\n     Rank {i + 1}: Arch(n={n}, L={L}, H={H})")
        print(f"     On Features: {', '.join(res['feature_names'])}")
        print(
            f"         - F1: {res['f1'] * 100:.2f}% | "
            f"Acc: {res['accuracy'] * 100:.2f}% | "
            f"P: {res['precision'] * 100:.2f}% | "
            f"R: {res['recall'] * 100:.2f}%"
        )
        if 'calibration' in res and res['calibration']:
            try:
                high_conf = res['calibration'][-3:] 
                print(f"         - High-Confidence Accuracy (Fraction of Positives):")
                for bin_res in high_conf:
                    bin_range = f"[{bin_res['bin_start']:.1f}-{bin_res['bin_end']:.1f}]"
                    acc_str = f"{bin_res['accuracy'] * 100:.1f}%" if bin_res['total_samples'] > 0 else "N/A"
                    print(f"           {bin_range}: {acc_str.ljust(6)} ({bin_res['total_samples']} samples)")
            except Exception:
                pass 
        print("") 

    log_results_to_file(results, filepath="nas_exhaustive_report.txt")

    print("\n--- Script Finished ---")