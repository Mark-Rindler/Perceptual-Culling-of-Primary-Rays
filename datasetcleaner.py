#!/usr/bin/env python3
"""
Balance a binary triangle dataset to achieve 40% hits / 60% misses ratio.

Memory-efficient: uses mmap and index-based sampling instead of loading
all samples into RAM.

Given:
- hits (y) should be 40% of total: 0.4 * total = y
- misses should be 60% of total: 0.6 * total = misses_needed

Solving:
- total = y / 0.4 = 2.5 * y
- misses_needed = 0.6 * total = 1.5 * y

So we keep all hits and randomly sample 1.5x hits worth of misses.

Binary format per sample: 57 float32 + 1 uint8 label + 3 pad bytes = 232 bytes
"""

import mmap
import random
import sys
import os
import argparse
import hashlib
from typing import Dict, List, Tuple, Optional


SAMPLE_SIZE = 232  # bytes per sample
FEATURE_COUNT = 57
FEATURE_BYTES = FEATURE_COUNT * 4  # 228 bytes


def scan_dataset(
    filepath: str,
    dedup_mode: str
) -> Tuple[List[int], List[int], int, int, int]:
    """
    First pass: scan dataset to collect indices of hits and misses.
    Memory efficient - only stores integer indices (plus hash dict if deduping).
    
    Args:
        filepath: path to binary file
        dedup_mode: 'none', 'exact', or 'features'
    
    Returns:
        hit_indices: list of sample indices with label=1
        miss_indices: list of sample indices with label=0
        invalid_count: number of samples with invalid labels
        duplicate_count: number of duplicate samples found
        conflict_count: number of samples with same features but different labels
    """
    file_size = os.path.getsize(filepath)
    
    if file_size % SAMPLE_SIZE != 0:
        remainder = file_size % SAMPLE_SIZE
        print(f"Error: File size {file_size} not divisible by {SAMPLE_SIZE}. "
              f"Trailing {remainder} bytes suggest corruption or partial write.",
              file=sys.stderr)
        sys.exit(1)
    
    num_samples = file_size // SAMPLE_SIZE
    print(f"Scanning {filepath} ({num_samples:,} samples)...")
    
    if dedup_mode != 'none':
        print(f"Deduplication mode: {dedup_mode}")
    
    hit_indices: List[int] = []
    miss_indices: List[int] = []
    invalid_count = 0
    duplicate_count = 0
    conflict_count = 0
    
    # For dedup: hash -> (first_index, label)
    # Only allocated if dedup is enabled
    seen: Optional[Dict[bytes, Tuple[int, int]]] = {} if dedup_mode != 'none' else None
    
    with open(filepath, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for i in range(num_samples):
                offset = i * SAMPLE_SIZE
                sample_bytes = mm[offset:offset + SAMPLE_SIZE]
                
                # Label is at byte 228 (after 57 float32s)
                label = sample_bytes[FEATURE_BYTES]
                
                # Validate label first
                if label not in (0, 1):
                    invalid_count += 1
                    if invalid_count <= 10:
                        print(f"Warning: Sample {i} has invalid label {label}, skipping.",
                              file=sys.stderr)
                    elif invalid_count == 11:
                        print(f"Warning: Suppressing further invalid label warnings...",
                              file=sys.stderr)
                    continue
                
                # Handle deduplication
                if seen is not None:
                    if dedup_mode == 'exact':
                        # Hash all 232 bytes
                        h = hashlib.md5(sample_bytes).digest()
                    else:  # dedup_mode == 'features'
                        # Hash only feature bytes
                        h = hashlib.md5(sample_bytes[:FEATURE_BYTES]).digest()
                    
                    if h in seen:
                        first_idx, first_label = seen[h]
                        
                        if dedup_mode == 'features' and first_label != label:
                            # Same features, different labels - this is a conflict
                            conflict_count += 1
                            if conflict_count <= 10:
                                print(f"Warning: Sample {i} has same features as sample "
                                      f"{first_idx} but different label "
                                      f"({label} vs {first_label}), dropping both.",
                                      file=sys.stderr)
                            elif conflict_count == 11:
                                print(f"Warning: Suppressing further conflict warnings...",
                                      file=sys.stderr)
                            
                            # Remove the first sample from its list
                            if first_label == 1:
                                try:
                                    hit_indices.remove(first_idx)
                                except ValueError:
                                    pass  # Already removed from a previous conflict
                            else:
                                try:
                                    miss_indices.remove(first_idx)
                                except ValueError:
                                    pass
                            
                            # Mark as conflicted so future dupes also get dropped
                            seen[h] = (first_idx, -1)  # -1 = conflicted
                            continue
                        
                        elif dedup_mode == 'features' and first_label == -1:
                            # This hash was already marked as conflicted
                            conflict_count += 1
                            continue
                        
                        else:
                            # Exact duplicate (or same features+label)
                            duplicate_count += 1
                            continue
                    
                    seen[h] = (i, label)
                
                # Add to appropriate list
                if label == 1:
                    hit_indices.append(i)
                else:
                    miss_indices.append(i)
        finally:
            mm.close()
    
    if invalid_count > 0:
        print(f"Warning: Skipped {invalid_count:,} samples with invalid labels (not 0 or 1).",
              file=sys.stderr)
    
    if duplicate_count > 0:
        print(f"Note: Skipped {duplicate_count:,} duplicate samples.")
    
    if conflict_count > 0:
        print(f"Warning: Dropped {conflict_count:,} samples due to label conflicts "
              f"(same features, different labels).", file=sys.stderr)
    
    return hit_indices, miss_indices, invalid_count, duplicate_count, conflict_count


def compute_balanced_indices(
    hit_indices: List[int],
    miss_indices: List[int],
    hit_ratio: float,
    seed: int = None
) -> Tuple[List[int], List[int]]:
    """
    Compute which indices to include in balanced dataset.
    
    Returns:
        selected_hits: all hit indices
        selected_misses: sampled miss indices
    """
    if seed is not None:
        random.seed(seed)
    
    num_hits = len(hit_indices)
    num_misses = len(miss_indices)
    total_original = num_hits + num_misses
    
    if total_original == 0:
        print("Error: No valid samples after filtering!", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nOriginal distribution (after filtering):")
    print(f"  Hits:   {num_hits:,} ({100*num_hits/total_original:.2f}%)")
    print(f"  Misses: {num_misses:,} ({100*num_misses/total_original:.2f}%)")
    
    # Calculate required misses: misses_needed = num_hits * (1 - hit_ratio) / hit_ratio
    miss_ratio = 1.0 - hit_ratio
    misses_needed = round(num_hits * miss_ratio / hit_ratio)
    
    print(f"\nTarget distribution ({hit_ratio*100:.0f}% hits / {miss_ratio*100:.0f}% misses):")
    print(f"  Hits:   {num_hits:,}")
    print(f"  Misses needed: {misses_needed:,}")
    
    if misses_needed > num_misses:
        print(f"\nWarning: Need {misses_needed:,} misses but only have {num_misses:,}.",
              file=sys.stderr)
        print(f"Using all available misses. Actual hit ratio will be higher.",
              file=sys.stderr)
        selected_misses = miss_indices[:]
    else:
        # Randomly sample misses (without replacement - guaranteed unique)
        selected_misses = random.sample(miss_indices, misses_needed)
    
    return hit_indices, selected_misses


def write_balanced_dataset(
    input_path: str,
    output_path: str,
    hit_indices: List[int],
    miss_indices: List[int],
    seed: int = None
):
    """
    Second pass: read selected samples and write to output.
    Shuffles the output order.
    """
    if seed is not None:
        random.seed(seed)
    
    # Combine and shuffle indices
    all_indices = hit_indices + miss_indices
    random.shuffle(all_indices)
    
    num_hits = len(hit_indices)
    num_misses = len(miss_indices)
    total = len(all_indices)
    
    actual_hit_ratio = num_hits / total
    print(f"\nFinal distribution:")
    print(f"  Total:  {total:,}")
    print(f"  Hits:   {num_hits:,} ({100*actual_hit_ratio:.2f}%)")
    print(f"  Misses: {num_misses:,} ({100*(1-actual_hit_ratio):.2f}%)")
    
    print(f"\nWriting {total:,} samples to {output_path}...")
    
    # Stream from input to output using mmap for reads
    with open(input_path, 'rb') as f_in:
        with mmap.mmap(f_in.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            with open(output_path, 'wb') as f_out:
                for i, sample_idx in enumerate(all_indices):
                    offset = sample_idx * SAMPLE_SIZE
                    # Write directly from mmap (still copies, but minimal)
                    f_out.write(mm[offset:offset + SAMPLE_SIZE])
                    
                    # Progress indicator for large datasets
                    if (i + 1) % 100000 == 0:
                        print(f"  Written {i+1:,}/{total:,} samples...")
    
    file_size = os.path.getsize(output_path)
    expected_size = total * SAMPLE_SIZE
    
    if file_size != expected_size:
        print(f"Error: Output size mismatch! Got {file_size}, expected {expected_size}",
              file=sys.stderr)
        sys.exit(1)
    
    print(f"Wrote {file_size:,} bytes")


def validate_hit_ratio(value: str) -> float:
    """Validate and parse hit ratio argument."""
    try:
        ratio = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid number: {value}")
    
    if ratio <= 0 or ratio >= 1:
        raise argparse.ArgumentTypeError(
            f"hit-ratio must be between 0 and 1 (exclusive), got {ratio}"
        )
    
    return ratio


def main():
    parser = argparse.ArgumentParser(
        description="Balance triangle dataset to target hit/miss ratio"
    )
    parser.add_argument("input", help="Input binary file")
    parser.add_argument("-o", "--output", default="balanced.bin",
                        help="Output binary file (default: balanced.bin)")
    parser.add_argument("--hit-ratio", type=validate_hit_ratio, default=0.4,
                        help="Target hit ratio, must be in (0,1) (default: 0.4 = 40%%)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--dedup", choices=['none', 'exact', 'features'], default='none',
                        help="Deduplication mode: 'none' (default, fastest), "
                             "'exact' (hash all 232 bytes), "
                             "'features' (hash features only, detect label conflicts)")
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # First pass: scan for indices
    hit_indices, miss_indices, invalid_count, dup_count, conflict_count = scan_dataset(
        args.input, args.dedup
    )
    
    if not hit_indices:
        print("Error: No valid hits found in dataset!", file=sys.stderr)
        sys.exit(1)
    
    if not miss_indices:
        print("Error: No valid misses found in dataset!", file=sys.stderr)
        sys.exit(1)
    
    # Compute balanced selection
    selected_hits, selected_misses = compute_balanced_indices(
        hit_indices, miss_indices, args.hit_ratio, args.seed
    )
    
    # Second pass: write output
    write_balanced_dataset(
        args.input, args.output,
        selected_hits, selected_misses,
        args.seed
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()