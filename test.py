import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # still fine to import even if unused


SAMPLE_FORMAT = '<57fB3x'   
SAMPLE_SIZE   = struct.calcsize(SAMPLE_FORMAT) 


def read_binary_dataset(filename):
    """Read the binary dataset and return structured data."""
    samples = []

    with open(filename, 'rb') as f:
        while True:
            chunk = f.read(SAMPLE_SIZE)
            if len(chunk) < SAMPLE_SIZE:
                break

            # Unpack 57 floats + 1 label (uint8)
            data_tuple = struct.unpack(SAMPLE_FORMAT, chunk)

            floats = data_tuple[:-1]
            label  = int(data_tuple[-1])

            sample = {
                'ray_dir': np.array(floats[0:3]),
                'v0': np.array(floats[3:6]),
                'v1': np.array(floats[6:9]),
                'v2': np.array(floats[9:12]),
                'perimeter': floats[12],
                'area': floats[13],
                'unit_normal': np.array(floats[14:17]),
                'plane_eq': np.array(floats[17:21]),
                'orientation_sign': floats[21],
                'aspect_ratio': floats[22],
                # 'barycentric_condition': floats[23], # if you need it
                'centroid': np.array(floats[24:27]),
                # other features (circumcenter, etc.) are in the remaining floats
                'label': label
            }

            samples.append(sample)

    return samples


def analyze_dataset(samples):
    """Analyze the dataset for patterns."""
    labels = np.array([s['label'] for s in samples])

    if len(labels) == 0:
        print("No samples found. Cannot analyze.")
        return np.array([]), [], []

    hits = np.sum(labels == 1)
    misses = np.sum(labels == 0)
    total = len(samples)

    print("Dataset Statistics:")
    print("=" * 50)
    print(f"Total samples: {total}")
    print(f"Hits: {hits} ({100 * hits / total:.2f}%)")
    print(f"Misses: {misses} ({100 * misses / total:.2f}%)")

    if hits > 0:
        print(f"Hit/Miss ratio: 1:{misses / hits:.2f}")
    else:
        print("Hit/Miss ratio: N/A (0 hits)")
    print()

    hit_samples = [s for s in samples if s['label'] == 1]
    miss_samples = [s for s in samples if s['label'] == 0]

    if hit_samples:
        hit_areas = [s['area'] for s in hit_samples]
        hit_centroids = np.array([s['centroid'] for s in hit_samples])
        hit_distances = np.linalg.norm(hit_centroids, axis=1)

        print("Hit Triangle Statistics:")
        print(f"  Average area: {np.mean(hit_areas):.3f}")
        print(f"  Average distance from origin: {np.mean(hit_distances):.3f}")

    if miss_samples:
        miss_areas = [s['area'] for s in miss_samples]
        miss_centroids = np.array([s['centroid'] for s in miss_samples])
        miss_distances = np.linalg.norm(miss_centroids, axis=1)

        print("\nMiss Triangle Statistics:")
        print(f"  Average area: {np.mean(miss_areas):.3f}")
        print(f"  Average distance from origin: {np.mean(miss_distances):.3f}")

    return labels, hit_samples, miss_samples


def visualize_patterns(samples):
    """Create visualizations of the dataset."""
    if not samples:
        print("No samples to visualize.")
        return

    labels = np.array([s['label'] for s in samples])
    centroids = np.array([s['centroid'] for s in samples])
    areas = np.array([s['area'] for s in samples])
    distances = np.linalg.norm(centroids, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Dataset Analysis (57f format)', fontsize=16)

    # 1. Hit/Miss pie chart
    hits = np.sum(labels == 1)
    misses = np.sum(labels == 0)
    if hits > 0 or misses > 0:
        axes[0, 0].pie(
            [hits, misses],
            labels=['Hits', 'Misses'],
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c']
        )
    axes[0, 0].set_title('Hit vs Miss Distribution')

    # 2. Area vs distance, colored by label
    axes[0, 1].scatter(
        distances[labels == 0],
        areas[labels == 0],
        alpha=0.3,
        c='red',
        label='Miss',
        s=10
    )
    axes[0, 1].scatter(
        distances[labels == 1],
        areas[labels == 1],
        alpha=0.3,
        c='green',
        label='Hit',
        s=10
    )
    axes[0, 1].set_xlabel('Distance from Origin')
    axes[0, 1].set_ylabel('Triangle Area')
    axes[0, 1].set_title('Area vs Distance (Hit vs Miss)')
    axes[0, 1].legend()

    # 3. Distance histogram
    hist_data_dist = []
    hist_labels_dist = []
    hist_colors_dist = []
    if misses > 0:
        hist_data_dist.append(distances[labels == 0])
        hist_labels_dist.append('Miss')
        hist_colors_dist.append('red')
    if hits > 0:
        hist_data_dist.append(distances[labels == 1])
        hist_labels_dist.append('Hit')
        hist_colors_dist.append('green')

    if hist_data_dist:
        axes[1, 0].hist(
            hist_data_dist,
            bins=30,
            label=hist_labels_dist,
            color=hist_colors_dist,
            alpha=0.7,
            stacked=True
        )
    axes[1, 0].set_xlabel('Distance from Origin')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distance Distribution')
    axes[1, 0].legend()

    # 4. Area histogram
    hist_data_area = []
    hist_labels_area = []
    hist_colors_area = []
    if misses > 0:
        hist_data_area.append(areas[labels == 0])
        hist_labels_area.append('Miss')
        hist_colors_area.append('red')
    if hits > 0:
        hist_data_area.append(areas[labels == 1])
        hist_labels_area.append('Hit')
        hist_colors_area.append('green')

    if hist_data_area:
        axes[1, 1].hist(
            hist_data_area,
            bins=30,
            label=hist_labels_area,
            color=hist_colors_area,
            alpha=0.7,
            stacked=True
        )
    axes[1, 1].set_xlabel('Triangle Area')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Area Distribution')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('dataset_analysis_corrected.png', dpi=150)
    print("\nVisualization saved as 'dataset_analysis_corrected.png'")
    plt.show()


if __name__ == "__main__":
    try:
        print("Reading dataset with 57f / 232-byte struct size...")
        samples = read_binary_dataset('triangles.bin')

        if samples:
            print(f"Loaded {len(samples)} samples\n")
            labels, hit_samples, miss_samples = analyze_dataset(samples)
            visualize_patterns(samples)
        else:
            print("No samples were loaded. Please check 'triangles.bin' and the struct layout.")
    except FileNotFoundError:
        print("Error: triangles.bin not found. Please run the C++ code first to generate the dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")
