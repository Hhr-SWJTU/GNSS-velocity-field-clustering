import scipy.cluster.hierarchy as sch
import numpy as np
from sklearn import preprocessing

# --------------------------
# Step 1: Load data (execute only once)
# --------------------------
file="data.txt"
with open(file, "r", encoding="utf-8") as f:
    Ve = []  # velocity in e direction
    Vn = []  # velocity in n direction
    longitude = []  # longitude coordinates
    latitude = []  # latitude coordinates
    sites = []  # site names (supplement reading if available in the file)
    for line in f:
        line = line.strip()
        if not line:
            continue
        columns = line.split()
        # Adjust according to actual column indices in the file
        longitude.append(float(columns[0]))
        latitude.append(float(columns[1]))
        Ve.append(float(columns[2]))
        Vn.append(float(columns[4]))
        # To read site names, add: sites.append(columns[X]) where X is the column index of site names

# Organize data matrix
A = np.array([Ve, Vn])  # Original velocity data (rows: Ve/Vn, columns: samples)
B = preprocessing.minmax_scale(A, axis=1)  # Normalize row-wise (consistent with original logic)

# --------------------------
# Step 2: Define linkage methods to be tested
# --------------------------
# List of common HAC linkage methods (adjustable as needed)
method_list = ['single', 'complete', 'average', 'ward', 'median', 'centroid']
metric = 'euclidean'  # Distance metric
n_clusters = 2  # Number of clusters (adjustable as needed)

# --------------------------
# Step 3: Execute HAC clustering with multiple linkage methods in loop
# --------------------------
for method in method_list:
    print(f"\n{'=' * 20} Executing linkage method: {method} {'=' * 20}")

    # 1. Compute linkage matrix for hierarchical clustering
    try:
        # linkage requires a sample matrix (rows: samples, columns: features), thus using B.T (consistent with original code)
        hac_linkage = sch.linkage(B.T, metric=metric, method=method)
    except Exception as e:
        print(f"⚠️ Failed to compute linkage method {method}: {e}")
        continue

    # 2. Prune to specified number of clusters and obtain cluster labels
    # cut_tree returns an array of shape (n_samples, 1), need to flatten to 1D list
    labels = sch.cut_tree(hac_linkage, n_clusters=n_clusters).flatten()

    # 3. Print cluster assignment results (sample indices corresponding to cluster labels)
    print(f'Cluster assignments when number of clusters = {n_clusters}:')
    for cluster_id in sorted(list(set(labels))):
        # Get all sample indices of the current cluster
        sample_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        print(f"Cluster {cluster_id}: sample indices {sample_indices} (total {len(sample_indices)} samples)")

    # 4. Save results to file (filename includes linkage method to avoid overwriting)
    output_filename = f"HAC_{method}_label_1.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        # Line format: longitude latitude cluster label (append {sites[i]} if site names are available)
        for i in range(len(labels)):
            line = f"{longitude[i]} {latitude[i]} {labels[i]}\n"
            f.write(line)
    print(f"✅ Cluster labels saved to: {output_filename}")


print("\nClustering process for all linkage methods completed!")