from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np

# Read raw data and store complete line information
original_lines = []  # Store raw line data from input file
longitude = []       # List of longitude coordinates
latitude = []        # List of latitude coordinates
sites = []           # List of site names
Ve = []              # List of velocity components in e-direction
Vn = []              # List of velocity components in n-direction

#  velocity field data
input_filename = "data.txt"
with open(input_filename, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        original_lines.append(line)  # Preserve raw line for reference
        columns = line.split()
        # Extract geospatial and velocity data from columns
        longitude.append(float(columns[0]))
        latitude.append(float(columns[1]))
        Ve.append(float(columns[2]))
        Vn.append(float(columns[4]))
        sites.append(columns[6])

# Data normalization
A = np.array([Ve, Vn])  # Construct velocity matrix (rows: e/n components, columns: samples)
B = preprocessing.minmax_scale(A, axis=1)  # Row-wise min-max normalization

# KMeans clustering implementation
n_clusters = 2  # Number of clusters
# Initialize KMeans with specified clusters, fixed random state for reproducibility, and sufficient initializations
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=100).fit(B.T)
labels = kmeans.labels_  # Retrieve cluster assignment labels

# Generate output filenames using input prefix
prefix = input_filename.split('.')[0]  # Extract filename prefix (without extension)

# Write coordinate data with cluster labels to output file
output_filename = f"kmeans_geospace_{prefix}.txt"
with open(output_filename, "w", encoding="utf-8") as output_file:
    for i in range(len(original_lines)):
        # Format: longitude (left-aligned) + latitude (left-aligned) + cluster label
        output_line = f"{longitude[i]:<15}{latitude[i]:<15} {labels[i]}\n"
        output_file.write(output_line)

# Write velocity data with cluster labels to output file
output_filename1 = f"kmeans_velo_{prefix}.txt"
with open(output_filename1, "w", encoding="utf-8") as output_file:
    for i in range(len(original_lines)):
        # Format: e-velocity (left-aligned) + n-velocity (left-aligned) + cluster label
        output_line = f"{Ve[i]:<15}{Vn[i]:<15} {labels[i]}\n"
        output_file.write(output_line)

print("Task completed")