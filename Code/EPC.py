# Euler vector clustering, returns reallocated cluster labels
import numpy as np
import pandas as pd
import math


def euler_cluster(data, k, n1):
    # k: number of clusters; data: array containing station coordinates, velocities, velocity uncertainties, site names, and k-means cluster labels
    shape = data.shape
    EULER = []  # Store Euler poles for each cluster

    for m in range(k):
        # Get data points belonging to the m-th cluster
        index = np.where(data[:, 7] == m)
        velo = data[index]
        # Calculate Euler pole for the cluster
        euler = solve_Euler(velo)
        euler = np.transpose(euler)
        euler = list(euler)
        EULER.append(euler)

    # V: Estimated velocity residuals (east and north components) for each cluster
    V = np.empty((shape[0], 2 * k))
    # V1: Magnitude of velocity residual vector for each cluster
    V1 = np.empty((shape[0], k))
    V2 = np.empty((shape[0]))
    s = np.empty((shape[0], k))  # Squared residual sum
    yv = np.empty((shape[0], 2))  # Original velocity data

    for i in range(shape[0]):
        row = 0
        t = int(data[i, 7])  # Current cluster label
        for j in range(k):
            velo = data[i, :]  # Get data for the i-th station
            eu = EULER[j]  # Get Euler pole of the j-th cluster
            eu = np.transpose(eu)
            # Estimate velocity using the Euler pole
            v = solve_velocity(velo, eu)
            v = np.transpose(v)
            # Calculate residuals (estimated - observed)
            V[i, row] = v[0] - data[i, 2]
            V[i, row + 1] = v[1] - data[i, 4]
            # Squared sum of residuals
            s[i, j] = (V[i, row] ** 2 + V[i, row + 1] ** 2)
            # Magnitude of residual vector
            V1[i, j] = math.sqrt(s[i, j])
            row += 2
        # Residual statistic (normalized by degrees of freedom)
        V2[i] = math.sqrt((s[i, t] / n1))

    sigma = sum(V2)  # Total residual sum

    # Reallocate clusters based on minimum residual
    redis = []  # Reallocated cluster labels
    for i in range(shape[0]):
        # Find cluster index with minimum residual
        min_index = np.argmin(V1[i, :])
        redis.append(min_index)

    return redis, sigma


# Calculate velocity using station coordinates and Euler pole
def solve_velocity(velo, EULER):
    # Input: station data (including coordinates); Output: velocity estimated from Euler pole
    lon = velo[0]  # Longitude
    lat = velo[1]  # Latitude
    Ve = velo[2]  # East component of velocity
    Vn = velo[4]  # North component of velocity
    sigE = velo[3]  # Uncertainty of east velocity
    sigN = velo[5]  # Uncertainty of north velocity
    # 移除协方差参数，第六列现在是站点名

    # Earth radius (meters)
    r = 6371393

    # Convert degrees to radians
    lamda = (lon * math.pi) / 180  # Longitude in radians
    fai = (lat * math.pi) / 180  # Latitude in radians

    # Euler vector formula: V = Ω × R, where Ω is Euler vector and R is coefficient matrix
    # Calculate elements of coefficient matrix R
    a = -r * math.cos(lamda) * math.sin(fai)
    b = -r * math.sin(lamda) * math.sin(fai)
    c = r * math.cos(fai)
    d = r * math.sin(lamda)
    e = -r * math.cos(lamda)
    f = 0

    # Construct coefficient matrix R
    R = []
    R.append([a, b, c])
    R.append([d, e, f])
    # Calculate velocity via matrix multiplication
    V = np.dot(R, EULER)
    return V


# Solve Euler pole using least squares method
# velo: array containing station coordinates, horizontal velocities, uncertainties, site names, and cluster labels
def solve_Euler(velo):
    lon = velo[:, 0]  # Longitude array
    lat = velo[:, 1]  # Latitude array
    Ve = velo[:, 2]  # East velocity array
    Vn = velo[:, 4]  # North velocity array
    sigE = velo[:, 3]  # East velocity uncertainty array
    sigN = velo[:, 5]  # North velocity uncertainty array
    # 移除协方差数组，第六列现在是站点名

    shape = velo.shape  # Number of data points

    # Earth radius (meters)
    r = 6371393

    # Convert degrees to radians
    lamda = np.empty((shape[0], 1))  # Longitude in radians
    fai = np.empty((shape[0], 1))  # Latitude in radians
    for i in range(shape[0]):
        lamda[i] = (lon[i] * math.pi) / 180
        fai[i] = (lat[i] * math.pi) / 180

    # Combine east and north velocities into a single vector
    Ven = []
    for i in range(shape[0]):
        Ven.append(Ve[i])
        Ven.append(Vn[i])

    # Calculate elements of coefficient matrix R for Euler vector equation
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    for j in range(shape[0]):
        a.append(-r * math.cos(lamda[j]) * math.sin(fai[j]))
        b.append(-r * math.sin(lamda[j]) * math.sin(fai[j]))
        c.append(r * math.cos(fai[j]))
        d.append(r * math.sin(lamda[j]))
        e.append(-r * math.cos(lamda[j]))
        f.append(0)

    # Construct full coefficient matrix R
    R = []
    for i in range(shape[0]):
        R.append([a[i], b[i], c[i]])
        R.append([d[i], e[i], f[i]])

    # Solve Euler vector using least squares
    Euler = np.linalg.lstsq(R, Ven, rcond=None)
    return Euler[0]  # Return estimated Euler pole


# Input file name
input_filename = 'data.xlsx'
df = pd.read_excel(input_filename, engine='openpyxl', header=None)

# Extract data columns (updated format with site name as 6th column)
# New data matrix structure: [longitude, latitude, east velocity (Ve), sigma_E,
#                           north velocity (Vn), sigma_N, site_name, initial label]
data = np.empty((len(df), 8), dtype=object)
data[:, 0] = df.iloc[:, 0].values  # Longitude (1st column)
data[:, 1] = df.iloc[:, 1].values  # Latitude (2nd column)
data[:, 2] = df.iloc[:, 2].values  # East velocity (Ve, 3rd column)
data[:, 3] = df.iloc[:, 3].values  # sigma_E (4th column)
data[:, 4] = df.iloc[:, 4].values  # North velocity (Vn, 5th column)
data[:, 5] = df.iloc[:, 5].values  # sigma_N (6th column)
data[:, 6] = df.iloc[:, 6].values  # Site name (7th column, string type)
data[:, 7] = df.iloc[:, 7].values.astype(int)  # Initial labels (8th column, converted to integers)

sites = data[:, 6]  # Station names (6th column in data matrix)
n_stations = len(sites)  # Number of stations

# 2. Key parameter settings
# ----------------------
k = int(data[:, 7].max() + 1)  # Number of clusters (derived from k-means initial labels; initial labels start from 0)
M = 100  # Number of synthetic data groups (recommended M=100 in literature for stability test)
n1 = 2  # Degrees of freedom (east + north velocity components, consistent with residual calculation)

# 4.2 Perform Euler vector clustering to obtain new labels
labels, _ = euler_cluster(data, k, n1)  # Call user-defined Euler Pole Clustering (EPC) function

# Save results to a text file
output_filename = 'EPC_label.txt'
with open(output_filename, "w", encoding="utf-8") as output_file:
    for i in range(len(labels)):
        # Include site name in output
        lon = data[i, 0]  # Longitude
        lat = data[i, 1]  # Latitude
        site_name = data[i, 6]  # Site name
        label = labels[i]  # Reallocated cluster label
        output_line = f"{lon:<12} {lat:<12} {site_name:<10} {label}\n"
        output_file.write(output_line)
print(f"Euler Pole Clustering completed successfully. Results saved to {output_filename}")