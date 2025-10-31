import numpy as np
import math
import pandas as pd
from math import log


def euler_cluster(data, k, n1):
    # k: number of clusters; data: data containing station longitude/latitude, velocity, velocity uncertainty, station name, and cluster labels (object-type matrix)
    shape = data.shape
    EULER = []

    for m in range(0, k):
        index = np.where(data[:, 7] == m)  # Get data of a specific cluster
        velo = data[index]
        euler = solve_Euler(velo)  # Calculate the Euler pole for the cluster
        euler = np.transpose(euler)  # Transpose
        euler = list(euler)
        EULER.append(euler)

    V = np.empty((shape[0], 2 * k))  # Velocity derived from Euler pole (east and north components)
    V1 = np.empty((shape[0], k))  # Magnitude of velocity vector residual
    V2 = np.empty((shape[0]))
    s = np.empty((shape[0], k))
    yv = np.empty((shape[0], 2))  # Original velocity data
    for i in range(0, shape[0]):
        row = 0
        t = int(data[i, 7])
        for j in range(0, k):
            velo = data[i, :]  # Get data of the current station
            eu = EULER[j]  # Get the Euler pole of the j-th cluster
            eu = np.transpose(eu)
            v = solve_velocity(velo, eu)
            v = np.transpose(v)  # Estimated velocity
            V[i, row] = v[0] - data[i, 2]
            V[i, row + 1] = v[1] - data[i, 4]
            s[i, j] = (V[i, row] * V[i, row] + V[i, row + 1] * V[i, row + 1])
            V1[i, j] = math.sqrt(s[i, j])
            row = row + 2
        V2[i] = math.sqrt((s[i, t] / n1))
    sigma = sum(V2)  # Total residual


    redis = []  # Reallocated cluster labels
    for i in range(0, shape[0]):
        min_index = np.argmin(V1[i, :])
        redis.append(min_index)

    return redis, sigma


# Calculate velocity using station coordinates and Euler pole
def solve_velocity(velo, EULER):
    # Input: station data (longitude, latitude, velocity, etc.); Output: velocity estimated from Euler pole
    lon = velo[0]
    lat = velo[1]
    Ve = velo[2]
    Vn = velo[4]
    sigE = velo[3]
    sigN = velo[5]
    # Removed covariance (cov) reference: original 6th column is now station name, no need to read
    shape = velo.shape  # Get data length

    # Earth radius (unit: m)
    r = 6371393

    # Convert longitude and latitude to radians
    lamda = (lon * math.pi) / 180
    fai = (lat * math.pi) / 180

    # Euler vector formula: V = Ω × R (Ω is Euler vector, R is coefficient matrix)
    a = -r * math.cos(lamda) * math.sin(fai)
    b = -r * math.sin(lamda) * math.sin(fai)
    c = r * math.cos(fai)
    d = r * math.sin(lamda)
    e = -r * math.cos(lamda)
    f = 0

    # Assemble coefficient matrix R
    R = []
    R.append([a, b, c])
    R.append([d, e, f])
    V = np.dot(R, EULER)
    return V


# Solve Euler pole using least squares method
# velo: data containing station longitude/latitude, horizontal velocity, velocity uncertainty, station name, and cluster labels
def solve_Euler(velo):
    lon = velo[:, 0]
    lat = velo[:, 1]
    Ve = velo[:, 2]
    Vn = velo[:, 4]
    sigE = velo[:, 3]
    sigN = velo[:, 5]
    # Removed covariance (cov) reference: original 6th column is now station name, no need to read
    shape = velo.shape  # Get data length

    # Earth radius (unit: m)
    r = 6371393

    # Convert longitude and latitude to radians
    lamda = np.empty((shape[0], 1))
    fai = np.empty((shape[0], 1))
    for i in range(0, shape[0]):
        lamda[i] = (lon[i] * (math.pi)) / 180
        fai[i] = (lat[i] * (math.pi)) / 180

    # Assemble velocity field matrix (east + north components)
    Ven = []
    for i in range(0, shape[0]):
        Ven.append(Ve[i])
        Ven.append(Vn[i])

    # Calculate elements of coefficient matrix R
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    for j in range(0, shape[0]):
        a.append(-r * math.cos(lamda[j]) * math.sin(fai[j]))
        b.append(-r * math.sin(lamda[j]) * math.sin(fai[j]))
        c.append(r * math.cos(fai[j]))
        d.append(r * math.sin(lamda[j]))
        e.append(-r * math.cos(lamda[j]))
        f.append(0)

    # Assemble coefficient matrix R
    R = []
    for i in range(0, shape[0]):
        R.append([a[i], b[i], c[i]])
        R.append([d[i], e[i], f[i]])

    # Solve Euler pole using least squares
    Euler = np.linalg.lstsq(R, Ven, rcond=None)
    return Euler[0]  # Return Euler pole


# Generate a set of velocity field data with Gaussian (normal) distribution noise
# This function is suitable for real measured data
def syndata(data):
    shape = data.shape
    Mdata = np.empty((shape[0], 2))
    for i in range(0, shape[0]):
        sigmae = data[i, 2] * data[i, 2]
        sigman = data[i, 3] * data[i, 3]

        noisee = np.random.normal(0, sigmae, 1)
        noisen = np.random.normal(0, sigman, 1)
        de = data[i, 0] + float(noisee[0])
        dn = data[i, 1] + float(noisen[0])
        Mdata[i, 0] = np.round(de, 6)
        Mdata[i, 1] = np.round(dn, 6)  # Keep 6 decimal places
    return Mdata

# Generate synthetic data
# This function is suitable for simulated velocity fields.
# Since the simulated data lack real standard deviations,
# we randomly generate standard deviations in accordance with the standard deviation level of real velocity fields.
def syndata1(data):
    shape = data.shape
    Mdata = np.empty((shape[0], 2))
    for i in range(0, shape[0]):
        sigmae = 0.02
        sigman = 0.02
        noisee = np.random.normal(0, sigmae, 1)  # generates normal distribution
        noisen = np.random.normal(0, sigman, 1)
        de = data[i, 0] + noisee[0]
        dn = data[i, 1] + noisen[0]
        Mdata[i, 0] = de
        Mdata[i, 1] = dn
    return Mdata

# ----------------------
# 1. Data preparation (based on read Excel data)
# ----------------------
# Input data (.xlsx) format: longitude latitude east velocity sigmaE north velocity sigmaN station name initial label
input_filename = 'data.xlsx'
df = pd.read_excel(input_filename, engine='openpyxl', header=None)


# Extract data columns (updated data matrix structure)
# Data matrix structure: [longitude, latitude, east velocity (Ve), sigma_E, north velocity (Vn), sigma_N, station name, initial label]
# Use object type to support string (station name) storage
data = np.empty((len(df), 8), dtype=object)
data[:, 0] = df.iloc[:, 0].values  # Longitude
data[:, 1] = df.iloc[:, 1].values  # Latitude
data[:, 2] = df.iloc[:, 2].values  # East velocity (Ve)
data[:, 3] = df.iloc[:, 3].values  # sigma_E
data[:, 4] = df.iloc[:, 4].values  # North velocity (Vn)
data[:, 5] = df.iloc[:, 5].values # sigma_N
data[:, 6] = df.iloc[:, 6].values  # Station name
data[:, 7] = df.iloc[:, 7].values.astype(int)  # Initial label (converted to integer)

# Extract station names (for relation matrix and entropy results)
sites = data[:, 6]  # Directly obtain from the 6th column of the data matrix
n_stations = len(sites)  # Number of stations

# ----------------------
# 2. Key parameter settings
# ----------------------
k = int(data[:, 7].max() + 1)  # Number of clusters (derived from initial labels, where initial labels start from 0)
M = 100  # Number of synthetic data groups (for stability test)
n1 = 2  # Degrees of freedom (east + north velocity components)

# ----------------------
# 3. Initialize relation matrix
# ----------------------
# Relation matrix R_ij: records the number of times each pair of stations is assigned to the same cluster in M clustering trials
relation_matrix = np.zeros((n_stations, n_stations), dtype=int)

# ----------------------
# 4. Iterate to generate synthetic data and perform clustering
# ----------------------
for m in range(M):
    # 4.1 Generate synthetic velocity data with noise
    A = data[:, [2, 4, 3, 5]]  # Extract Ve, Vn, sigma_E, sigma_N
    synthetic_vel = syndata1(A)  # Call noise generation function; use syndata (for measured data) or syndata1 (for simulated data) according to the data type

    # Update velocity values in data (replace original velocity with synthetic velocity)
    data[:, 2] = synthetic_vel[:, 0]  # East velocity (Ve)
    data[:, 4] = synthetic_vel[:, 1]  # North velocity (Vn)

    # 4.2 Perform Euler Pole Clustering (EPC) to obtain new labels
    new_labels, _ = euler_cluster(data, k, n1)

    # 4.3 Update relation matrix: count co-cluster times for each station pair
    for i in range(n_stations):
        for j in range(i, n_stations):
            if new_labels[i] == new_labels[j]:
                if i == j:
                    relation_matrix[i, j] += 1
                else:
                    relation_matrix[i, j] += 1
                    relation_matrix[j, i] += 1

# ----------------------
# 6. Calculate entropy values
# ----------------------
# Initialize probability matrix P and entropy contribution matrix S
P = np.empty((n_stations, n_stations), dtype=float)  # P_ij = co-cluster times / total trials M
S = np.empty((n_stations, n_stations), dtype=float)  # Entropy contribution of each station pair

for i in range(n_stations):
    for j in range(n_stations):
        P[i, j] = relation_matrix[i, j] / M
        q = 1 - P[i, j]
        if q in (0, 1):  # Avoid log(0) error
            S[i, j] = 0
        else:
            S[i, j] = -P[i, j] * log(P[i, j], 2) - q * log(q, 2)
S = np.sum(S, axis=1)  # Entropy values


# ----------------------
# 7. Save entropy results
# ----------------------
# Integrate results: longitude + latitude + station name + entropy value
entropy_result = np.column_stack([
    data[:, 0].astype(float),  # Longitude
    data[:, 1].astype(float),  # Latitude
    sites,  # Station name
    S  # Entropy value
])

# Save to txt file (space-separated for easy reading)
entropy_output = f"EPC_entropy.txt"
np.savetxt(
    entropy_output,
    entropy_result,
    fmt='%10.6f %10.6f %-10s %10.6f',  # Format: longitude  latitude  entropy
)

print(f"Entropy calculation completed. Results saved to {entropy_output}")
print(f"Entropy range: {np.min(S):.6f} - {np.max(S):.6f}")