import pandas as pd
import numpy
from sklearn import preprocessing
from sklearn.cluster import KMeans
from math import log, exp
import scipy.cluster.hierarchy as sch


# Calculate entropy of clustering results to characterize the impact of noise on clustering
# Generate synthetic data
# This function is suitable for real velocity fields.
def syndata(data):
    shape = data.shape
    Mdata = numpy.empty((shape[0], 2))
    for i in range(0, shape[0]):
        sigmae = data[i, 3] * data[i, 3]
        sigman = data[i, 5] * data[i, 5]
        noisee = numpy.random.normal(0, sigmae, 1)  # generates normal distribution
        noisen = numpy.random.normal(0, sigman, 1)
        de = data[i, 2] + noisee[0]
        dn = data[i, 4] + noisen[0]
        Mdata[i, 0] = de
        Mdata[i, 1] = dn
    return Mdata


# Generate synthetic data
# This function is suitable for simulated velocity fields.
#Since the simulated data lack real standard deviations,
# we randomly generate standard deviations in accordance with the standard deviation level of real velocity fields.
def syndata1(data):
    shape = data.shape
    Mdata = numpy.empty((shape[0], 2))
    for i in range(0, shape[0]):
        sigmae = 0.02
        sigman = 0.02
        noisee = numpy.random.normal(0, sigmae, 1)  # generates normal distribution
        noisen = numpy.random.normal(0, sigman, 1)
        de = data[i, 2] + noisee[0]
        dn = data[i, 4] + noisen[0]
        Mdata[i, 0] = de
        Mdata[i, 1] = dn
    return Mdata


# Import velocity field data
input_filename = 'data.xlsx'
df = pd.read_excel(input_filename, engine='openpyxl', header=None)

data1 = numpy.zeros((len(df), 7))
data1 = data1.astype(object)
data1[:, 0] = df.iloc[:, 0].values  # Longitude (1st column)
data1[:, 1] = df.iloc[:, 1].values  # Latitude (2nd column)
data1[:, 2] = df.iloc[:, 2].values  # East velocity (Ve, 3rd column)
data1[:, 3] = df.iloc[:, 3].values  # sigma_E (4th column)
data1[:, 4] = df.iloc[:, 4].values  # North velocity (Vn, 5th column)
data1[:, 5] = df.iloc[:, 5].values  # sigma_N (6th column)
data1[:, 6] = df.iloc[:, 6].astype(str).values

# Define parameters
shape = data1.shape
R = numpy.zeros((shape[0], shape[0]), dtype=int)  # Relationship matrix
k = 3  # Number of clusters
sdata = []  # Synthetic data
M = 100  # Number of synthetic trials

for t in range(M):
    sdata = syndata1(data1)
    scaleddata1 = preprocessing.scale(sdata)  # Standardize data
    # K-means clustering
    cluster1 = KMeans(n_clusters=k, n_init=10, max_iter=1000).fit(scaleddata1)
    out1 = cluster1.labels_

    #  #HAC hierarchical clustering (using scipy's linkage and cut_tree)
    # metric = 'euclidean'  # distance metric (compatible with HAC)
    # method = 'ward'  # linkage method (compatible with HAC)
    # hac_linkage = sch.linkage(scaleddata1, metric=metric, method=method)
    # out1 = sch.cut_tree(hac_linkage, n_clusters=k).flatten()

    # Update matrix with results
    for i in range(shape[0]):
        for j in range(i,shape[0]):
            if out1[i] == out1[j]:
                if i == j:
                    R[i, j] += 1
                else:
                    R[i, j] += 1
                    R[j, i] += 1

# Calculate entropy values
P = numpy.empty((shape[0], shape[0]), dtype=float)
S = numpy.empty((shape[0], shape[0]), dtype=float)
for i in range(shape[0]):
    for j in range(shape[0]):
        P[i, j] = R[i, j] / M
        q = 1 - P[i, j]
        # Avoid log(0) error (when P=0 or P=1)
        if q in (0.0, 1.0):
            S[i, j] = 0.0
        else:
            S[i, j] = -P[i, j] * log(P[i, j], 2) - q * log(q, 2)
S = numpy.sum(S, axis=1)  # Total entropy for each station


# Output results
result = data1[:, :2]
result = numpy.column_stack((result, S))  # Combine coordinates and entropy
numpy.savetxt("Entropy.txt", result, fmt='%f')  # Save results
print("finish")
