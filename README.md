# GNSS-velocity-field-clustering
Source code and data for the Paper" Comparative Evaluation of Clustering Algorithms for GNSS Velocity Field Analysis: Insights from Synthetic Datasets and Southern California Tectonic Motion"


This project consists of two folders: the **code** folder contains scripts for GNSS velocity field clustering analysis, and the **data** folder includes the Southern California velocity field and simulated velocity fields.

### Code Folder Description  
1. `EPC.py`, `HAC.py`, and `k-means.py` are scripts for calculating the clustering results of three algorithms for GNSS velocity fields, respectively. **Note**: The input velocity field for `EPC.py` must include K-means cluster labels, which should be placed in the last column.  
2. `Entropy1.py` is used to compute the entropy values of clustering results from the K-means and HAC algorithms.  
3. `Entropy2.py` is designed to calculate the entropy values of clustering results from the EPC algorithm. **Note**: Similarly, the input velocity field here must include K-means cluster labels.  

### Data Folder Description  
1. `GPS velocity data in the Southern California.xlsx` is the GPS velocity field of Southern California, USA.  
2. All other files are simulated velocity fields:  
   - `vT400.xlsx` corresponds to Scheme a in Table 2,  
   - `vTn400.xlsx` corresponds to Scheme b in Table 2,  
   - `vT50.xlsx` corresponds to Scheme c in Table 2,  
   - `vTn50.xlsx` corresponds to Scheme d in Table 2,  
   - `Vtt1400.xlsx` corresponds to Scheme e in Table 2,  
   - `vtt1n400.xlsx` corresponds to Scheme f in Table 2,  
   - `vtt150.xlsx` corresponds to Scheme g in Table 2,  
   - `vtt1n50.xlsx` corresponds to Scheme h in Table 2.  
3. All velocity fields follow the unified format below:  
   Lon (°E)    Lat (°N)    vel_E (mm/a)    Sig_E (mm/a)    vel_N (mm/a)    Sig_N (mm/a)    Station  
