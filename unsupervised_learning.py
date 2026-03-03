# ===============================================
# UNSUPERVISED LEARNING - CHIP CLUSTERING
# INDUSTRIAL VISUALIZATION VERSION
# ===============================================

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Dataset
path = "/content/drive/MyDrive/chip_unsupervised.csv"
data = pd.read_csv(path)

print("Dataset Shape:", data.shape)
print(data.head())

# Feature Selection
X = data[['Frequency_MHz', 'Area_mm2', 'Power_Watts', 'Leakage_mW']]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# ---------------------------------------------------
# VISUALIZATION 1: Frequency vs Area (Clustered)
# ---------------------------------------------------
plt.figure()
plt.scatter(data['Frequency_MHz'], data['Area_mm2'], c=data['Cluster'])
plt.xlabel("Frequency (MHz)")
plt.ylabel("Area (mm²)")
plt.title("Chip Clusters Based on Design Parameters")
plt.show()

# ---------------------------------------------------
# VISUALIZATION 2: Power vs Leakage
# ---------------------------------------------------
plt.figure()
plt.scatter(data['Power_Watts'], data['Leakage_mW'], c=data['Cluster'])
plt.xlabel("Power (Watts)")
plt.ylabel("Leakage (mW)")
plt.title("Power vs Leakage by Cluster")
plt.show()

# ---------------------------------------------------
# VISUALIZATION 3: PCA Projection (2D Industrial View)
# ---------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Cluster Visualization using PCA")
plt.show()
