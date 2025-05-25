# Import library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Memuat dataset Iris
iris = load_iris()
X = iris.data  # fitur: [sepal length, sepal width, petal length, petal width]
y = iris.target  # label kelas

# Standardisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membangun PCA dan mengurangi dimensi ke 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Melihat seberapa besar varian yang dijelaskan oleh masing-masing komponen
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Visualisasi hasil PCA
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel("Komponen Utama 1")
plt.ylabel("Komponen Utama 2")
plt.title("Visualisasi PCA pada Dataset Iris")
plt.legend(*scatter.legend_elements(), title="Kelas")
plt.show()
