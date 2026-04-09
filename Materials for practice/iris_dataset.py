import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score, silhouette_score

# ====================== 1. 載入 Iris 資料集 ======================
iris = load_iris()
X = iris.data        
y_true = iris.target 

print(f"資料形狀: {X.shape}")  # (150, 4)

# ====================== 2. 資料預處理 ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====================== 3. 用 Epsilon-ball 建立 adjacency matrix ======================
# 這裡用標準化後資料的 pairwise distance，並以距離分位數當作 epsilon
distances = pairwise_distances(X_scaled, metric='euclidean')
epsilon = np.percentile(distances[distances > 0], 10) #setting epsilon ball's distance

adjacency_matrix = (distances <= epsilon).astype(float)
np.fill_diagonal(adjacency_matrix, 0.0)

print(f"Epsilon: {epsilon:.4f}")
print(f"Adjacency matrix edge count: {int(adjacency_matrix.sum() / 2)}")

# ====================== 4. 建立並執行 Spectral Clustering ======================
spectral = SpectralClustering(
    n_clusters=3,                   
    affinity='precomputed',          
    assign_labels='discretize',       # clustering method "kmeans" or "discretize" or "cluster_qr"
    eigen_solver='arpack',           
    random_state=42,                 
    n_init=10
)

labels = spectral.fit_predict(adjacency_matrix)

# ====================== 5. 評估結果 ======================
print("\n=== Spectral Clustering 結果 ===")
print(f"Adjusted Rand Score (與真實標籤比較): {adjusted_rand_score(y_true, labels):.4f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")

# 顯示每個類別的樣本數
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"集群 {u}: {c} 個樣本")
    
# ====================== 6. 視覺化 (PCA 與 Spectral Embedding 分別比較分群與真實標籤) ======================
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Spectral Embedding：將資料映射到二維
embedding = SpectralEmbedding(
    n_components=2,
    affinity='precomputed',
    random_state=42
)
X_se = embedding.fit_transform(adjacency_matrix)

plt.figure(figsize=(12, 10))

# 第一張：PCA 下的分群結果
plt.subplot(2, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=60, edgecolors='k')
plt.title('PCA - Clustering Result')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster')

# 第二張：PCA 下的真實數據
plt.subplot(2, 2, 2)
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', s=60, edgecolors='k')
plt.title('PCA - True Labels')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter2, label='True Species')

# 第三張：Embedding 到二維的結果圖
plt.subplot(2, 2, 3)
scatter3 = plt.scatter(X_se[:, 0], X_se[:, 1], c=labels, cmap='viridis', s=60, edgecolors='k')
plt.title('Spectral Embedding - Clustering Result')
plt.xlabel('Embedding Component 1')
plt.ylabel('Embedding Component 2')
plt.colorbar(scatter3, label='Cluster')

# 第四張：Embedding 到二維的真實數據
plt.subplot(2, 2, 4)
scatter4 = plt.scatter(X_se[:, 0], X_se[:, 1], c=y_true, cmap='viridis', s=60, edgecolors='k')
plt.title('Spectral Embedding - True Labels')
plt.xlabel('Embedding Component 1')
plt.ylabel('Embedding Component 2')
plt.colorbar(scatter4, label='True Species')

plt.tight_layout()
plt.show()


# ====================== 7. 視覺化：epsilon 百分位數 vs Adjusted Rand Score ======================
percentiles = np.arange(1, 101)
ari_scores = []

for p in percentiles:
    eps_p = np.percentile(distances[distances > 0], p)
    adj_p = (distances <= eps_p).astype(float)
    np.fill_diagonal(adj_p, 0.0)

    spectral_p = SpectralClustering(
        n_clusters=3,
        affinity='precomputed',
        assign_labels='kmeans',
        eigen_solver='arpack',
        random_state=42,
        n_init=10
    )

    labels_p = spectral_p.fit_predict(adj_p)
    ari_scores.append(adjusted_rand_score(y_true, labels_p))

plt.figure(figsize=(8, 5))
plt.plot(percentiles, ari_scores, marker='o', markersize=3, linewidth=1)
plt.title('Adjusted Rand Score vs Epsilon-ball Percentile')
plt.xlabel("Epsilon-ball percentile")
plt.ylabel('Adjusted Rand Score')
plt.xlim(1, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
