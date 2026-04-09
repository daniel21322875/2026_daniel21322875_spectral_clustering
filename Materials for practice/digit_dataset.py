import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score, silhouette_score

# ====================== 1. 載入 Digits 資料集 ======================
digits = load_digits()
X = digits.data       
y_true = digits.target

print(f"資料形狀: {X.shape}")  # (1797, 64)

# ====================== 2. 資料預處理 ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====================== 3. 用 Epsilon-ball 建立 adjacency matrix ======================
# 這裡用標準化後資料的 pairwise distance，並以距離分位數當作 epsilon
distances = pairwise_distances(X_scaled, metric='euclidean')
epsilon = np.percentile(distances[distances > 0], 2) #setting epsilon ball's distance

adjacency_matrix = (distances <= epsilon).astype(float)
np.fill_diagonal(adjacency_matrix, 0.0)

print(f"Epsilon: {epsilon:.4f}")
print(f"Adjacency matrix edge count: {int(adjacency_matrix.sum() / 2)}")

# ====================== 4. 建立並執行 Spectral Clustering ======================
spectral = SpectralClustering(
    n_clusters=10,                 
    affinity='precomputed',
    assign_labels='cluster_qr',      # clustering method "kmeans" or "discretize" or "cluster_qr"
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

plt.figure(figsize=(12, 5))

cmap_digits = plt.cm.get_cmap('tab10', 10)
digit_ticks = np.arange(10)

# 第一張：PCA 下的分群結果
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap_digits, s=12, alpha=0.75, linewidths=0)
plt.title('PCA - Clustering Result (Digits)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
cbar = plt.colorbar(scatter, ticks=digit_ticks)
cbar.set_label('Cluster ID (0-9)')

# 第二張：PCA 下的真實數據
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap=cmap_digits, s=12, alpha=0.75, linewidths=0)
plt.title('PCA - True Digits')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
cbar2 = plt.colorbar(scatter2, ticks=digit_ticks)
cbar2.set_label('True Digit (0-9)')

plt.tight_layout()
plt.show()

# ====================== 7. 視覺化：epsilon 百分位數 vs Adjusted Rand Score ======================
percentiles = np.arange(1, 101)
ari_scores = []
total_steps = len(percentiles)
bar_width = 30

for i, p in enumerate(percentiles, start=1):
    eps_p = np.percentile(distances[distances > 0], p)
    adj_p = (distances <= eps_p).astype(float)
    np.fill_diagonal(adj_p, 0.0)

    spectral_p = SpectralClustering(
        n_clusters=10,
        affinity='precomputed',
        assign_labels='kmeans',
        eigen_solver='arpack',
        random_state=42,
        n_init=10
    )

    labels_p = spectral_p.fit_predict(adj_p)
    ari_scores.append(adjusted_rand_score(y_true, labels_p))

    filled = int(bar_width * i / total_steps)
    bar = '█' * filled + '-' * (bar_width - filled)
    print(f"\r[進度] epsilon sweep |{bar}| {i}/{total_steps} ({i / total_steps * 100:5.1f}%)", end='', flush=True)

print()  # 進度條結束後換行

plt.figure(figsize=(8, 5))
plt.plot(percentiles, ari_scores, marker='o', markersize=3, linewidth=1)
plt.title('Adjusted Rand Score vs Epsilon-ball Percentile')
plt.xlabel("Epsilon-ball percentile")
plt.ylabel('Adjusted Rand Score')
plt.xlim(1, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
