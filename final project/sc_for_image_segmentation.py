import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.cluster._spectral import cluster_qr
from sklearn.neighbors import kneighbors_graph
from PIL import Image
import os
import time
import time


def report_progress(message):
    print(f"[進度] {message}")


def rgb_to_cielab(rgb_data):
    """將 RGB 影像轉成 CIELAB (Lab) 色彩空間。"""
    rgb = rgb_data.astype(np.float64) / 255.0

    linear_rgb = np.empty_like(rgb, dtype=np.float64)
    low_mask = rgb <= 0.04045
    linear_rgb[low_mask] = rgb[low_mask] / 12.92
    linear_rgb[~low_mask] = ((rgb[~low_mask] + 0.055) / 1.055) ** 2.4

    # D65 white point, sRGB to XYZ
    rgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float64)
    xyz = linear_rgb @ rgb_to_xyz.T

    reference_white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz = xyz / reference_white

    delta = 6 / 29
    delta_cubed = delta ** 3

    def f(t):
        return np.where(t > delta_cubed, np.cbrt(t), (t / (3 * delta ** 2)) + (4 / 29))

    fx = f(xyz[..., 0])
    fy = f(xyz[..., 1])
    fz = f(xyz[..., 2])

    l = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    lab = np.stack([l, a, b], axis=-1)
    lab[..., 0] = lab[..., 0] / 100.0
    lab[..., 1] = (lab[..., 1] + 128.0) / 255.0
    lab[..., 2] = (lab[..., 2] + 128.0) / 255.0
    return np.clip(lab, 0.0, 1.0)

def build_similarity_graph(features, n_neighbors):
    """根據特徵建立對稱相似度圖。"""
    distance_graph = kneighbors_graph(
        features,
        n_neighbors=n_neighbors,
        mode='distance',
        include_self=False
    )
    report_progress("已完成建圖")

    nonzero_distances = distance_graph.data[distance_graph.data > 0]
    sigma = np.median(nonzero_distances) if nonzero_distances.size > 0 else 1.0
    gamma = 1.0 / (2.0 * sigma ** 2) if sigma > 0 else 1.0
    report_progress("已完成 gamma 計算")

    similarity_graph = distance_graph.copy().tocsr()
    similarity_graph.data = np.exp(-gamma * np.square(similarity_graph.data))
    similarity_graph = similarity_graph.maximum(similarity_graph.T)

    print(f"相似度 gamma: {gamma:.6f}")
    report_progress("已完成對稱化")

    return similarity_graph


def detect_connected_partitions(similarity_graph):
    """偵測稀疏相似度圖的連通分量數量與分配標籤。"""
    try:
        n_components, labels = connected_components(csgraph=similarity_graph, directed=False, connection='weak')
    except Exception as e:
        report_progress(f"連通性檢測失敗: {e}")
        return 1, None

    if n_components == 1:
        report_progress("圖為連通圖 (1 partition)")
    else:
        report_progress(f"圖包含 {n_components} 個連通分量 (partitions)")

    return n_components, labels

def compute_spectral_eigenpairs(similarity_graph, max_eigenvalues=10):
    """使用 normalized Laplacian 計算前幾個特徵值與特徵向量。"""
    # 盡可能使用稀疏運算以避免建立巨大的 dense 矩陣
    n_nodes = similarity_graph.shape[0]

    # 計算 degree
    degrees = np.asarray(similarity_graph.sum(axis=1)).ravel()
    degree_inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
    nonzero = degrees > 0
    degree_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

    D_inv_sqrt = sp.diags(degree_inv_sqrt)
    normalized_similarity = D_inv_sqrt.dot(similarity_graph).dot(D_inv_sqrt)
    laplacian = sp.eye(n_nodes, dtype=np.float64) - normalized_similarity

    eigen_count = min(max_eigenvalues, max(2, n_nodes - 1))

    if n_nodes <= 2:
        raise ValueError("節點數過少，無法進行分群")

    # 優先使用稀疏 eigsh，失敗時回退到 dense eigh
    try:
        eigenvalues, eigenvectors = eigsh(laplacian, k=eigen_count, which='SM', tol=1e-3)
    except Exception as e:
        report_progress(f"稀疏特徵分解失敗，改用 dense (原因: {e})")
        similarity_dense = normalized_similarity.toarray().astype(np.float64)
        lap_dense = np.eye(n_nodes, dtype=np.float64) - similarity_dense
        eigenvalues, eigenvectors = eigh(lap_dense, subset_by_index=[0, eigen_count - 1])
        print(f"（回退 dense）計算得特徵值: {np.array2string(eigenvalues, precision=6)}")

    # 確保特徵值從小到大排序
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"前 {eigen_count} 個特徵值: {np.array2string(eigenvalues, precision=6)}")
    report_progress("已完成特徵分解")

    if eigenvectors.shape[1] < eigen_count:
        raise ValueError("特徵向量數量不足，無法進行分群")

    return eigenvalues, eigenvectors

def plot_eigenvalue_bar_chart(eigenvalues):
    """輸出前幾個特徵值的長條圖。"""
    labels = [f'λ_{i + 1}' for i in range(len(eigenvalues))]
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    gaps = np.diff(eigenvalues)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, eigenvalues, color='#4c72b0', width=0.4)
    ax.set_title('smallest 10 eigenvalues')
    ax.set_ylabel('eigenvalue')
    ax.set_xlabel('λ_i')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    if gaps.size > 0:
        gap_positions = np.arange(len(gaps)) + 0.5
        gap_labels = [f'gap {i + 1}-{i + 2}' for i in range(len(gaps))]
        gap_ax = ax.twinx()
        gap_ax.plot(gap_positions, gaps, color='#dd8452', marker='o', linewidth=2, label='eigenvalue gap')
        gap_ax.set_ylabel('gap')
        gap_ax.set_xticks(gap_positions)
        gap_ax.set_xticklabels(gap_labels, rotation=45, ha='right')
        gap_ax.grid(False)

    plt.show()
    report_progress("已完成特徵值長條圖")
    plt.close()

    return None

def visualize_eigenvectors(eigenvectors, height, width, n_show=6):
    """將前幾個 eigenvector reshape 成影像並視覺化。"""
    n_show = min(n_show, eigenvectors.shape[1])
    if n_show <= 0:
        return None

    n_cols = min(3, n_show)
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for i in range(n_show):
        eig_img = eigenvectors[:, i].reshape(height, width)
        axes[i].imshow(eig_img, cmap='gray')
        axes[i].set_title(f'eig {i + 1}')
        axes[i].axis('off')

    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    report_progress("已完成 eigenvector 視覺化")
    plt.close()

    return None

def build_embedding_from_eigenvectors(eigenvectors, n_clusters, discard_n, use_constant_vector=False):
    """依照 n 值捨棄前 n 個特徵向量，建立 spectral embedding。"""
    start_index = discard_n
    required_vectors = n_clusters - 1 if use_constant_vector else n_clusters
    end_index = discard_n + required_vectors

    if required_vectors < 1:
        raise ValueError("分群數必須至少為 2")

    if end_index > eigenvectors.shape[1]:
        if use_constant_vector:
            raise ValueError(
                f"目前只計算前 {eigenvectors.shape[1]} 個特徵值，"
                f"無法捨棄前 {discard_n} 項後再取 {n_clusters - 1} 個 eigenvector 來建立 improve_qr。"
            )
        raise ValueError(
            f"目前只計算前 {eigenvectors.shape[1]} 個特徵值，"
            f"無法捨棄前 {discard_n} 項後再取 {n_clusters} 群所需的向量。請減少 n 或分群數。"
        )

    embedding = eigenvectors[:, start_index:end_index]

    if use_constant_vector:
        constant_vector = np.full((eigenvectors.shape[0], 1), 1.0 / np.sqrt(eigenvectors.shape[0]), dtype=np.float64)
        embedding = np.concatenate([embedding, constant_vector], axis=1)

    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    embedding = embedding / row_norms
    report_progress("已完成 spectral embedding")

    return embedding


def build_embedding_from_selected_eigenvectors(eigenvectors, selected_indices):
    """依照指定的 eigenvector 標籤建立 spectral embedding。"""
    if not selected_indices:
        raise ValueError("請至少指定一個 eigenvector 標籤")

    selected_indices = np.asarray(selected_indices, dtype=int)
    if np.any(selected_indices < 0) or np.any(selected_indices >= eigenvectors.shape[1]):
        raise ValueError(
            f"eigenvector 標籤必須介於 1 到 {eigenvectors.shape[1]} 之間"
        )

    embedding = eigenvectors[:, selected_indices]
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    embedding = embedding / row_norms
    report_progress("已完成指定 eigenvector 的 spectral embedding")

    return embedding

def qr_cluster_labels(embedding):
    """用 QR 分解對 spectral embedding 分群。"""
    labels = cluster_qr(embedding)
    report_progress("已完成 QR 分群")
    return labels


def kmeans_cluster_labels(embedding, n_clusters):
    """用 KMeans 對 spectral embedding 分群。"""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(embedding)
    report_progress("已完成 KMeans 分群")
    return labels

def prepare_spectral_image(image_path, output_dir='output', spatial_weight=0.3, n_neighbors=25):
    """讀取圖片並預先完成建圖與特徵值計算。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    report_progress("已確認輸出目錄")

    start_time = time.time()

    img = Image.open(image_path)
    img_array = np.array(img)
    report_progress("已讀取圖片")
    image_stem = os.path.splitext(os.path.basename(image_path))[0]

    original_height, original_width = img_array.shape[:2]

    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            rgb_data = img_array[:, :, :3]
        elif img_array.shape[2] == 3:
            rgb_data = img_array
        else:
            raise ValueError("圖片必須是 RGB 或 RGBA 格式")
    else:
        rgb_data = np.stack([img_array] * 3, axis=-1)

    lab_data = rgb_to_cielab(rgb_data)
    lab_pixels = lab_data.reshape(-1, lab_data.shape[2])

    y_coords, x_coords = np.indices((original_height, original_width))
    x_scaled = x_coords.astype(np.float64) / max(original_width - 1, 1)
    y_scaled = y_coords.astype(np.float64) / max(original_height - 1, 1)
    coords = np.stack([x_scaled, y_scaled], axis=-1).reshape(-1, 2)

    features = np.concatenate([lab_pixels, coords * spatial_weight], axis=1)
    report_progress("已完成特徵組合")

    print(f"圖片大小: {original_width}x{original_height}")
    print(f"總像素數: {lab_pixels.shape[0]}")
    print("正在進行譜聚類前處理...")

    similarity_graph = build_similarity_graph(features, n_neighbors=n_neighbors)
    # 偵測 kNN 圖是否為不連通，以及各節點的分配
    n_components, component_labels = detect_connected_partitions(similarity_graph)
    print(f"連通分量數量: {n_components}")
    if n_components > 1:
        print("注意：kNN 圖為不連通，分群可能會先反映各連通分量。")
    eigenvalues, eigenvectors = compute_spectral_eigenpairs(similarity_graph, max_eigenvalues=10)
    plot_eigenvalue_bar_chart(eigenvalues)
    visualize_eigenvectors(eigenvectors, original_height, original_width, n_show=6)

    return {
        'rgb_data': rgb_data,
        'original_height': original_height,
        'original_width': original_width,
        'start_time': start_time,
        'image_stem': image_stem,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'component_labels': component_labels,
    }

def spectral_cluster_image(
    prepared_data,
    n_clusters=2,
    discard_n=0,
    output_dir='output',
    cluster_method='cluster_qr',
    selected_eigenvector_labels=None,
):
    """
    使用譜聚類對圖片進行分群
    
    參數:
    - prepared_data: 預先計算好的圖片與特徵值資料
    - n_clusters: 分群數量（默認為2）
    - discard_n: 捨棄前 n 個特徵向量
    - output_dir: 輸出目錄
    - cluster_method: 分群方法，支援 'cluster_qr'、'improve_qr' 或 'kmeans'
    - selected_eigenvector_labels: kmeans 使用的 eigenvector 標籤（從 1 開始）
    """
    rgb_data = prepared_data['rgb_data']
    original_height = prepared_data['original_height']
    original_width = prepared_data['original_width']
    start_time = prepared_data['start_time']
    image_stem = prepared_data['image_stem']
    eigenvectors = prepared_data['eigenvectors']

    cluster_method = cluster_method.lower().strip()
    if cluster_method == 'cluster_qr':
        embedding = build_embedding_from_eigenvectors(eigenvectors, n_clusters=n_clusters, discard_n=discard_n)
        labels = qr_cluster_labels(embedding)
    elif cluster_method == 'improve_qr':
        embedding = build_embedding_from_eigenvectors(
            eigenvectors,
            n_clusters=n_clusters,
            discard_n=discard_n,
            use_constant_vector=True,
        )
        labels = qr_cluster_labels(embedding)
    elif cluster_method == 'kmeans':
        if selected_eigenvector_labels is None:
            raise ValueError("kmeans 模式需要指定 eigenvector 標籤")

        selected_indices = [label - 1 for label in selected_eigenvector_labels]
        embedding = build_embedding_from_selected_eigenvectors(eigenvectors, selected_indices)
        labels = kmeans_cluster_labels(embedding, n_clusters=n_clusters)
    else:
        raise ValueError("cluster_method 必須是 'cluster_qr'、'improve_qr' 或 'kmeans'")
    
    print(f"聚類完成！分為 {n_clusters} 群")
    report_progress("已完成分群")
    
    # 為每個群生成一張圖片
    for cluster_id in range(n_clusters):
        # 建立白色背景
        cluster_img = np.ones((original_height, original_width, 3), dtype=np.uint8) * 255
        
        # 取得屬於此群的像素索引
        cluster_mask = labels == cluster_id
        
        # 將屬於此群的像素還原到原圖片對應位置
        cluster_img_flat = cluster_img.reshape(-1, 3)
        cluster_img_flat[cluster_mask] = (rgb_data.reshape(-1, 3)[cluster_mask])
        cluster_img = cluster_img_flat.reshape(original_height, original_width, 3)
        
        # 儲存圖片
        output_path = os.path.join(output_dir, f'{image_stem}_cluster_{cluster_id}.png')
        Image.fromarray(cluster_img).save(output_path)
        print(f"已儲存: {output_path}")
        
        # 統計此群的像素數
        num_pixels = np.sum(cluster_mask)
        print(f"  - 像素數: {num_pixels} ({100*num_pixels/len(labels):.1f}%)")
        report_progress(f"已完成 cluster {cluster_id}")
    
    # 視覺化所有聚類結果
    fig, axes = plt.subplots(1, n_clusters + 1, figsize=(15, 5))
    
    # 原始圖片
    axes[0].imshow(rgb_data.astype(np.uint8))
    axes[0].set_title('original_image')
    axes[0].axis('off')
    
    # 每個聚類
    for cluster_id in range(n_clusters):
        cluster_img = np.ones((original_height, original_width, 3), dtype=np.uint8) * 255
        cluster_mask = labels == cluster_id
        cluster_img_flat = cluster_img.reshape(-1, 3)
        cluster_img_flat[cluster_mask] = (rgb_data.reshape(-1, 3)[cluster_mask])
        cluster_img = cluster_img_flat.reshape(original_height, original_width, 3)
        
        axes[cluster_id + 1].imshow(cluster_img.astype(np.uint8))
        axes[cluster_id + 1].set_title(f'cluster {cluster_id}')
        axes[cluster_id + 1].axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, f'{image_stem}_clustering_result.png')
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')
    print(f"已儲存視覺化結果: {viz_path}")
    report_progress("已完成視覺化輸出")
    plt.close()

    # 執行摘要
    elapsed = time.time() - start_time
    print("--- 執行摘要 ---")
    print(f"執行時間: {elapsed:.2f} 秒")
    print("----------------")


if __name__ == '__main__':
    image_path = input('請輸入圖片檔名或路徑: ').strip()
    if os.path.exists(image_path):
        prepared_data = prepare_spectral_image(image_path, output_dir='output')

        cluster_method = input('請選擇分群方法 (cluster_qr / improve_qr / kmeans): ').strip().lower()
        n_clusters_input = input('請輸入分群數(cluster數): ').strip()

        try:
            n_clusters = int(n_clusters_input)
        except ValueError:
            print('錯誤: cluster 數必須是整數')
        else:
            if cluster_method not in ('cluster_qr', 'improve_qr', 'kmeans'):
                print("錯誤: 分群方法只能是 cluster_qr、improve_qr 或 kmeans")
                raise SystemExit(1)

            if n_clusters < 2:
                print('錯誤: cluster 數必須大於或等於 2')
                raise SystemExit(1)

            discard_n = 0
            selected_eigenvector_labels = None

            if cluster_method == 'kmeans':
                labels_input = input('請輸入要使用的 eigenvector 標籤（例如 2, 4, 6）: ').strip()
                try:
                    selected_eigenvector_labels = [int(value.strip()) for value in labels_input.split(',') if value.strip()]
                except ValueError:
                    print('錯誤: eigenvector 標籤必須是整數，且以逗號分隔')
                    raise SystemExit(1)

                if not selected_eigenvector_labels:
                    print('錯誤: 至少要指定一個 eigenvector 標籤')
                    raise SystemExit(1)

                if any(label < 1 for label in selected_eigenvector_labels):
                    print('錯誤: eigenvector 標籤必須從 1 開始')
                    raise SystemExit(1)
            else:
                n_input = input('請輸入 n 值（捨棄前 n 項特徵值）: ').strip()
                try:
                    discard_n = int(n_input)
                except ValueError:
                    print('錯誤: n 值必須是整數')
                    raise SystemExit(1)

                if discard_n < 0:
                    print('錯誤: n 值不能小於 0')
                    raise SystemExit(1)

            spectral_cluster_image(
                prepared_data,
                n_clusters=n_clusters,
                discard_n=discard_n,
                output_dir='output',
                cluster_method=cluster_method,
                selected_eigenvector_labels=selected_eigenvector_labels,
            )
    else:
        print(f"錯誤: 找不到圖片 {image_path}")
        print('請確保圖片存在，或輸入正確的檔案路徑')
