
# Final Project Code: Spectral Clustering for Image Segmentation

This folder contains the final implementation of the project **Spectral Clustering for Image Segmentation**, along with weekly development notes from the final stage of the project.

The main program file is:

```text
sc_for_image_segmentation.py
```

This program reads an input image, treats each pixel as a data point, builds a similarity graph based on Lab color features and spatial coordinates, applies spectral clustering, and outputs segmented image results for each cluster.

---

## File Structure

```text
.
├── Note_for_week_12.pdf
├── Note_for_week_14.pdf
├── Note_for_week_15.pdf
├── sc_for_image_segmentation.py
└── readme.md
```

---

## Files

### `sc_for_image_segmentation.py`

This is the final version of the image segmentation program.

The program includes the following main functions:

1. Read the input image.
2. Convert RGB color space to Lab color space.
3. Add pixel coordinates as spatial features.
4. Build a sparse distance graph using k-nearest neighbors.
5. Convert the distance graph into a Gaussian similarity graph.
6. Check whether the kNN graph is connected.
7. Construct the normalized graph Laplacian.
8. Use `eigsh` to compute the first few smallest eigenvalues and eigenvectors.
9. Visualize eigenvalues and eigenvectors.
10. Build the spectral embedding from selected eigenvectors.
11. Support three clustering methods: `cluster_qr`, `improve_qr`, and `kmeans`.
12. Output each cluster as an individual image.
13. Output an overview image containing the original image and all cluster results.
14. Display the total execution time.

---

### `Note_for_week_12.pdf`

This note records the development progress during week 12, including output images, experiment observations, problems encountered, and possible improvements.

### `Note_for_week_14.pdf`

This note records the development progress during week 14, including further testing results, code adjustments, segmentation result comparisons, and implementation issues.

### `Note_for_week_15.pdf`

This note records the development progress during week 15, including near-final experiment results, final adjustments, and observations about the clustering results.

---

## Requirements

Please install the required Python packages before running the program:

```bash
pip install numpy scipy scikit-learn matplotlib pillow
```

The main packages used in this program are:

```text
numpy
matplotlib
scipy
scikit-learn
Pillow
```

---

## How to Run

Run the program in this folder:

```bash
python sc_for_image_segmentation.py
```

After running the program, it will ask for the following inputs:

1. Image filename or image path
2. Clustering method
3. Number of clusters
4. Number of eigenvectors to discard, or selected eigenvector labels

---

## Input Instructions

### 1. Input Image Path

The program first asks for the image filename or image path:

```text
Please enter the image filename or path:
```

Example:

```text
test.png
```

or:

```text
images/test.png
```

If the image exists, the program will start preprocessing, build the similarity graph, compute eigenvalues and eigenvectors, and display eigenvalue and eigenvector visualizations.

---

### 2. Select Clustering Method

The program then asks for the clustering method:

```text
Please select clustering method (cluster_qr / improve_qr / kmeans):
```

The program supports three clustering methods:

```text
cluster_qr
improve_qr
kmeans
```

#### `cluster_qr`

This method uses QR clustering on the spectral embedding.

It selects eigenvectors according to the number of clusters and uses them to build the spectral embedding before clustering.

#### `improve_qr`

This is the improved QR clustering method used in this project.

When the target is to divide the image into `N` clusters, the program uses `N-1` eigenvectors and additionally adds a constant vector. This allows QR clustering to separate `N` clusters in a lower-dimensional embedding space.

The constant vector used in the program is:

```python
constant_vector = np.full((eigenvectors.shape[0], 1), 1.0 / np.sqrt(eigenvectors.shape[0]))
```

#### `kmeans`

This method uses KMeans to cluster the spectral embedding built from selected eigenvectors.

In this mode, the user needs to manually specify which eigenvectors should be used.

---

### 3. Enter the Number of Clusters

The program asks for the number of clusters:

```text
Please enter the number of clusters:
```

Example:

```text
2
```

This means the image will be divided into 2 clusters.

---

### 4. Enter `n` Value or Eigenvector Labels

If `cluster_qr` or `improve_qr` is selected, the program asks for the `n` value:

```text
Please enter n value, which means discarding the first n eigenvectors:
```

Example:

```text
1
```

This means the first eigenvector will be discarded, and the spectral embedding will be built from the following eigenvectors.

If `kmeans` is selected, the program asks for selected eigenvector labels:

```text
Please enter the eigenvector labels, for example: 2, 4, 6
```

Example:

```text
2,4,6
```

This means the program will use the 2nd, 4th, and 6th eigenvectors to build the spectral embedding, then apply KMeans clustering.

---

## Program Workflow

```text
Input image
    ↓
Convert RGB to Lab color space
    ↓
Add spatial coordinate features
    ↓
Build k-nearest neighbor graph
    ↓
Convert distance graph to similarity graph
    ↓
Check graph connectivity
    ↓
Construct normalized graph Laplacian
    ↓
Compute eigenvalues and eigenvectors using eigsh
    ↓
Visualize eigenvalues and eigenvectors
    ↓
Build spectral embedding
    ↓
Cluster by cluster_qr / improve_qr / kmeans
    ↓
Output segmented images
```

---

## Output

The program automatically creates or uses the `output/` folder and stores the results there.

The output files follow this format:

```text
output/
├── image_name_cluster_0.png
├── image_name_cluster_1.png
├── ...
└── image_name_clustering_result.png
```

### Individual Cluster Images

Each cluster is saved as a separate image.

Pixels belonging to the selected cluster keep their original colors, while pixels outside the cluster are shown as a white background.

### Clustering Result Overview

The file:

```text
image_name_clustering_result.png
```

shows the original image and all cluster results together for comparison.

---

## Example

Run the program:

```bash
python sc_for_image_segmentation.py
```

Example input:

```text
Please enter the image filename or path: test.png
Please select clustering method (cluster_qr / improve_qr / kmeans): improve_qr
Please enter the number of clusters: 2
Please enter n value, which means discarding the first n eigenvectors: 1
```

Possible output:

```text
output/test_cluster_0.png
output/test_cluster_1.png
output/test_clustering_result.png
```

---

## Main Parameters

### `spatial_weight`

This parameter controls the influence of spatial coordinates on clustering.

Default value:

```python
spatial_weight = 0.3
```

A larger value gives more importance to spatial distance.  
A smaller value makes the clustering result depend more on color features.

---

### `n_neighbors`

This parameter controls the number of neighbors used in the kNN graph.

Default value:

```python
n_neighbors = 25
```

If `n_neighbors` is too small, the graph may become disconnected.  
If `n_neighbors` is too large, the computation time may increase and unnecessary long-distance connections may appear.

---

### `n_clusters`

This parameter represents the number of clusters used for image segmentation.

---

### `discard_n`

This parameter represents how many eigenvectors should be discarded before building the spectral embedding.

In spectral clustering, the first few eigenvectors may contain constant-vector information or may not be useful for segmentation. Therefore, this parameter allows the user to adjust which eigenvectors are used.

---

## Notes

- If the image resolution is too large, the number of pixels will increase, making graph construction and eigenvalue decomposition slower.
- If the kNN graph is disconnected, the clustering result may mainly reflect connected components.
- The final result is affected by several parameters, including:
  - clustering method
  - number of clusters
  - `spatial_weight`
  - `n_neighbors`
  - `discard_n`
  - selected eigenvector labels

If the segmentation result is not ideal, try adjusting these parameters.

---

## Weekly Notes

The weekly notes in this folder record experiment results, output images, problems encountered during implementation, and possible solutions.

These notes are used as supporting materials for the final project report and help show the development process of the final code.

---

## Author

林琮詠
