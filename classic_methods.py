import numpy as np
from utils import eva


def run_dbscan(img, gt):
    """Run cuML DBSCAN on hyperspectral image."""
    try:
        import cupy as cp
        from cuml.cluster import DBSCAN
    except Exception as e:
        print("cuML not available:", e)
        return
    X = cp.asarray(img.reshape(-1, img.shape[2]).astype(np.float32))
    labels = DBSCAN().fit_predict(X)
    eva(gt.reshape(-1), cp.asnumpy(labels))


def run_kmeans(img, gt, n_clusters):
    """Run cuML KMeans."""
    try:
        import cupy as cp
        from cuml.cluster import KMeans
    except Exception as e:
        print("cuML not available:", e)
        return
    X = cp.asarray(img.reshape(-1, img.shape[2]).astype(np.float32))
    labels = KMeans(n_clusters=n_clusters).fit_predict(X)
    eva(gt.reshape(-1), cp.asnumpy(labels))


def run_pca_kmeans(img, gt, n_clusters, n_components=30):
    """Run cuML PCA followed by KMeans."""
    try:
        import cupy as cp
        from cuml.cluster import KMeans
        from cuml.decomposition import PCA
    except Exception as e:
        print("cuML not available:", e)
        return
    X = cp.asarray(img.reshape(-1, img.shape[2]).astype(np.float32))
    pca = PCA(n_components=min(n_components, img.shape[2]))
    X_pca = pca.fit_transform(X)
    labels = KMeans(n_clusters=n_clusters).fit_predict(X_pca)
    eva(gt.reshape(-1), cp.asnumpy(labels))


def run_classic_methods(img, gt, n_clusters):
    """Evaluate several classic ML methods."""
    print("\n=== cuML DBSCAN ===")
    run_dbscan(img, gt)
    print("\n=== cuML KMeans ===")
    run_kmeans(img, gt, n_clusters)
    print("\n=== cuML PCA + KMeans ===")
    run_pca_kmeans(img, gt, n_clusters)
