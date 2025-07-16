import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


def _evaluate(gt, pred):
    """Print OA, NMI, AMI, ARI and FMI for predictions."""
    y_true = gt.reshape(-1)
    y_pred = pred.reshape(-1)

    # Hungarian matching for overall accuracy
    classes_true = np.unique(y_true)
    classes_pred = np.unique(y_pred)
    cost = np.zeros((classes_true.size, classes_pred.size), dtype=np.int64)
    for i, c1 in enumerate(classes_true):
        mask = y_true == c1
        for j, c2 in enumerate(classes_pred):
            cost[i, j] = np.sum(y_pred[mask] == c2)
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapped = np.zeros_like(y_pred)
    for i, j in zip(row_ind, col_ind):
        mapped[y_pred == classes_pred[j]] = classes_true[i]

    oa = metrics.accuracy_score(y_true, mapped)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    fmi = metrics.fowlkes_mallows_score(y_true, y_pred)
    print(f":OA {oa:.4f}, NMI {nmi:.4f}, AMI {ami:.4f}, ARI {ari:.4f}, FMI {fmi:.4f}")


def run_dbscan(img, gt):
    """Run DBSCAN using scikit-learn."""
    from sklearn.cluster import DBSCAN

    X = img.reshape(-1, img.shape[2]).astype(np.float32)
    labels = DBSCAN().fit_predict(X)
    _evaluate(gt, labels)


def run_kmeans(img, gt, n_clusters):
    """Run KMeans using scikit-learn."""
    from sklearn.cluster import KMeans

    X = img.reshape(-1, img.shape[2]).astype(np.float32)
    labels = KMeans(n_clusters=n_clusters, n_init="auto").fit_predict(X)
    _evaluate(gt, labels)


def run_pca_kmeans(img, gt, n_clusters, n_components=30):
    """Run PCA for dimensionality reduction followed by KMeans."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    X = img.reshape(-1, img.shape[2]).astype(np.float32)
    pca = PCA(n_components=min(n_components, img.shape[2]))
    X_pca = pca.fit_transform(X)
    labels = KMeans(n_clusters=n_clusters, n_init="auto").fit_predict(X_pca)
    _evaluate(gt, labels)


def run_classic_methods(img, gt, n_clusters):
    """Evaluate several classic ML clustering methods."""
    print("\n=== DBSCAN ===")
    run_dbscan(img, gt)
    print("\n=== KMeans ===")
    run_kmeans(img, gt, n_clusters)
    print("\n=== PCA + KMeans ===")
    run_pca_kmeans(img, gt, n_clusters)