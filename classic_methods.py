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


def run_autoencoder_kmeans(img, gt, n_clusters, hidden_dim=128, epochs=50,
                           batch_size=1024, lr=1e-3):
    """Train a simple autoencoder and cluster the latent features using KMeans."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.cluster import KMeans

    X = img.reshape(-1, img.shape[2]).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class AE(nn.Module):
        def __init__(self, in_dim, hid_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, in_dim)
            )

        def forward(self, x):
            z = self.encoder(x)
            x_rec = self.decoder(z)
            return x_rec, z

    model = AE(X.shape[1], hidden_dim).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(X)),
                        batch_size=batch_size, shuffle=True)
    optm = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for batch, in loader:
            batch = batch.to(device)
            optm.zero_grad()
            out, _ = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optm.step()

    model.eval()
    feats = []
    with torch.no_grad():
        for batch, in DataLoader(TensorDataset(torch.from_numpy(X)),
                                 batch_size=batch_size):
            _, z = model(batch.to(device))
            feats.append(z.cpu().numpy())
    feats = np.concatenate(feats, axis=0)

    labels = KMeans(n_clusters=n_clusters, n_init="auto").fit_predict(feats)
    _evaluate(gt, labels)


def run_classic_methods(img, gt, n_clusters):
    """Evaluate several classic ML clustering methods."""
    print("\n=== DBSCAN ===")
    run_dbscan(img, gt)
    print("\n=== KMeans ===")
    run_kmeans(img, gt, n_clusters)
    print("\n=== PCA + KMeans ===")
    run_pca_kmeans(img, gt, n_clusters)
    print("\n=== Autoencoder + KMeans ===")
    run_autoencoder_kmeans(img, gt, n_clusters)