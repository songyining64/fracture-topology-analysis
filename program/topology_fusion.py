# -*- coding: utf-8 -*-
"""
拓扑属性融合模块：将多个高级拓扑属性通过 PCA / 自编码器 / UMAP / VAE 融合成新属性。
用于机器学习扩展：输入为网格 CSV 中的拓扑指标，输出为融合主成分、簇标签等。
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

DEFAULT_FEATURE_COLUMNS = [
    "Fracture Intensity B21",
    "Fracture Intensity P21",
    "Dimensionless Intensity B22",
    "Dimensionless Intensity P22",
    "Areal Frequency B20",
    "Areal Frequency P20",
    "Connections per Branch",
    "Connections per Trace",
    "Connection Frequency",
    "Number of Traces (Real)",
    "Number of Branches (Real)",
    "Branch Mean Length",
    "Trace Mean Length",
    "Trace Min Length",
    "Trace Max Length",
]


def load_and_prepare(
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    drop_all_nan: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS
    available = [c for c in feature_columns if c in df.columns]
    if not available:
        raise ValueError(f"CSV 中未找到任何特征列，请检查列名: {feature_columns}")
    X = df[available].copy()
    X = X.fillna(0.0)
    if drop_all_nan:
        valid = (X != 0).any(axis=1)
        X = X.loc[valid]
        df = df.loc[valid].copy()
    return df, X.values.astype(np.float64), available


def fuse_with_pca(
    X: np.ndarray,
    n_components: int = 2,
    standardize: bool = True,
) -> Tuple[np.ndarray, Optional[StandardScaler], Optional[PCA]]:
    scaler = StandardScaler() if standardize else None
    if scaler is not None:
        X = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, scaler, pca


def cluster_labels(
    X_latent: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_latent)
    return labels, kmeans


def interpret_clusters(
    df: pd.DataFrame,
    feature_columns: List[str],
    cluster_id: np.ndarray,
    n_clusters: int,
) -> pd.DataFrame:
    df = df.copy()
    df["_cluster"] = cluster_id
    means = df.groupby("_cluster")[feature_columns].mean()
    df.drop(columns=["_cluster"], inplace=True)
    return means


def _check_min_data(df: pd.DataFrame, X: np.ndarray, used_cols: list, min_samples: int = 2, min_features: int = 2):
    n_samples, n_features = X.shape
    if n_samples < min_samples or n_features < min_features:
        raise ValueError(
            f"有效样本数或特征数不足（当前 {n_samples} 样本、{n_features} 特征）。"
            f"融合/聚类至少需要 {min_samples} 样本和 {min_features} 特征。请换用数据量更大的区域（如英买2区、KB11）。"
        )


def run_fusion_pipeline(
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    n_components: int = 2,
    n_clusters: int = 4,
) -> Tuple[pd.DataFrame, StandardScaler, PCA, KMeans, pd.DataFrame]:
    df, X, used_cols = load_and_prepare(csv_path, feature_columns=feature_columns)
    n_samples, n_features = X.shape
    if n_samples < 2 or n_features < 2:
        raise ValueError(
            f"有效样本数或特征数不足（当前 {n_samples} 样本、{n_features} 特征）。"
            f"PCA/聚类至少需要 2 样本和 2 特征。请换用数据量更大的区域（如英买2区、KB11），或先运行 export_grid_csv.py 生成网格 CSV。"
        )
    n_components = min(n_components, n_samples, n_features)
    X_pca, scaler, pca = fuse_with_pca(X, n_components=n_components, standardize=True)
    labels, kmeans = cluster_labels(X_pca, n_clusters=n_clusters)
    for i in range(n_components):
        df[f"PC{i+1}"] = X_pca[:, i]
    df["cluster_id"] = labels
    cluster_means = interpret_clusters(df, used_cols, labels, n_clusters)
    return df, scaler, pca, kmeans, cluster_means


def fuse_with_autoencoder(
    X: np.ndarray,
    n_latent: int = 2,
    standardize: bool = True,
    epochs: int = 100,
    lr: float = 1e-2,
) -> Tuple[np.ndarray, Optional[StandardScaler], Optional[object]]:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("自编码器需要 PyTorch，请执行: pip install torch")
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None
    n_features = X.shape[1]
    X_t = torch.tensor(X, dtype=torch.float32)

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, max(n_features // 2, n_latent)),
                nn.ReLU(),
                nn.Linear(max(n_features // 2, n_latent), n_latent),
            )
            self.decoder = nn.Sequential(
                nn.Linear(n_latent, max(n_features // 2, n_latent)),
                nn.ReLU(),
                nn.Linear(max(n_features // 2, n_latent), n_features),
            )
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    model = AE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        recon, _ = model(X_t)
        loss = ((X_t - recon) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, Z = model(X_t)
    return Z.numpy(), scaler, model


def run_fusion_pipeline_ae(
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    n_latent: int = 2,
    n_clusters: int = 4,
    ae_epochs: int = 100,
) -> Tuple[pd.DataFrame, StandardScaler, KMeans, pd.DataFrame]:
    df, X, used_cols = load_and_prepare(csv_path, feature_columns=feature_columns)
    _check_min_data(df, X, used_cols)
    Z, scaler, _ = fuse_with_autoencoder(X, n_latent=n_latent, standardize=True, epochs=ae_epochs)
    labels, kmeans = cluster_labels(Z, n_clusters=n_clusters)
    for i in range(n_latent):
        df[f"Z{i+1}"] = Z[:, i]
    df["cluster_id"] = labels
    cluster_means = interpret_clusters(df, used_cols, labels, n_clusters)
    return df, scaler, kmeans, cluster_means


def fuse_with_umap(
    X: np.ndarray,
    n_components: int = 2,
    standardize: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, Optional[StandardScaler], Optional[object]]:
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP 需要 umap-learn，请执行: pip install umap-learn")
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, max(2, len(X) - 1)),
        min_dist=min_dist,
        random_state=random_state,
    )
    X_umap = reducer.fit_transform(X)
    return X_umap, scaler, reducer


def run_fusion_pipeline_umap(
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    n_components: int = 2,
    n_clusters: int = 4,
    n_neighbors: int = 15,
) -> Tuple[pd.DataFrame, StandardScaler, object, KMeans, pd.DataFrame]:
    df, X, used_cols = load_and_prepare(csv_path, feature_columns=feature_columns)
    _check_min_data(df, X, used_cols)
    X_umap, scaler, reducer = fuse_with_umap(
        X, n_components=n_components, standardize=True, n_neighbors=n_neighbors
    )
    labels, kmeans = cluster_labels(X_umap, n_clusters=n_clusters)
    for i in range(n_components):
        df[f"U{i+1}"] = X_umap[:, i]
    df["cluster_id"] = labels
    cluster_means = interpret_clusters(df, used_cols, labels, n_clusters)
    return df, scaler, reducer, kmeans, cluster_means


def fuse_with_vae(
    X: np.ndarray,
    n_latent: int = 2,
    standardize: bool = True,
    epochs: int = 150,
    lr: float = 1e-3,
) -> Tuple[np.ndarray, Optional[StandardScaler], Optional[object]]:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("VAE 需要 PyTorch，请执行: pip install torch")
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None
    n_features = X.shape[1]
    X_t = torch.tensor(X, dtype=torch.float32)
    hidden = max(n_features // 2, n_latent)

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_fc1 = nn.Linear(n_features, hidden)
            self.enc_fc2 = nn.Linear(hidden, n_latent * 2)
            self.dec_fc1 = nn.Linear(n_latent, hidden)
            self.dec_fc2 = nn.Linear(hidden, n_features)

        def encode(self, x):
            h = torch.relu(self.enc_fc1(x))
            mu_logvar = self.enc_fc2(h)
            mu, logvar = mu_logvar[:, :n_latent], mu_logvar[:, n_latent:]
            return mu, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h = torch.relu(self.dec_fc1(z))
            return self.dec_fc2(h)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar, z

    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        recon, mu, logvar, _ = model(X_t)
        mse = ((X_t - recon) ** 2).mean()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        loss = mse + 0.5 * kl
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, _, _, Z = model(X_t)
    return Z.numpy(), scaler, model


def run_fusion_pipeline_vae(
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    n_latent: int = 2,
    n_clusters: int = 4,
    vae_epochs: int = 150,
) -> Tuple[pd.DataFrame, StandardScaler, KMeans, pd.DataFrame]:
    df, X, used_cols = load_and_prepare(csv_path, feature_columns=feature_columns)
    _check_min_data(df, X, used_cols)
    Z, scaler, _ = fuse_with_vae(X, n_latent=n_latent, standardize=True, epochs=vae_epochs)
    labels, kmeans = cluster_labels(Z, n_clusters=n_clusters)
    for i in range(n_latent):
        df[f"Z{i+1}"] = Z[:, i]
    df["cluster_id"] = labels
    cluster_means = interpret_clusters(df, used_cols, labels, n_clusters)
    return df, scaler, kmeans, cluster_means


if __name__ == "__main__":
    import sys
    csv_path = "Yingmai 2 area in Tarim Basin.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    df_out, scaler, pca, kmeans, means = run_fusion_pipeline(csv_path, n_components=2, n_clusters=4)
    print("PCA 融合后新增列:", [c for c in df_out.columns if c.startswith("PC") or c == "cluster_id"])
    out_path = csv_path.replace(".csv", "_fused.csv")
    df_out.to_csv(out_path, index=False)
    print("已写出:", out_path)
