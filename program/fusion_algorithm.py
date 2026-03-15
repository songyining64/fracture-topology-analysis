# -*- coding: utf-8 -*-
"""
智能融合算法：加权融合（专家规则） + GAT 动态加权融合。
将裂缝网格抽象为图（节点=网格/裂缝段，边=拓扑邻接），GAT 自动学习属性对勘探价值的贡献权重。
"""
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

# 高价值属性（油气领域专家规则：连通率等赋更高权重）
HIGH_VALUE_ATTRS = [
    "Connections per Branch", "Connections per Trace", "Connection Frequency",
]
# 默认高价值权重（其余基础属性权重 1.0）
DEFAULT_HIGH_VALUE_WEIGHT = 1.5


def weighted_fusion(
    X: np.ndarray,
    feature_names: List[str],
    high_value_weight: float = DEFAULT_HIGH_VALUE_WEIGHT,
    high_value_attrs: Optional[List[str]] = None,
) -> np.ndarray:
    """
    基础版：加权融合。对 HIGH_VALUE_ATTRS 中的列赋更高权重，得到一维「勘探价值」得分（或保持多维）。
    返回：每个样本一个标量得分，形状 (n_samples,)。
    """
    if high_value_attrs is None:
        high_value_attrs = HIGH_VALUE_ATTRS
    w = np.ones(X.shape[1], dtype=np.float64)
    for i, name in enumerate(feature_names):
        if name in high_value_attrs:
            w[i] = high_value_weight
    # 加权求和后归一化到 0-1 区间便于对比
    score = X @ w
    if score.max() > score.min():
        score = (score - score.min()) / (score.max() - score.min())
    return score


def build_grid_graph(
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    n_nodes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建规则网格图：节点数 = n_rows * n_cols（或给定 n_nodes 时用近似网格）。
    返回 (edge_index (2, E), node_to_grid (n_nodes,) 行优先索引)。
    """
    if n_nodes is not None and (n_rows is None or n_cols is None):
        n_rows = int(np.sqrt(n_nodes)) or 1
        n_cols = (n_nodes + n_rows - 1) // n_rows
        n = n_nodes
    else:
        n = (n_rows or 0) * (n_cols or 0)
    edges = []
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= n:
                continue
            if r > 0:
                j = (r - 1) * n_cols + c
                if j < n:
                    edges.append((idx, j))
            if r < n_rows - 1:
                j = (r + 1) * n_cols + c
                if j < n:
                    edges.append((idx, j))
            if c > 0:
                j = r * n_cols + (c - 1)
                if j < n:
                    edges.append((idx, j))
            if c < n_cols - 1:
                j = r * n_cols + (c + 1)
                if j < n:
                    edges.append((idx, j))
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return edge_index, np.arange(n, dtype=np.int64)


def gat_fusion(
    X: np.ndarray,
    edge_index: np.ndarray,
    in_channels: Optional[int] = None,
    hidden_channels: int = 32,
    out_channels: int = 1,
    heads: int = 2,
    dropout: float = 0.2,
    epochs: int = 100,
    lr: float = 0.01,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, object, Dict]:
    """
    进阶版：图注意力网络（GAT）动态加权融合。节点特征 X，边 edge_index (2, E)。
    完整训练循环：自编码式重构（编码→低维→解码→原空间），损失=MSE(recon, x)。
    返回 (融合得分 (n_samples,), model, metrics_dict)。
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GATConv
    except ImportError:
        raise ImportError("GAT 需要 PyTorch 与 PyTorch Geometric：pip install torch torch_geometric")
    if in_channels is None:
        in_channels = X.shape[1]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    X_t = torch.tensor(X, dtype=torch.float32)
    edge_t = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=X_t, edge_index=edge_t).to(device)

    class GATEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

        def forward(self, x, edge_index):
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    class GATAutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = GATEncoder()
            self.decoder = nn.Linear(out_channels, in_channels)

        def forward(self, x, edge_index):
            z = self.encoder(x, edge_index)
            recon = self.decoder(z)
            return recon, z

    model = GATAutoEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        recon, z = model(data.x, data.edge_index)
        loss = F.mse_loss(recon, data.x)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
    model.eval()
    with torch.no_grad():
        _, z = model(data.x, data.edge_index)
    Z = z.cpu().numpy()
    if Z.shape[1] == 1:
        Z = Z.ravel()
    # 归一化到 0-1 便于与加权融合对比
    if Z.size > 0 and np.ptp(Z) > 0:
        Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    return Z, model, {"loss_final": loss_history[-1], "loss_history": loss_history}


def run_fusion_comparison_experiment(
    csv_path: str,
    out_dir: Optional[str] = None,
    high_value_weight: float = DEFAULT_HIGH_VALUE_WEIGHT,
    gat_epochs: int = 80,
    save_boxplot: bool = True,
) -> Dict:
    """
    加权融合 vs GAT 融合对比实验：输出两种得分及箱线图，突出 GAT 对拓扑结构的建模优势。
    返回 dict：weighted_scores, gat_scores, df_with_both, boxplot_path。
    """
    from feature_engineering import build_feature_matrix, DEFAULT_FEATURE_COLUMNS
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    r = build_feature_matrix(csv_path, feature_columns=DEFAULT_FEATURE_COLUMNS, out_processed_dir=None)
    X, names = r["X"], r["feature_names"]
    n = X.shape[0]
    # 加权融合
    weighted_scores = weighted_fusion(X, names, high_value_weight=high_value_weight)
    # 网格图 + GAT
    edge_index, _ = build_grid_graph(n_nodes=n)
    try:
        gat_scores, _, gat_metrics = gat_fusion(X, edge_index, epochs=gat_epochs, out_channels=1)
    except ImportError:
        gat_scores = np.zeros(n)  # 无 PyG 时用占位
        gat_metrics = {}
    df = r["df"].copy()
    df["weighted_fusion_score"] = weighted_scores
    df["gat_fusion_score"] = gat_scores
    boxplot_path = None
    if save_boxplot:
        try:
            from evaluation import plot_fusion_comparison_boxplot
            boxplot_path = os.path.join(out_dir, "fusion_comparison_boxplot.png")
            plot_fusion_comparison_boxplot(
                {"加权融合": weighted_scores, "GAT 融合": gat_scores},
                save_path=boxplot_path,
                title="加权融合 vs GAT 融合 得分分布",
            )
        except Exception:
            pass
    return {
        "weighted_scores": weighted_scores,
        "gat_scores": gat_scores,
        "df_with_both": df,
        "boxplot_path": boxplot_path,
        "gat_metrics": gat_metrics,
    }


def run_weighted_fusion_pipeline(
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    high_value_weight: float = DEFAULT_HIGH_VALUE_WEIGHT,
    out_dir: Optional[str] = None,
) -> pd.DataFrame:
    """从 CSV 到加权融合得分的完整流程（依赖 feature_engineering）。"""
    from feature_engineering import build_feature_matrix, DEFAULT_FEATURE_COLUMNS
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "data", "processed")
    out_processed = out_dir
    r = build_feature_matrix(csv_path, out_processed_dir=out_processed, feature_columns=feature_columns)
    X, names = r["X"], r["feature_names"]
    score = weighted_fusion(X, names, high_value_weight=high_value_weight)
    df = r["df"].copy()
    df["weighted_fusion_score"] = score
    return df


if __name__ == "__main__":
    import sys
    csv_path = "Yingmai 2 area in Tarim Basin.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    df = run_weighted_fusion_pipeline(csv_path)
    print("加权融合得分列已加入，前 5 行:", df["weighted_fusion_score"].head().tolist())
