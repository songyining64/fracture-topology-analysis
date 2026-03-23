# -*- coding: utf-8 -*-
"""
主线 A：面向断裂网络的空间–拓扑特征融合学习框架

目标：
- 将 fractopo 的拓扑指标 + 空间统计特征（长度、密度、数量等）+ 地质约束
  统一到一个特征矩阵中；
- 在此之上使用加权融合 + GAT 图模型 + XGBoost 回归做预测与解释。

说明：
- 这里假定「空间统计特征 + fractopo 拓扑指标」已经通过前置脚本
  写入网格 CSV（例如 Yingmai 2 area in Tarim Basin.csv）；
- 本模块负责：特征工程 → 融合（加权 / GAT）→ XGBoost 训练 → SHAP 解释，
  形成一条完整的算法主线，便于在国奖说明书中展示。
"""

import os
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from feature_engineering import (
    build_feature_matrix,
    DEFAULT_FEATURE_COLUMNS,
)
from fusion_algorithm import (
    weighted_fusion,
    build_grid_graph,
    gat_fusion,
    adaptive_weighted_fusion,
)
from gnn_embeddings import (
    gnn_embedding_graphsage,
)
from multiscale_features import build_multiscale_pyramid
from ml.train import train_xgboost_regression
from ml.explain import shap_feature_importance
from utils.config_loader import load_config


def run_spatial_topology_fusion_pipeline(
    csv_path: str,
    target_column: Optional[str],
    config_path: Optional[str] = None,
    use_gat: bool = True,
    use_gnn_embedding: bool = True,
    use_multiscale: bool = True,
    use_adaptive_weight: bool = True,
    compute_shap: bool = True,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    面向断裂网络的「空间–拓扑特征融合学习」完整流水线。

    步骤：
    1. 利用 feature_engineering.build_feature_matrix 构建特征矩阵（拓扑 + 空间统计）；
    2. 基于专家规则做加权融合，得到勘探价值得分；
    3. 在规则网格图上运行 GAT，自适应地学习拓扑结构加权（可选）；
    4. 用 XGBoost 对目标属性做回归预测，评估 CV 与测试集指标；
    5. 使用 SHAP 对 XGBoost 模型做特征贡献度解释（可选）。

    返回：
        {
            "df": 带有融合得分和预测结果的 DataFrame,
            "xgb_result": XGBoost 训练结果 dict,
            "shap_importance": SHAP 特征重要性 DataFrame（如 compute_shap=False 则为 None),
            "gat_metrics": GAT 训练指标（若 use_gat 且依赖满足）
        }
    """
    cfg = load_config(config_path)
    fe_cfg = cfg.get("feature_engineering", {}) if isinstance(cfg, dict) else {}
    fusion_cfg = cfg.get("fusion", {}) if isinstance(cfg, dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    high_value_attrs: Optional[List[str]] = cfg.get("high_value_attrs")

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 特征工程：空间 + 拓扑特征矩阵
    r = build_feature_matrix(
        csv_path,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        target_column=target_column,
        normalize_method=fe_cfg.get("normalize_method", "standard"),
        outlier_method=fe_cfg.get("outlier_method", "iqr"),
        variance_threshold=float(fe_cfg.get("variance_threshold", 1e-6)),
        n_select_mi=fe_cfg.get("n_select_mi"),
        out_processed_dir=None,
    )
    X = r["X"]
    y = r["y"]
    feature_names = r["feature_names"]
    df = r["df"].copy()

    if y is None:
        raise ValueError("run_spatial_topology_fusion_pipeline 需要提供有效的 target_column。")

    # 可选：多尺度空间特征金字塔（仅当样本数能构成规则网格时生效）
    if use_multiscale:
        n_samples = X.shape[0]
        n_rows = int(np.sqrt(n_samples)) or 1
        n_cols = (n_samples + n_rows - 1) // n_rows
        if n_rows * n_cols == n_samples:
            X_ms, suffixes = build_multiscale_pyramid(
                X,
                n_rows=n_rows,
                n_cols=n_cols,
                scales=fusion_cfg.get("multiscale_scales", [1, 2, 4]),
                mode=fusion_cfg.get("multiscale_mode", "mean"),
            )
            new_feature_names = []
            for suf in suffixes:
                new_feature_names.extend([f"{name}_{suf}" for name in feature_names])
            X = X_ms
            feature_names = new_feature_names
        else:
            # 样本数非规则网格（如过滤后数量变化），跳过多尺度
            import warnings as _w
            _w.warn(f"样本数 {n_samples} 无法构成规则网格 (n_rows*n_cols={n_rows}*{n_cols}={n_rows*n_cols})，跳过多尺度金字塔。")

    # 2. 加权融合：凸显高价值拓扑属性（规则版 + 自适应版）
    high_value_weight = float(fusion_cfg.get("high_value_weight", 1.5))
    fused_weighted = weighted_fusion(
        X,
        feature_names,
        high_value_weight=high_value_weight,
        high_value_attrs=high_value_attrs,
    )
    df["fusion_weighted_score"] = fused_weighted

    # 自适应可学习加权（Adaptive Feature Weighting）
    adaptive_scores = None
    adaptive_weights_mean = None
    if use_adaptive_weight:
        try:
            adaptive_scores, adaptive_weights_mean = adaptive_weighted_fusion(
                X,
                feature_names,
                context_features=None,
                hidden_dim=int(fusion_cfg.get("adaptive_hidden_dim", 32)),
                epochs=int(fusion_cfg.get("adaptive_epochs", 200)),
                lr=float(fusion_cfg.get("adaptive_lr", 1e-3)),
            )
        except ImportError:
            adaptive_scores = None
    if adaptive_scores is not None:
        df["fusion_adaptive_score"] = adaptive_scores

    # 3. GAT 图模型融合（可选）
    gat_scores = None
    gat_metrics: Dict[str, Any] = {}
    edge_index, _ = build_grid_graph(n_nodes=X.shape[0])
    if use_gat:
        try:
            gat_scores, _, gat_metrics = gat_fusion(
                X,
                edge_index,
                epochs=int(fusion_cfg.get("gat_epochs", 80)),
                out_channels=1,
            )
        except ImportError:
            gat_scores = None
            gat_metrics = {"error": "PyTorch / PyTorch Geometric 未安装，GAT 未运行"}
    if gat_scores is not None:
        df["fusion_gat_score"] = gat_scores

    # 3.1 图级表示学习（GraphSAGE 嵌入）
    gnn_node_emb = None
    gnn_graph_emb = None
    gnn_metrics: Dict[str, Any] = {}
    if use_gnn_embedding:
        try:
            gnn_node_emb, gnn_graph_emb, gnn_metrics = gnn_embedding_graphsage(
                X,
                edge_index,
                hidden_dim=int(fusion_cfg.get("gnn_hidden_dim", 64)),
                out_dim=int(fusion_cfg.get("gnn_out_dim", 32)),
                epochs=int(fusion_cfg.get("gnn_epochs", 200)),
            )
        except ImportError:
            gnn_metrics = {"error": "PyTorch / PyTorch Geometric 未安装，GraphSAGE 未运行"}
    if gnn_node_emb is not None:
        # 节点级嵌入拼接到特征矩阵，形成「工程特征 + 图嵌入」联合表示
        X = np.concatenate([X, gnn_node_emb], axis=1)
        # 为嵌入生成列名
        emb_names = [f"gnn_sage_{i}" for i in range(gnn_node_emb.shape[1])]
        feature_names = feature_names + emb_names

    # 4. XGBoost 回归：在融合特征基础上做目标预测
    xgb_result = train_xgboost_regression(
        X,
        y,
        n_splits=int(train_cfg.get("n_splits", 5)),
        test_size=float(train_cfg.get("test_size", 0.1)),
        random_state=int(train_cfg.get("random_state", 42)),
    )
    model = xgb_result["model"]
    # 使用全量 X 计算当前样本的预测值，便于与融合得分对比
    try:
        import numpy as _np

        y_pred_all = model.predict(_np.asarray(X))
    except Exception:
        y_pred_all = None
    if y_pred_all is not None and len(y_pred_all) == len(df):
        df["xgb_pred"] = y_pred_all

    # 5. SHAP 解释：特征贡献度
    shap_df = None
    if compute_shap:
        try:
            shap_df = shap_feature_importance(
                model,
                X,
                feature_names=feature_names,
                is_tree=True,
                out_plot_path=os.path.join(out_dir, "shap_summary.png"),
            )
        except Exception as e:
            shap_df = pd.DataFrame(
                {"feature": feature_names, "importance": 0.0, "contribution_pct": 0.0}
            )
            shap_df.attrs["error"] = f"SHAP 计算失败: {e}"

    return {
        "df": df,
        "xgb_result": xgb_result,
        "shap_importance": shap_df,
        "gat_metrics": gat_metrics,
        "gnn_metrics": gnn_metrics,
        "adaptive_weights_mean": adaptive_weights_mean,
    }


if __name__ == "__main__":
    """
    示例用法（命令行）：

    python spatial_topology_framework.py \
        "Yingmai 2 area in Tarim Basin.csv" \
        --target "Porosity"
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="面向断裂网络的空间–拓扑特征融合学习框架（示例运行入口）"
    )
    parser.add_argument("csv_path", help="网格级特征 CSV，包含 fractopo 拓扑指标与空间统计特征")
    parser.add_argument(
        "--target", dest="target_column", required=True, help="监督学习目标列名（如 Porosity）"
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="可选 config.yaml 路径，未提供时使用 program/config.yaml",
    )
    parser.add_argument(
        "--no-gat", dest="use_gat", action="store_false", help="关闭 GAT 融合，仅使用加权融合与 XGBoost"
    )
    parser.add_argument(
        "--no-shap", dest="compute_shap", action="store_false", help="关闭 SHAP 解释计算"
    )
    args = parser.parse_args()

    result = run_spatial_topology_fusion_pipeline(
        csv_path=args.csv_path,
        target_column=args.target_column,
        config_path=args.config_path,
        use_gat=args.use_gat,
        compute_shap=args.compute_shap,
    )
    print("融合框架运行完成。")
    print("XGBoost CV 指标聚合：", result["xgb_result"]["cv_agg"])
    print("XGBoost 测试集指标：", result["xgb_result"]["test_metrics"])
    if result["shap_importance"] is not None:
        print("Top5 特征贡献：")
        print(result["shap_importance"].head(5).to_string(index=False))

