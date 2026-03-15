# -*- coding: utf-8 -*-
"""
多区块 CSV 批量流水线：特征工程 → 融合 → 训练。
用法：python batch_run.py [csv1.csv csv2.csv ...]  或  python batch_run.py  # 使用 config 中列表或默认英买2区
"""
import os
import sys
import glob
import argparse
from typing import List, Optional

_PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

try:
    from utils.logging_utils import get_logger
    from utils.config_loader import load_config
    from utils.validation import check_path_exists
except ImportError:
    get_logger = lambda **kw: __import__("logging").getLogger("batch")
    load_config = lambda: {}
    check_path_exists = lambda p, **kw: p

from feature_engineering import build_feature_matrix, DEFAULT_FEATURE_COLUMNS
from fusion_algorithm import run_weighted_fusion_pipeline

logger = get_logger("batch_run")


def run_pipeline_for_one(
    csv_path: str,
    out_processed_dir: Optional[str] = None,
    out_fusion_csv: bool = True,
    high_value_weight: float = 1.5,
) -> dict:
    """对单个 CSV 执行：特征工程 → 加权融合。返回摘要。"""
    check_path_exists(csv_path, "CSV")
    cfg = load_config()
    fe_cfg = cfg.get("feature_engineering", {})
    if out_processed_dir is None:
        out_processed_dir = os.path.join(_PROGRAM_DIR, "data", "processed")
    os.makedirs(out_processed_dir, exist_ok=True)
    # 特征工程
    logger.info("特征工程: %s", csv_path)
    r = build_feature_matrix(
        csv_path,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        normalize_method=fe_cfg.get("normalize_method", "standard"),
        outlier_method=fe_cfg.get("outlier_method", "iqr"),
        variance_threshold=fe_cfg.get("variance_threshold", 1e-6),
        n_select_mi=fe_cfg.get("n_select_mi"),
        out_processed_dir=out_processed_dir,
    )
    logger.info("特征矩阵形状: %s, 特征数: %s", r["X"].shape, r["n_features"])
    # 加权融合
    logger.info("加权融合: %s", csv_path)
    df_fusion = run_weighted_fusion_pipeline(
        csv_path,
        high_value_weight=high_value_weight,
        out_dir=out_processed_dir,
    )
    if out_fusion_csv:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(out_processed_dir, f"{base}_fusion.csv")
        df_fusion.to_csv(out_path, index=False)
        logger.info("融合结果已写: %s", out_path)
    return {
        "csv": csv_path,
        "n_samples": r["n_samples"],
        "n_features": r["n_features"],
        "fusion_score_mean": float(df_fusion["weighted_fusion_score"].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="多区块批量：特征工程 → 融合")
    parser.add_argument("csv_list", nargs="*", help="CSV 路径列表，不传则用默认英买2区")
    parser.add_argument("--no-fusion-csv", action="store_true", help="不写融合结果 CSV")
    args = parser.parse_args()
    cfg = load_config()
    fusion_weight = (cfg.get("fusion") or {}).get("high_value_weight", 1.5)
    if args.csv_list:
        csv_paths = [p for p in args.csv_list if os.path.isfile(p)]
    else:
        default_csv = os.path.join(_PROGRAM_DIR, "Yingmai 2 area in Tarim Basin.csv")
        csv_paths = [default_csv] if os.path.isfile(default_csv) else []
    if not csv_paths:
        logger.warning("未找到任何 CSV，请传入路径或将 Yingmai 2 area in Tarim Basin.csv 放在 program 目录")
        return
    results = []
    for p in csv_paths:
        try:
            res = run_pipeline_for_one(
                p,
                out_fusion_csv=not args.no_fusion_csv,
                high_value_weight=fusion_weight,
            )
            results.append(res)
        except Exception as e:
            logger.exception("处理失败 %s: %s", p, e)
    logger.info("批量完成，共 %s 个文件", len(results))
    return results


if __name__ == "__main__":
    main()
