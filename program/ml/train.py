# -*- coding: utf-8 -*-
"""
训练逻辑：5 折交叉验证 + 独立测试集，训练 XGBoost / GAT，保存模型与效果报告。
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.model_selection import KFold, train_test_split

# 将 program 加入路径以便 import feature_engineering 等
_PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

from evaluation import regression_metrics, classification_metrics


def train_xgboost_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    test_size: float = 0.1,
    random_state: int = 42,
    **xgb_params,
) -> Dict[str, Any]:
    """XGBoost 回归 + 5 折 CV + 留出测试集，返回模型、CV 指标、测试集指标。"""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("请安装 xgboost: pip install xgboost")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_metrics = []
    for train_idx, val_idx in kf.split(X_train):
        Xt, Xv = X_train[train_idx], X_train[val_idx]
        yt, yv = y_train[train_idx], y_train[val_idx]
        model = xgb.XGBRegressor(random_state=random_state, **xgb_params)
        model.fit(Xt, yt)
        pred = model.predict(Xv)
        cv_metrics.append(regression_metrics(yv, pred))
    # 用划分出的训练集（全量数据的 1-test_size 部分）训练最终模型，测试集仅用于评估，不参与训练
    model = xgb.XGBRegressor(random_state=random_state, **xgb_params)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_pred)
    cv_agg = {
        "MAE_mean": np.mean([m["MAE"] for m in cv_metrics]),
        "RMSE_mean": np.mean([m["RMSE"] for m in cv_metrics]),
        "R2_mean": np.mean([m["R2"] for m in cv_metrics]),
    }
    return {
        "model": model,
        "cv_metrics": cv_metrics,
        "cv_agg": cv_agg,
        "test_metrics": test_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def train_xgboost_classification(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    test_size: float = 0.1,
    random_state: int = 42,
    **xgb_params,
) -> Dict[str, Any]:
    """XGBoost 分类 + 5 折 CV + 留出测试集。"""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("请安装 xgboost: pip install xgboost")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_metrics = []
    for train_idx, val_idx in kf.split(X_train):
        Xt, Xv = X_train[train_idx], X_train[val_idx]
        yt, yv = y_train[train_idx], y_train[val_idx]
        model = xgb.XGBClassifier(random_state=random_state, **xgb_params)
        model.fit(Xt, yt)
        pred = model.predict(Xv)
        cv_metrics.append(classification_metrics(yv, pred))
    # 用划分出的训练集（全量数据的 1-test_size 部分）训练最终模型，测试集仅用于评估，不参与训练
    model = xgb.XGBClassifier(random_state=random_state, **xgb_params)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_metrics = classification_metrics(y_test, test_pred)
    cv_agg = {
        "accuracy_mean": np.mean([m["accuracy"] for m in cv_metrics]),
        "f1_mean": np.mean([m["f1"] for m in cv_metrics]),
    }
    return {
        "model": model,
        "cv_metrics": cv_metrics,
        "cv_agg": cv_agg,
        "test_metrics": test_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def save_model_report(
    result: Dict[str, Any],
    model_dir: str,
    name: str = "xgboost",
    feature_names: Optional[list] = None,
) -> str:
    """保存模型与效果报告到 model_dir。"""
    os.makedirs(model_dir, exist_ok=True)
    model = result.get("model")
    if model is not None:
        path = os.path.join(model_dir, f"{name}.json")
        model.save_model(path)
    report = {k: v for k, v in result.items() if k != "model" and not callable(v)}

    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(x) for x in obj]
        return obj

    report = _serialize(report)
    report_path = os.path.join(model_dir, f"{name}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if feature_names is not None:
        with open(os.path.join(model_dir, f"{name}_features.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(feature_names))
    return report_path


if __name__ == "__main__":
    from feature_engineering import build_feature_matrix, DEFAULT_FEATURE_COLUMNS
    csv_path = os.path.join(_PROGRAM_DIR, "Yingmai 2 area in Tarim Basin.csv")
    if not os.path.isfile(csv_path):
        print("未找到 CSV，请指定网格 CSV 路径。")
        sys.exit(1)
    r = build_feature_matrix(csv_path, target_column=None, out_processed_dir=None)
    X, y = r["X"], r["y"]
    if y is None:
        y = r["X"][:, 0]  # 无标签时用第一列做演示
    res = train_xgboost_regression(X, y, n_estimators=50, max_depth=4)
    model_dir = os.path.join(_PROGRAM_DIR, "model")
    save_model_report(res, model_dir, name="xgboost_reg", feature_names=r.get("feature_names"))
    print("CV R2 mean:", res["cv_agg"]["R2_mean"])
    print("Test R2:", res["test_metrics"]["R2"])
    print("报告已保存至 model/")
