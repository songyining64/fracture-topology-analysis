# -*- coding: utf-8 -*-
"""
统一配置 matplotlib 中文字体，避免图表中出现乱码或方框。
支持 Windows、macOS、Linux 跨平台。
"""
import os
import sys


def _ensure_mpl_cache_dir() -> None:
    """
    确保 matplotlib 缓存目录可写。
    若默认的 ~/.matplotlib 不可写（常见于 macOS com.apple.provenance 锁定），
    自动回退到 ~/.cache/matplotlib_fracture，并设置 MPLCONFIGDIR 环境变量。
    必须在 import matplotlib 之前调用，否则 matplotlib 已经选定了缓存目录。
    """
    if "MPLCONFIGDIR" in os.environ:
        return
    default_dir = os.path.join(os.path.expanduser("~"), ".matplotlib")
    # 用写文件测试真正的可写性（仅靠 os.access 在某些 macOS 版本上不可靠）
    test_file = os.path.join(default_dir, ".write_test")
    try:
        os.makedirs(default_dir, exist_ok=True)
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
    except OSError:
        fallback = os.path.join(os.path.expanduser("~"), ".cache", "matplotlib_fracture")
        os.makedirs(fallback, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = fallback


def setup_matplotlib_chinese():
    """
    配置 matplotlib 正确显示中文及负号。
    应在任何绘图操作之前调用（如 main.py 入口、独立脚本开头）。
    """
    _ensure_mpl_cache_dir()

    import matplotlib
    import matplotlib.pyplot as plt

    # 按系统选择中文字体优先级（matplotlib 会使用第一个可用的）
    if sys.platform == "darwin":
        fonts = [
            "PingFang SC",
            "Heiti SC",
            "STHeiti",
            "Songti SC",
            "Arial Unicode MS",
            "Microsoft YaHei",
            "SimHei",
        ]
    elif sys.platform == "win32":
        fonts = [
            "Microsoft YaHei",
            "SimHei",
            "KaiTi",
            "Arial Unicode MS",
        ]
    else:
        fonts = [
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "Noto Sans CJK SC",
            "Droid Sans Fallback",
            "Microsoft YaHei",
            "SimHei",
        ]

    plt.rcParams["font.sans-serif"] = fonts
    plt.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.sans-serif"] = fonts
    matplotlib.rcParams["axes.unicode_minus"] = False
