# -*- coding: utf-8 -*-
"""
统一配置 matplotlib 中文字体，避免图表中出现乱码或方框。
支持 Windows、macOS、Linux 跨平台。
"""
import sys


def setup_matplotlib_chinese():
    """
    配置 matplotlib 正确显示中文及负号。
    应在任何绘图操作之前调用（如 main.py 入口、独立脚本开头）。
    """
    import matplotlib
    import matplotlib.pyplot as plt

    # 按系统选择中文字体优先级（matplotlib 会使用第一个可用的）
    if sys.platform == "darwin":
        # macOS：蘋方、黑体、华文黑体
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
        # Windows：微软雅黑、黑体
        fonts = [
            "Microsoft YaHei",
            "SimHei",
            "KaiTi",
            "Arial Unicode MS",
        ]
    else:
        # Linux 及其他
        fonts = [
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "Noto Sans CJK SC",
            "Droid Sans Fallback",
            "Microsoft YaHei",
            "SimHei",
        ]

    plt.rcParams["font.sans-serif"] = fonts
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
    # 确保新创建的 figure 使用当前配置
    matplotlib.rcParams["font.sans-serif"] = fonts
    matplotlib.rcParams["axes.unicode_minus"] = False
