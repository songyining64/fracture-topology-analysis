#!/usr/bin/env bash
# ================================================
#  油气区断裂网络连通性智能分析与预测系统
#  macOS 打包脚本 v1.1
# ================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="油气区断裂网络连通性智能分析与预测系统_Mac"

echo "========================================"
echo " 油气区断裂网络连通性智能分析与预测系统"
echo " macOS 打包脚本 v1.1"
echo "========================================"
echo ""

# ── 检查 Python ──────────────────────────────
echo "[检查] 正在检测 Python 版本..."
if ! command -v python3 &>/dev/null; then
    echo "[错误] 未找到 python3，请先安装 Python 3.11"
    echo "下载地址：https://www.python.org/downloads/release/python-31115/"
    exit 1
fi
PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
echo "[检查] 当前 Python 版本：$PY_VER"
if [[ "$PY_VER" != 3.11* ]]; then
    echo "[警告] 推荐使用 Python 3.11，当前版本可能存在兼容性问题"
fi
echo ""

# ── 创建虚拟环境 ──────────────────────────────
echo "[1/4] 创建虚拟环境 .venv_mac_build ..."
if [ ! -d ".venv_mac_build" ]; then
    python3 -m venv .venv_mac_build
    echo "      虚拟环境创建成功"
else
    echo "      虚拟环境已存在，跳过创建"
fi

source .venv_mac_build/bin/activate

# ── 安装依赖 ──────────────────────────────────
echo ""
echo "[2/4] 安装项目依赖（首次约需 10-20 分钟）..."
python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

# ── 执行打包 ──────────────────────────────────
echo ""
echo "[3/4] 执行 PyInstaller 打包（约需 3-8 分钟）..."
# 指定 matplotlib 缓存目录，避免权限问题导致字体缓存构建卡住
export MPLCONFIGDIR="$SCRIPT_DIR/.mplconfig_build"
mkdir -p "$MPLCONFIGDIR"
pyinstaller build_app.spec --noconfirm

# ── 完成 ──────────────────────────────────────
echo ""
echo "========================================"
echo "[4/4] 打包完成！"
echo "========================================"
echo ""
echo "输出目录："
echo "  dist/${APP_NAME}/"
echo ""
echo "macOS .app 包："
echo "  dist/${APP_NAME}.app"
echo ""
echo "分发说明："
echo "  分发 .app 包给其他 Mac 用户；"
echo "  若分发 onedir 版本，需将整个 dist/${APP_NAME}/ 文件夹一起打包压缩。"
echo ""
