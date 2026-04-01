@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ========================================
echo  油气区断裂网络连通性智能分析与预测系统
echo  Windows 打包脚本 v1.1
echo ========================================
echo.

:: ── 检查 Python 3.11 ──────────────────────────────────
echo [检查] 正在检测 Python 版本...
python --version 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [错误] 未找到 Python，请先安装 Python 3.11
    echo 下载地址：https://www.python.org/downloads/release/python-31115/
    echo 安装时请勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

:: 检查是否为 3.11（仅警告，不强制退出）
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [检查] 当前 Python 版本：%PY_VER%
echo %PY_VER% | findstr /b "3.11" >nul
if %errorlevel% neq 0 (
    echo [警告] 推荐使用 Python 3.11，当前版本可能存在兼容性问题
    echo        如遇问题请从 https://www.python.org/downloads/release/python-31115/ 安装 3.11
    echo.
)

:: ── 创建虚拟环境 ──────────────────────────────────────
echo [1/4] 创建虚拟环境 .venv_win_build ...
if not exist ".venv_win_build" (
    python -m venv .venv_win_build
    if %errorlevel% neq 0 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo       虚拟环境创建成功
) else (
    echo       虚拟环境已存在，跳过创建
)

call .venv_win_build\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [错误] 虚拟环境激活失败
    pause
    exit /b 1
)

:: ── 安装依赖 ──────────────────────────────────────────
echo.
echo [2/4] 安装项目依赖（含 torch / torch-geometric，首次约需 10-20 分钟）...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [警告] pip 升级失败，继续尝试安装依赖...
)

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [错误] 依赖安装失败，请检查网络连接或 requirements.txt
    pause
    exit /b 1
)

pip install pyinstaller
if %errorlevel% neq 0 (
    echo [错误] PyInstaller 安装失败
    pause
    exit /b 1
)

:: ── 执行打包 ──────────────────────────────────────────
echo.
echo [3/4] 执行 PyInstaller 打包（约需 3-8 分钟）...
pyinstaller build_app.spec --noconfirm
if %errorlevel% neq 0 (
    echo.
    echo [错误] PyInstaller 打包失败，请查看上方错误信息
    pause
    exit /b 1
)

:: ── 完成 ──────────────────────────────────────────────
echo.
echo ========================================
echo [4/4] 打包完成！
echo ========================================
echo.
echo 输出目录：
echo   dist\油气区断裂网络连通性智能分析与预测系统\
echo.
echo 可执行文件：
echo   dist\油气区断裂网络连通性智能分析与预测系统\油气区断裂网络连通性智能分析与预测系统.exe
echo.
echo 分发说明：
echo   将整个 dist\油气区断裂网络连通性智能分析与预测系统\ 文件夹打包压缩后分发
echo   不可只发送 .exe 文件，必须连同 _internal\ 文件夹一起分发
echo.
pause
