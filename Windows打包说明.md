# Windows 版本打包说明

> **重要：PyInstaller 必须在目标平台上运行。**
> 要打包 Windows `.exe`，必须在 **Windows 系统**上执行以下步骤。
> macOS 上无法直接编译出 Windows 可执行文件。

---

## 前置要求

| 要求 | 说明 |
|------|------|
| 操作系统 | Windows 10 / 11（64位） |
| Python | **3.11**（推荐）[下载地址](https://www.python.org/downloads/release/python-31115/) |
| 磁盘空间 | 至少 **10 GB** 可用（含 torch 等大型依赖） |
| 内存 | 建议 **8 GB** 以上 |
| 网络 | 首次打包需要下载约 3-5 GB 依赖包 |

---

## 步骤一：安装 Python 3.11

1. 前往 https://www.python.org/downloads/release/python-31115/
2. 下载 **Windows installer (64-bit)**
3. 安装时**务必勾选** `Add Python 3.11 to PATH`
4. 安装完成后，打开命令提示符验证：
   ```
   python --version
   ```
   应输出 `Python 3.11.x`

---

## 步骤二：将项目传输到 Windows

将整个 `fracture-topology-analysis` 文件夹复制到 Windows 机器上，例如：
```
C:\fracture-topology-analysis\
```

**需要包含的文件：**
- `program\`（所有源代码）
- `qgis_styles\`
- `requirements.txt`
- `build_app.spec`
- `build_windows.bat`
- `pyi_rth_multiprocessing.py`
- `VERSION`

---

## 步骤三：执行打包脚本

1. 打开**文件资源管理器**，进入项目目录
2. 双击运行 **`build_windows.bat`**
   - 或者右键 → "以管理员身份运行"（推荐，避免权限问题）
3. 脚本会自动完成：
   - 创建虚拟环境 `.venv_win_build\`
   - 安装所有依赖（首次约需 10-20 分钟）
   - 执行 PyInstaller 打包（约需 3-8 分钟）

> 如果弹出 Windows Defender 防火墙提示，点击**允许访问**

---

## 步骤四：获取打包结果

打包成功后，输出位于：
```
dist\
└── 油气区断裂网络连通性智能分析与预测系统\
    ├── 油气区断裂网络连通性智能分析与预测系统.exe   ← 双击启动
    └── _internal\                                    ← 依赖文件（不可删除）
```

**分发方式：** 将整个 `油气区断裂网络连通性智能分析与预测系统\` 文件夹压缩为 zip，分发给用户。  
**注意：** 不可只发送 `.exe` 文件，必须连同 `_internal\` 文件夹一起分发。

---

## 常见问题

### Q: 提示 "找不到 Python"
确认安装时勾选了 `Add Python to PATH`，或手动将 Python 路径加入系统环境变量。

### Q: pip install 失败 / 超时
可以先配置国内镜像加速：
```bat
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
然后重新运行 `build_windows.bat`。

### Q: 打包时报 "ModuleNotFoundError"
某个包未被 PyInstaller 自动检测到，在 `build_app.spec` 的 `hidden_imports` 中添加对应包名后重新运行。

### Q: .exe 运行时报 "worker unexpectedly terminated"
已在代码中修复（`pyi_rth_multiprocessing.py` 运行时钩子），确保该文件与 `build_app.spec` 在同一目录。

### Q: 启动后界面显示乱码
Windows 系统中文字体路径与 macOS 不同，程序启动时会自动检测并配置，若仍有问题请检查系统是否安装了中文字体（如微软雅黑）。

---

