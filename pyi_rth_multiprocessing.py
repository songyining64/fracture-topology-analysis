import multiprocessing
import os

multiprocessing.freeze_support()

# 限制 joblib/loky 不使用多进程，避免 PyInstaller frozen 环境下 spawn worker 崩溃
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
