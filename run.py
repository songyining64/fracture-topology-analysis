#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨平台启动入口（在项目根目录执行即可）：
python run.py
"""
import os
import runpy
import sys


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    program_dir = os.path.join(root, "program")
    main_py = os.path.join(program_dir, "main.py")
    if not os.path.isfile(main_py):
        print("未找到 program/main.py，请确认在项目根目录运行。")
        return 1
    if program_dir not in sys.path:
        sys.path.insert(0, program_dir)
    runpy.run_path(main_py, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

