"""直接运行测试文件时的路径引导。

pytest 会自动加载 conftest.py，但用户在编辑器里点击运行单个 test_*.py 时不会加载。
这里把项目根目录加入 sys.path，让直接运行和 pytest 运行都能找到 src、config、util 包。
"""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
