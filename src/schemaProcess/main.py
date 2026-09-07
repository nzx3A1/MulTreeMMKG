"""Schema 对齐命令行入口，兼容模块运行和按文件路径直接运行。"""

from __future__ import annotations

import sys
from pathlib import Path


if __package__:
    from .stage_05_schema_alignment import main
else:
    # 直接运行本文件时，Python 不会建立 ``src.schemaProcess`` 包上下文。
    # 将项目根目录加入模块搜索路径后，改用绝对导入。
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.schemaProcess.stage_05_schema_alignment import main


if __name__ == "__main__":
    raise SystemExit(main())
