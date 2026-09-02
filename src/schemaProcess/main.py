"""兼容 ``python -m src.schemaProcess.main`` 的入口。"""

from .stage_05_schema_alignment import main


if __name__ == "__main__":
    raise SystemExit(main())
