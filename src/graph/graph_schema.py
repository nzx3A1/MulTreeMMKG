"""图谱 Schema 定义。

以 Python 类 / 数据类的形式描述节点标签、属性、关系类型，
供 graph_merger、graph_validator、neo4j_writer 共用。
"""
from __future__ import annotations

# 节点类型枚举：Paper / Section / Chunk / Entity / Formula ...
# 关系类型枚举：HAS_SECTION / HAS_CHUNK / MENTIONED_IN / <自定义关系>
# 属性 schema：name / type / attributes ...
