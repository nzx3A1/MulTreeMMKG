"""公式抽取器。

解析 LaTeX/MathML 公式，识别公式符号与物理含义，
建立公式-变量-地质参数之间的语义关联（如渗透率 k、孔隙度 φ）。

输出：stage_09_formula_extraction.json
对应阶段：09 公式抽取
"""
from __future__ import annotations


def extract_from_formulas(formulas: list[dict], llm_client, schema_cfg) -> list[dict]:
    """从公式列表中抽取符号、变量与语义关系。"""
    raise NotImplementedError
