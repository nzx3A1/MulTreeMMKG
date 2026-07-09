// =============================================================================
// 石油地质多模态知识图谱 —— Neo4j 约束与索引脚本
// 由 src/graph/neo4j_writer.py 在首次写入前自动执行。
// =============================================================================

// ---- 唯一性约束 ----
// CREATE CONSTRAINT paper_id        IF NOT EXISTS FOR (p:Paper)        REQUIRE p.id IS UNIQUE;
// CREATE CONSTRAINT section_id      IF NOT EXISTS FOR (s:Section)      REQUIRE s.id IS UNIQUE;
// CREATE CONSTRAINT entity_id       IF NOT EXISTS FOR (e:Entity)       REQUIRE e.id IS UNIQUE;
// CREATE CONSTRAINT chunk_id        IF NOT EXISTS FOR (c:Chunk)        REQUIRE c.id IS UNIQUE;

// ---- 常用查询索引 ----
// CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
// CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
// CREATE INDEX relation_type IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type);
