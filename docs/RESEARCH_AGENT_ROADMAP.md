# Research Agent 技术路线图（1-Pager）
**聚焦：AutoResearch × Knowledge Graph**

**作者：** Zhan Li、Penny Wang  
**日期：** 2026-05-15  
**背景：** Vicino World-Model PRD（Sprint 2 主轴）· AutoResearch 试点（`autoresearch/may15`，RTX 5060）

---

## 摘要

**Research Agent** 是 Vicino 的推理层：在结构化的 brand/campaign state 与 **Knowledge Graph（KG）** 上做多步、自驱动研究，产出 **`Hypothesis`** 与 **`CreativeDirection`**（非最终素材）。**Radar** 仍是面向用户主流程的**最后一步**（Strategy → Research → Creative → **Radar** → 交付）；本路线图不改变该顺序。

**AutoResearch**（Karpathy 式自治实验）在 Sprint 2 **不用于训练专用 Research LLM**，而是沿用其**运行范式**：固定 evaluation harness、可变的 research policy、指标驱动的 keep/discard、episodic 全量日志——用于优化 Research **怎么跑**（prompt、tool 顺序、ReAct 上限），KG 作为稳定的「世界模型」（类比 `prepare.py`）。

---

## 问题与 North Star

| 现状 | 目标 |
|------|------|
| 对 brief 做单次 LLM 调用 | 在 **typed state + KG** 上多步 **ReAct** |
| 每次重复输入品牌上下文 | Strategy + KG 提供持久上下文 |
| 研究过程不可审计 | 完整 **episodic log**；假设 **retain/refute** |
| 研究质量无度量 | 用 **`research_score`** 支撑夜间 policy 迭代 |

**North star：** 减少重复外搜、提高 KG-grounded 假设比例、提升 Creative 一次通过 + Radar 通过率——且不改变 PRD 流水线（Radar 仍在末端评估 **Creative 产出**）。

---

## 架构

```
Strategy ──写入──► pulse_knowledge (KG) ◄──读/写── Research Agent
     │                      ▲                              │
     └── Brand/Campaign ──┴── pulse_memory ───────────────┘
                                    │
                                    ▼
                         CreativeDirection ──► Creative ──► Radar（最后一步）
```

| 层级 | 职责 | Sprint 2 定位 |
|------|------|----------------|
| **Knowledge Graph**（`pulse_knowledge`） | 实体/关系、图遍历 + pgvector 混合检索；Strategy 写入，Research 查询并写增量 | NetworkX + pgvector；由 contracts 定义 `Entity`/`Relation` |
| **Research Agent**（`pulse_research/`） | ReAct loop；经 **Capability Router** 调用 `kg_query`、`vector_search`、`web_search` | 不训权重；policy + prompt 经 AutoResearch 演进 |
| **AutoResearch loop** | 夜间/自治调优 **research policy** | `research_eval/` 只读；`research_policy.py` 可改 |

### AutoResearch ↔ Vicino 对照

| AutoResearch（试点仓库） | Research Agent |
|--------------------------|----------------|
| `prepare.py`（固定 eval、数据） | `research_eval/` + fixtures + KG seed + rubric |
| `train.py`（Agent 可改） | `research_policy.py` / prompts / ReAct config |
| `program.md` | `research_program.md` + Orchestration 门控 |
| 固定 5 分钟 GPU budget | 固定 **步数 + API budget + fixture 集** |
| `val_bpb` ↓ 更好 | `research_score` ↑ 更好 |
| `results.tsv` + git keep/discard | `research_results.tsv` + hypothesis retain/refute + episodic memory |

**试点记录（2026-05）：** RTX 5060 上 baseline `val_bpb` 1.283 → batch-8 微调 1.283（keep）；LR 实验 1.292（discard）。说明固定指标 + 回退机制可行——Research policy 实验沿用同一纪律。

---

## 评分设计（Research 版 AutoResearch）

分两层：支撑夜间高频实验，并与流水线末端的 Radar 对齐。

### L1 — 快分（夜间循环）

仅在冻结的 **fixture set** 上跑 **Research**（建议 10–30 个 campaign：含 `brand_state`、`campaign_state`、`kg_seed`、`must_cite` / `must_not` 约束）。

| 分量 | 权重 | 定义 |
|------|------|------|
| **G** Grounded | 0.40 | `evidence_refs` 可解析到 KG 或当次 tool 输出的比例 |
| **K** Constraints | 0.25 | 通过 `must_cite`、taboo、`must_not_claim` |
| **C** Contract | 0.15 | `Hypothesis` / `CreativeDirection` schema 合法 |
| **E** Efficiency | 0.10 | 在 budget 内惩罚过多 step / web call |
| **J** Judge（v2） | 0.10 | 冻结 rubric 的 LLM 对方向质量打分 |

**合成分：** `research_score = 0.40·G + 0.25·K + 0.15·C + 0.10·E [+ 0.10·J]`  
**硬门槛：** `K` 失败（触犯 taboo）或 `G < 0.6` → discard（类比 autoresearch crash）。  
**Keep 规则：** 全部 fixture 的 `mean(research_score)` 提升 ≥ ε → keep commit；否则 revert。

### L2 — 慢分（每周 / hold-out）

在 hold-out fixture 上跑 **Research → Creative → Radar**。  
`pipeline_score = 0.5·brand_consistency + 0.3·groundness + 0.2·(1 − retry)`  
仅当 L1 与 hold-out L2 均不退化时，才将 policy 晋升到生产。

生产中 Radar **仍为最后一步**；L2 是 Research autoresearch 的 ground-truth 校验，**不替代** Radar。

---

## 分阶段交付

| 阶段 | 时间（PRD） | 交付物 | 验收标准 |
|------|-------------|--------|----------|
| **P0** | Sprint 0 | contracts：`Entity`/`Relation`/`Hypothesis`/`CreativeDirection`；`pulse_knowledge` API（traverse、vector、hybrid、upsert） | Strategy 写入 → Research 读出 round-trip |
| **P1** | Sprint 2 | `pulse_research/` ReAct + 3 tools；Orchestration await；episodic log | E2E：brief → CreativeDirection；实体已存在时 ≥80% step 走 KG |
| **P2** | Sprint 2+ | `research_eval/` + fixtures；`research_program.md`；`research_results.tsv` | 夜间 policy 循环；20 轮内 `research_score` 均值 +0.5% |
| **P3** | Sprint 3–4 | Orchestration diff 门控；Radar L2 校准；`FeedbackSignal` 回灌 Research | Creative retry 率 ↓；重复外搜 ↓ |
| **P4** | Sprint 4 | Episodic「相似 campaign」检索；跨 campaign 洞察（brand 范围） | hold-out `pipeline_score` 可度量提升 |

---

## Knowledge Graph：Research 职责边界

| 操作 | 负责方 | 规则 |
|------|--------|------|
| 外部信号入库 | Strategy → extract → KG | Research 不替代 crawler |
| 启动前查询 | Research | **`web_search` 之前 KG-first** |
| 写增量 | Research | 每条 episode 带来源的 entity/relation |
| 隔离 scope | 查询层 | 默认按 `campaign_id` + `brand_id` 过滤（PRD Q7：全局图 + 标签） |

**数据形态（非 ML shard）：** 运行时为 JSON/图状态 + eval 用 fixture 文件——不是 tokenized 训练语料。Sprint 2 不做 `(state, kg) → direction` 的 SFT。

---

## 风险与对策

| 风险 | 对策 |
|------|------|
| KG 为空导致滥搜 | P1 前 Strategy 须 seed 最小实体集；policy 强制 KG-first |
| L1 与 L2 脱节 | hold-out fixtures；每周 L2 gate 后再 merge |
| 与 GPT autoresearch 混淆 | 文档区分两条线：**`train.py` / `val_bpb`**（模型 R&D）vs **`research_policy` / `research_score`**（Agent R&D） |
| Radar 与 Research 评分重叠 | L1 评 **方向**；Radar 评 **素材**——harness 分离 |

---

## 近期行动（2 周）

1. 锁定 `research_score` v1（仅 G/K/C）及 5 个 golden fixtures。  
2. 实现 `pulse_knowledge` 最小 API + contract schemas。  
3. 上线 KG-first ReAct（`kg_query`、`vector_search`、`web_search` 经 Capability Router）。  
4. 增加 `research_eval.py` + `research_results.tsv`，对齐 autoresearch 试点工作流。  
5. 与 Penny、Zhan Li 对齐 fixture 维护方与 hold-out 划分。

---

## 本期不做

- 训练专用 Research LLM（Sprint 2）。  
- 调整主流程中 Radar 的顺序或职责。  
- Neo4j 迁移（仅预留接口）。  
- Creative fine-tuning。

---

*与 Vicino World-Model PRD 一致：Research 产出 `Hypothesis` + `CreativeDirection` + KG 增量；Radar 在用户交付前对 Creative 产出做最终同步 gate。*
