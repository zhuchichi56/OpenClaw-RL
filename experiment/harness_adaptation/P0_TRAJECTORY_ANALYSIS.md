# PawBench 24×3 完整轨迹分析

## 结论与证据边界

**结论：QwenPaw 的总分领先主要来自“在长程任务中持续行动并把要求产物真正写入 workspace”，而不是在所有任务上都更聪明；OpenClaw 的优势是短路径和失败恢复，Hermes 的优势是直接工具调用与部分检索/计算任务。三者最值得复用的不是某个 harness 的整条轨迹，而是“产物闭环、阶梯恢复、路径校验、避免无效循环”四类对照经验。**

本报告逐项分析冻结运行 `P0-matrix-24x3-final` 的 24 个任务、3 个 harness、共 72 条完整轨迹。三端使用同一 `custom/qwen3.5-4b-polar` checkpoint、同一 endpoint、temperature 0 和同一 deterministic grader；每个 cell 只有一次运行，因此结果能说明这组任务上的 harness sensitivity，不能估计跨 seed 稳定性，也不能证明某个 prompt/tool 差异的单独因果效应。

| Harness | 均分 | 满分 | 独占最高 | 平均模型 turn | 平均 tool call | 发生写操作的任务 | 估算总 token |
|---|---:|---:|---:|---:|---:|---:|---:|
| QwenPaw | **0.8075** | **8/24** | **6** | 11.54 | 11.83 | **20/24** | **132,711** |
| OpenClaw | 0.7387 | 5/24 | 3 | 4.08 | 4.42 | 16/24 | 64,012 |
| Hermes | 0.7003 | 5/24 | 3 | 6.33 | 5.71 | 15/24 | 72,607 |

QwenPaw 对 OpenClaw 为 9 胜/9 平/6 负，均分差 `+0.0688`；对 Hermes 为 11 胜/8 平/5 负，均分差 `+0.1072`。但这不是 compute-matched 比较：QwenPaw 实际 token 是 OpenClaw 的 2.07 倍、Hermes 的 1.83 倍。因此可支持的说法是“QwenPaw 的控制循环更愿意投入计算并完成产物”，不能说“同等推理计算下 QwenPaw 纯能力更强”。

## 为什么 QwenPaw 总体领先，而另外两端会在局部更好

| 证据切片 | n | QwenPaw | OpenClaw | Hermes | 解释 |
|---|---:|---:|---:|---:|---|
| ClawEval | 17 | **0.8931** | 0.8828 | 0.7685 | QwenPaw 与 OpenClaw 基本相当，非主要差距来源 |
| QwenClawBench | 4 | **0.5898** | 0.0000 | 0.4951 | 总体差距的核心；QwenPaw 4/4 写产物，OpenClaw 0/4，Hermes 3/4 |
| PinchBench | 3 | 0.6125 | **0.9069** | 0.5875 | OpenClaw 的文档处理与恢复路径显著更好 |

QwenPaw 的核心优势是**完成定义更接近 grader 的完成定义**。它的短 system message 明确规定“没有工具调用就表示任务完成”，轨迹中观察到 7 类核心工具，控制循环通常会继续到文件落盘。在社交档案、套利监控和 House Robber 三个长程任务中，OpenClaw 都在读完材料后停止，要求文件不存在而直接得 0；QwenPaw 则继续写文件，分别得到 0.812、0.680、0.867。这里不是知识差距，而是“读懂了”有没有转化为 grader 可见状态的差距。

但持续行动并不等于高质量推理。QwenPaw 出现 52 次连续完全相同的工具调用；社交档案任务达到 65 turn/71 calls，并多次重写同一 JSON，最长轨迹达到 67 turn。其 turn 数与分数的相关系数反而为 `-0.20`；OpenClaw 和 Hermes 也接近 0。这说明有效机制是**状态有进展的坚持**，不是无条件增加 turn。QwenPaw 在 PDF 任务中反复尝试 browser/shell 仍未生成 `answer.txt`，而 OpenClaw 用 17 turn 完成 `pdftotext → 安装失败 → 隔离 venv → PyPDF2 → 抽取 → 写文件`，得到 0.833。OpenClaw 还用 4 turn 完整完成投资优先级矩阵，而 QwenPaw 用 20 turn 仍漏掉两个 composite score（0.778）。这两例证明“最短正确恢复路径”可以胜过长循环。

OpenClaw 的优势集中在**简洁求解和有层次的环境恢复**：它在投资矩阵、newsletter、PDF comprehension 三题独占最高。但其轨迹 metadata 显示 system prompt 约 29,108 字符、声明 25 个工具、注入大量 workspace/skill 说明，同时 `thinkLevel=off`、`reasoningLevel=off`。对 4B 模型而言，这与短轨迹和长程任务早停一致，但现有数据只能给出相关性，不能单独归因于 prompt 长度、工具数量或关闭 reasoning 中的某一项。更多工具也没有自动转化成更多有效动作。

Hermes 的优势集中在**直接的 read/write/execute 路径**：它在项目成本分析、费用报销和知识库检索三题独占最高，并能在两个长程任务中持续写出产物。其主要损失不是统一的能力不足，而是路径与 runtime 稳定性：日历任务把结果写到 grader 不可见的绝对路径 `/workspace/output/...`，只得 0.75；投资矩阵和现金流任务到进程时限后留下 `KeyboardInterrupt`，sector 任务也到 900 秒。runner 末尾的 `|| true` 使这些进程仍被记为 `status=success`，因此必须将 runtime failure 与模型任务失败分开。后验排除两次明确 timeout 后，22 题均分为 QwenPaw 0.8001、OpenClaw 0.7149、Hermes 0.7640，QwenPaw–Hermes 差距从 0.107 缩至 0.036；这只是诊断，不替代正式 24 题结果。Hermes 的完整 system prompt 未可靠封存在标准轨迹中，因此 prompt 级归因保持 Unknown。

## 关键对照轨迹与失效机制

| 同题对照 | 实际轨迹 | 机制判断 |
|---|---|---|
| 社交档案 | QwenPaw 0.812（65 turn）和 Hermes 0.781（44）持续写 JSON；OpenClaw 0（2）读若干文件后停止、无输出 | **终止条件错误**：完成阅读不等于完成任务；required artifact 必须存在 |
| PDF comprehension | QwenPaw 0（14）多路尝试但未落盘；OpenClaw 0.833（17）创建 venv、安装 PyPDF2、抽取并写答案；Hermes 0（1）一次读取后停止 | **恢复质量**：改变工具/环境并获得新证据，比重复尝试更重要 |
| 投资优先级矩阵 | QwenPaw 0.778（20）漏两项分数；OpenClaw 1.0（4）完整计算；Hermes 0（runtime traceback） | **闭环校验**：长轨迹不保证字段完整；runtime 错误不可混作策略经验 |
| 日历安排 | QwenPaw/OpenClaw 1.0；Hermes 0.75 | **路径协议**：写到错误绝对路径等价于 grader 看不到产物 |
| House Robber | QwenPaw 0.867、Hermes 0.833 均创建两个要求文件；OpenClaw 0，读完材料却未写 | **产物清单**：停止前必须逐项核对所有输出，而非只核对答案内容 |

整体上，差距可以压缩为四个控制变量：`是否继续行动到产物落盘` 决定长程任务的下限；`恢复是否产生新证据` 决定受阻任务能否翻盘；`路径/schema/字段是否验证` 决定已完成工作的最后一段得分；`runtime 是否稳定` 决定轨迹能否被解释为模型行为。QwenPaw主要赢第一项，OpenClaw主要赢第二项和部分短路径任务，Hermes在直接操作上有效但被第三、第四项拖累。

## 可复用经验：如何服务 Polar experience learning

下面是从对照轨迹中提取的**候选经验规则**；它们有单题或小样本机制证据，但尚未通过训练后 held-out 测试，因此不能写成已验证的普遍规律。

| 经验规则 | 可执行形式 | 正例/反例来源 |
|---|---|---|
| Artifact completion invariant | 停止前列出 required outputs；确认每个文件位于 grader 可见 workspace、可重新读取且关键字段存在 | Qwen/Hermes 写出长程产物 vs OpenClaw 读完即停 |
| Write early, refine later | 长任务先创建 schema 正确的最小产物，再逐段补全，避免预算耗尽后 0 产出 | 社交档案、House Robber |
| Canonical workspace rule | 开始时确认 workspace root；优先相对路径；写后用相同路径重新读取 | Hermes 日历错误绝对路径 |
| Recovery ladder | 专用工具失败后依次尝试本地 CLI、Python 库、隔离 venv；只有获得新观察才进入下一步 | OpenClaw PDF 成功 vs QwenPaw 盲目重试 |
| Loop breaker | 禁止无新观察地重复完全相同动作；重复前必须改变假设、输入或工具 | QwenPaw 52 次连续重复调用 |
| Verifier-driven closure | 根据要求检查文件存在性、JSON/schema、关键字段、数值与小型 reproducer，再终止 | OpenClaw 投资矩阵满分 vs QwenPaw 漏字段 |
| Runtime/policy separation | timeout、adapter 崩溃、路径同步错误单独标记，不直接蒸馏成模型决策经验 | Hermes A02/A03/sector |
| Budget-aware persistence | 仅当 workspace 状态持续改善时继续；否则触发换方法或最小交付 | QwenPaw 长程优势及过度循环并存 |

Polar 不应直接蒸馏 65-turn 的原始轨迹；更合适的是构造同题对照的 experience card：`触发状态 → 失败动作 → 环境证据 → 诊断 → 最小恢复动作 → verifier → 适用边界`。优先级最高的三组卡片是：OpenClaw“读完即停”对 Qwen/Hermes“完成文件”；QwenPaw“PDF 盲目重试”对 OpenClaw“阶梯恢复”；Hermes“错误绝对路径”对正确 workspace 路径。训练后必须在未见任务、不给经验文本的条件下验证：required-artifact 缺失率、无新证据重复率、恢复成功率和最终 deterministic score 是否改善；否则只能证明 prompt assistance，不能证明经验被内化。

## 24 题逐题账本

括号内是模型 turn 数，一个 turn 定义为一条 assistant-role message；“最高”允许并列。逐题 breakdown 与完整原始轨迹位于同一 run root。

| Task | 来源 | QwenPaw 分数(turn) | OpenClaw 分数(turn) | Hermes 分数(turn) | 最高端/主要判定 |
|---|---|---:|---:|---:|---|
| `CTB_A01_financial_reconciliation` | ClawEval | 0.700 (2) | 0.800 (3) | 0.800 (2) | O/H；Q 漏 summary count，三端 total impact 均错 |
| `CTB_A02_investment_priority_matrix` | ClawEval | 0.778 (20) | **1.000 (4)** | 0.000 (1) | O；Q 漏 composite scores，H runtime failure |
| `CTB_A03_cashflow_risk_memo` | ClawEval | **1.000 (2)** | **1.000 (3)** | 0.000 (1) | Q/O；H runtime failure |
| `CTB_DATA_20_project_cost_vs_plan` | ClawEval | 0.417 (6) | 0.583 (14) | **0.667 (10)** | H；三端均有计算/引用缺项 |
| `CTB_MGMT_02_budget_allocation_proposal` | ClawEval | **1.000 (2)** | 0.833 (2) | 0.833 (2) | Q；O/H 来源与 reduction 说明不全 |
| `CTB_OPS_04_cross_team_dependency_map` | ClawEval | **1.000 (6)** | 0.945 (2) | 0.833 (3) | Q；O 风险分析不全，H 缺来源 |
| `CTB_SALES_10_key_account_health` | ClawEval | **0.917 (6)** | 0.667 (2) | 0.667 (2) | Q；O/H 风险标签和 expansion 不全 |
| `T002_email_triage` | ClawEval | 0.944 (5) | 0.944 (3) | 0.944 (3) | 并列；reason 均略不完整 |
| `T003zh_calendar_scheduling` | ClawEval | **1.000 (4)** | **1.000 (5)** | 0.750 (8) | Q/O；H 写到错误绝对路径 |
| `T006_email_reply_draft` | ClawEval | **1.000 (4)** | 0.950 (3) | **1.000 (3)** | Q/H；O 未充分读取多份材料 |
| `T011zh_expense_report` | ClawEval | 0.875 (5) | 0.875 (2) | **0.958 (5)** | H；Q/O 总额错误，H 类别略缺 |
| `T012_expense_report` | ClawEval | **1.000 (20)** | 0.875 (3) | **1.000 (4)** | Q/H；O 总额错误 |
| `T016_kb_search` | ClawEval | 0.857 (3) | 0.857 (2) | **1.000 (3)** | H；Q/O 缺 cross-reference |
| `T017zh_ticket_triage` | ClawEval | **1.000 (5)** | **1.000 (4)** | **1.000 (3)** | 三端并列满分 |
| `T022_newsletter_curation` | ClawEval | 0.917 (4) | **1.000 (4)** | 0.833 (3) | O；Q/H 混入 irrelevant item |
| `T097_pinbench_eli5_model_summary` | ClawEval | **1.000 (4)** | 0.900 (4) | **1.000 (3)** | Q/H；O 内容略损 |
| `T098_pinbench_openclaw_facts` | ClawEval | 0.778 (4) | 0.778 (4) | 0.778 (3) | 三端并列 |
| `task_00052_generate_openai_social_media_profile_from_workspace_data` | QwenClawBench | **0.812 (65)** | 0.000 (2) | 0.781 (44) | Q；O 无产物，Q/H 部分字段错误 |
| `task_00068_enhanced_poly_arb_monitoring_script_execution` | QwenClawBench | **0.680 (67)** | 0.000 (4) | 0.366 (14) | Q；O 无有效产物，H 多项检查不全 |
| `task_00075_sector_momentum_rotation_backtest_with_data_quality_traps` | QwenClawBench | 0.000 (16) | 0.000 (1) | 0.000 (1) | 三端失败；H 到 900 秒限制 |
| `task_00100_house_robber_algorithm_deep_dive_explanation` | QwenClawBench | **0.867 (7)** | 0.000 (3) | 0.833 (27) | Q；O 无两个要求文件 |
| `task_blog` | PinchBench | 0.900 (2) | **0.950 (2)** | **0.950 (2)** | O/H；主要差异为长度 |
| `task_meeting_blog_post` | PinchBench | **0.938 (4)** | **0.938 (5)** | 0.812 (4) | Q/O；H 长度与语气不足 |
| `task_openclaw_comprehension` | PinchBench | 0.000 (14) | **0.833 (17)** | 0.000 (1) | O；只有 O 恢复 PDF 并写出答案 |

## 可复现证据

- 机器可读统计：`experiment/harness_adaptation/runs/P0-matrix-24x3-final/trajectory_analysis.json`
- 分析脚本：`scripts/harness_adaptation/analyze_pawbench_trajectories.py`
- 原始轨迹：`experiment/harness_adaptation/runs/P0-matrix-24x3-final/{qwenpaw,openclaw,hermes}/transcripts/`
- 冻结协议：`experiment/harness_adaptation/pawbench_pilot_manifest.json`
- Phase 0 主结果：`experiment/harness_adaptation/P0_REPORT.md`
