# 实验室危险场景识别与自进化 Agent

工科导论合作课题实现：**基于多模态大模型与自进化智能体的危险场景识别与预警系统**

聚焦场景：**高校化学 / 材料实验室**。系统输入实验室图片或短视频，输出风险类别、位置/时间段、
风险等级、可见证据、处理建议与不确定性说明，并通过人工纠错持续改进策略。

---

## 一、5 分钟跑起来

```bash
cd <项目目录>

# 1. 看看当前环境（会告诉你有没有配置 API Key）
python src/main.py status

# 2. 用模拟模式分析一张图（无需任何 API Key 和第三方库）
python src/main.py analyze --input data/dev/ppe_example.jpg --scenario chemistry_lab --mock

# 报告在 reports/ppe_example_report.md 和 .json
```

启动可操作的 Web 界面（需要 `pip install flask`）：

```bash
python src/main.py serve --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

直接双击打开 `ui/index.html` 也能用：检测不到后端时会自动进入**离线演示模式**，
用内置示例展示完整界面流程（数据是示例数据，会明确标注）。

## 二、接入真实多模态模型

系统对接任意 **OpenAI 兼容**的视觉模型接口。

Windows PowerShell：

```powershell
$env:SAFETY_LLM_API_KEY  = "你的 API Key"
$env:SAFETY_LLM_BASE_URL = "https://你的服务地址/v1"
$env:SAFETY_LLM_MODEL    = "你的视觉模型名"
```

macOS / Linux：

```bash
export SAFETY_LLM_API_KEY="你的 API Key"
export SAFETY_LLM_BASE_URL="https://你的服务地址/v1"
export SAFETY_LLM_MODEL="你的视觉模型名"
```

然后照常运行命令即可，系统会自动从模拟模式切到真实模式：

```bash
python src/main.py analyze --input data/test/lab_0001.jpg
```

> 未配置 Key 时系统**自动降级为模拟模式**，并把所有结论标记为 `simulated`，
> 绝不会把模拟结果冒充真实识别结论。

## 三、命令一览

| 命令 | 作用 |
| --- | --- |
| `status` | 查看运行环境（是否已接入模型、视频支持、案例数、版本） |
| `analyze` | 分析图片或视频，输出 JSON + Markdown 报告 |
| `feedback` | 记录一次人工纠错，并生成策略草案 |
| `promote` | 把草案晋升为当前生效策略（会把新规则写入规则库） |
| `rollback` | 回退到历史策略版本 |
| `versions` | 列出所有策略版本 |
| `evaluate` | 按标注 CSV 计算精确率 / 召回率 / 误报 / 漏报 / 成本 |
| `serve` | 启动本地 Web 界面 |

`analyze` 的消融开关（用于课程要求的“关闭一项关键功能进行对比”）：

```bash
python src/main.py analyze --input data/test/x.jpg --no-review         # 关闭证据复核 Agent
python src/main.py analyze --input data/test/x.jpg --no-rules          # 关闭规则检索
python src/main.py analyze --input data/test/x.jpg --no-memory         # 关闭案例记忆
python src/main.py analyze --input data/test/x.jpg --no-agent-tools    # 关闭 Agent 自主工具调用
python src/main.py analyze --input data/test/x.jpg --mode baseline     # 基础对照系统（单模型直出）
```

## 四、系统架构

```text
图片 / 短视频
    │
    ├─ 视频抽帧（按时间轴均匀采样 4 帧）
    ▼
┌──────────────────────────────────────────────┐
│ 场景感知 Agent                                │
│  看图 → 描述场景 → 提出候选风险                │
│  ↕ 自主工具调用循环（最多 2 轮）               │
│    crop_region / detect_objects /            │
│    search_rules / search_cases               │
└──────────────────────────────────────────────┘
    ▼
┌──────────────────────────────────────────────┐
│ 风险分析 Agent                                │
│  候选风险 + 检索到的安全规则 + 历史纠错案例     │
│  → 风险等级 / 处理建议 / 置信度 / 依据         │
└──────────────────────────────────────────────┘
    ▼
┌──────────────────────────────────────────────┐
│ 证据复核 Agent（质疑者立场）                   │
│  → confirmed / uncertain / rejected          │
│  → 与风险分析 Agent 的分歧检测与仲裁           │
└──────────────────────────────────────────────┘
    ▼
跨帧时间融合（持续帧数 / 出现比例 / 合并重复预警）
    ▼
风险报告（JSON + Markdown）+ Web 界面展示
    ▼
人工纠错 → 案例记忆 → 策略进化 Agent 提案 → 组员确认 → 新版本 → 复测 → 可回退
```

### 三个 Agent 的分工与输入输出

| Agent | 输入 | 输出 | 意见不一致时 |
| --- | --- | --- | --- |
| 场景感知 Agent | 图像（含裁剪后的局部图）、场景说明、策略提示词、工具清单 | 场景描述、候选风险（含证据与位置）、工具调用申请 | — |
| 风险分析 Agent | 候选风险、检索到的规则、历史纠错案例 | 风险等级、处理建议、置信度、判定依据（规则 id / 案例 id） | 无规则支持时强制给 `uncertain`，不硬编常识 |
| 证据复核 Agent | 风险分析的结论、证据、置信度 | `confirmed` / `uncertain` / `rejected` + 理由 | 记录分歧、给出仲裁结论，高风险分歧标记为「需人工优先复核」 |

### 意见冲突的处理方式

复核 Agent 是**质疑者**，不附和。当两个 Agent 结论不一致时：

1. 在风险条目上记录 `disagreement`（谁说了什么）与 `review_verdict`；
2. 按保守原则仲裁：复核判 `rejected` → 记为误报并交人工确认；复核判 `uncertain` → 降级为不确定；
3. 若原本是 `high` 却被降级，或直接判误报，则额外标记 `needs_human_review`，进入人工优先复核队列；
4. 报告里单列「Agent 意见冲突与仲裁」表格，可追溯每一处分歧。

### Agent 自主工具调用

工具清单会作为提示词的一部分交给感知 Agent，由 Agent 自己决定要不要调、调哪个：

| 工具 | 用途 |
| --- | --- |
| `crop_region` | 把可疑区域裁剪放大后重新判断（裁剪图会作为新图像回灌给模型） |
| `detect_objects` | 目标检测（需要 `ultralytics`，未安装时该工具返回不可用，不影响其他功能） |
| `search_rules` | 检索安全规则，确认判定依据 |
| `search_cases` | 检索历史纠错案例，参考人工先例 |

每轮工具执行结果会作为观察回灌给 Agent，最多 2 轮，全部调用写入 `reports/tool_calls.jsonl`。

## 五、视频的时间维度分析

课程要求“视频分析应结合多个时间点的信息”。本系统按时间轴均匀抽帧后逐帧分析，
再做**跨帧时间融合**：

- 记录同一风险出现在哪些帧、持续帧数与出现比例（`persistence`）；
- 只在极少数帧一闪而过 → 判定可能是瞬时误检，降置信度并降级为 `uncertain`；
- 在多数帧中持续出现 → 证据更强，升置信度；
- 同类风险跨帧合并为一条，避免重复预警。

报告里会显示类似 `第 2~4 帧（持续 3/4 帧，占 75%）` 的时间维度说明。

## 六、自进化是怎么做的

课程要求“由 Agent 提出提示词、规则或案例记忆的修改方案，经组员确认后更新系统”。

```bash
# 1. 记录一次人工纠错
python src/main.py feedback \
  --report reports/sessions/xxx/xxx_report.json \
  --original-risk-type ppe_missing \
  --correct-risk-type food_or_drink \
  --correct-severity low \
  --note "画面中是水杯放在实验台边缘，不是未佩戴护目镜。"

# 2. 系统会：写入案例 → 聚合误判模式 → 策略进化 Agent 提案 → 生成 v0.x 草案
# 3. 组员确认后晋升
python src/main.py promote --version v0.3

# 效果不好就回退
python src/main.py rollback --version v0.2
```

进化方案有三种载体：

- **易混淆说明**（`risk_type_notes`）：把「A 被误判为 B」沉淀成显式约束，注入感知提示词；
- **提示词补丁**（`prompt_patch`）：追加具体可执行的感知要求（每轮替换而非累加，避免无限膨胀）；
- **新增规则**（`proposed_rules`）：晋升时真正写入 `rules/lab_safety_rules.json`。

每次版本变更都会在 `change_log` 中记录：改了什么、为什么改、依据哪些纠错案例。
**只有具体的策略更新才算完成自进化**——单纯让模型“重新思考”在本系统中不算。

## 七、实验与评测

一键跑通课程要求的三组对比：

```bash
# 先准备数据（占位数据仅用于跑通流程）
python scripts/make_demo_dataset.py --count 80 --video

# 基础系统 / 多 Agent / 两轮自进化后，三组对比
python scripts/run_experiment.py \
  --labels data/demo_labels_test.csv \
  --dev-labels data/demo_labels_dev.csv \
  --groups base,multi --auto-evolve --rounds 2
```

输出：`reports/experiments/comparison.md`（含精确率、召回率、误报漏报、处理时间、调用成本）
和 `comparison.json`。

> ⚠️ **模拟模式下的对比表不能当实验结果提交**。模拟数据是按文件名驱动的，
> 三组指标会趋于一致，脚本会在表头自动加上警示。接入真实 API 后重跑才是有效数据。

单独评测：

```bash
python src/main.py evaluate --labels data/labels_test.csv \
  --reports reports/experiments/base reports/experiments/multi reports/experiments/evolved \
  --names 基础系统 多Agent版 两轮自进化版 \
  --out-md reports/eval.md --out-json reports/eval.json
```

## 八、目录结构

```text
├── src/                    核心代码
│   ├── main.py             命令行入口
│   ├── pipeline.py         主流程：预处理 → 三 Agent → 时间融合 → 报告
│   ├── agents.py           三个协作 Agent + 基础对照系统
│   ├── toolkit.py          工具注册表（Agent 自主工具调用的载体）
│   ├── tools.py            规则检索 / 案例记忆 / 工具日志 / 视频抽帧 / 局部裁剪
│   ├── llm.py              模型客户端（真实 API + 模拟模式）
│   ├── self_evolution.py   案例记录 / 聚合 / 策略进化 Agent / 版本管理
│   ├── evaluate.py         多标签评测与多组对比
│   └── models.py           数据结构
├── app/server.py           Web 服务（上传→分析→纠错→版本→报告）
├── ui/index.html           可操作界面（连后端；无后端时自动离线演示）
├── configs/                策略版本（strategy_current.json 为当前生效）
├── versions/               历史策略版本
├── rules/                  安全规则库（每条带权威来源）
├── memory/cases.json       人工纠错案例库
├── data/                   数据集与标注（详见 docs/数据说明.md）
├── reports/                分析报告 + tool_calls.jsonl 工具调用日志
├── scripts/                数据集生成 / 对比实验 / 重置
├── tests/                  单元测试（python -m pytest tests -q）
└── docs/                   方案、执行清单、数据说明、报告大纲、分工、失败案例
```

## 九、测试与重置

```bash
python -m pytest tests -q                   # 运行测试

python scripts/reset_demo_state.py          # 预演：看看会清掉什么
python scripts/reset_demo_state.py --yes    # 确认：清掉占位数据产生的版本与案例
```

跑完占位数据集的演示后建议执行一次重置，避免把模拟数据当成真实实验记录。

## 十、重要提醒

- **不得制造危险情境采集数据**；数据须获得授权并保护隐私。
- 安全规则已注明来源，但具体条款请以学校现行实验室安全手册与标准最新版本为准。
- 系统用于教学与辅助分析，**结果必须人工复核**。
- **“未识别到风险”不等于“场景安全”**。
- 模拟模式的输出不构成任何真实识别结论。
