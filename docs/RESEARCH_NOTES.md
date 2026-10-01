> Historical full-package research notes. Public data/output archives are deliberately omitted; some historical commands require separately licensed datasets. The VeighNa execution adapter is dry-run by default and not verified against a live brokerage gateway.

# SentimentCycle Alpha
## 多智能体 A 股情绪周期与交易执行系统

**Multi-Agent A-Share Sentiment Cycle & Execution System**

**版本：7.9.0（V7 Final）**

---

## 1. 项目简介

SentimentCycle Alpha 是一个面向 **A 股情绪周期研究、个股联动建模、实时仓位管理与交易执行** 的多智能体量化研究系统，并以 **中芯国际（688981.SH）** 作为重点案例。

项目核心目标不是单独预测某只股票下一周期涨跌，而是将多源市场信息整合为一条完整、可追踪、可回测、可实时刷新、可执行的研究链路：

```text
信息采集
    ↓
内容解析
    ↓
Prompt Engineering 结构化评分
    ↓
全 A 市场情绪
    ↓
板块情绪
    ↓
板块—个股联动
    ↓
日频 Alpha
    ↓
动态仓位管理
    ↓
5min 盘中确认
    ↓
T+1 / Next-Bar / 交易成本 / 风控
    ↓
最终执行指令
    ↓
VeighNa CTA 执行适配
```

项目主要覆盖：

- 全 A 市场情绪与市场周期研究
- 半导体板块情绪与中芯国际联动
- 多智能体决策与 Bull–Bear Debate
- 新闻事件解析与 LLM / Prompt Engineering
- 动态仓位管理
- 5 分钟级盘中确认
- A 股 T+1 约束
- Next-Bar Execution
- 交易成本与风险控制
- VeighNa CTA 执行适配

---

## 2. 研究目标

原始研究目标可以概括为：

> **A 股情绪周期研究：获取东财等平台数据，构建“板块情绪—个股”联动模型，开发基于实时情绪周期的动态仓位管理，结合主观信息与盘面数据构建合理的动量突破与盘中确认机制。以中芯国际为例，实现信息获取 → 内容解析 → 结构化评分（Prompt Engineering）→ 交易信号生成 → 交易执行 → 风控约束的完整链路。**

---

## 3. 系统总体架构

```text
                    ┌──────────────────────────────┐
                    │        信息采集层             │
                    │ Eastmoney / AKShare          │
                    │ BaoStock / Tushare           │
                    │ News / Market / ETF / 5min   │
                    └──────────────┬───────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │        内容解析层             │
                    │ 新闻清洗 / 事件抽取           │
                    │ Prompt Engineering           │
                    │ LLM Structured Output        │
                    └──────────────┬───────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │       情绪与联动建模层        │
                    │ Full-A Market Sentiment      │
                    │ Market Cycle                 │
                    │ Sector Sentiment             │
                    │ Sector-Stock Linkage         │
                    └──────────────┬───────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │       日频 Alpha / 仓位层     │
                    │ V7.7 Frozen Daily Model      │
                    │ V7.8 Market Overlay          │
                    │ Dynamic Position Multiplier  │
                    └──────────────┬───────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │       盘中执行确认层          │
                    │ VWAP                         │
                    │ 5min / 15min Momentum        │
                    │ RVOL                         │
                    │ Opening Range Breakout       │
                    │ Intraday Volatility Control  │
                    └──────────────┬───────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │       风控与执行约束          │
                    │ T+1                          │
                    │ Next-Bar Execution           │
                    │ Transaction Cost             │
                    │ Event Risk                   │
                    │ Position Cap                 │
                    └──────────────┬───────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │       最终执行适配层          │
                    │ Final Execution Instruction  │
                    │ VeighNa CTA Adapter          │
                    │ Dry-Run / Simulated Order    │
                    └──────────────────────────────┘
```

---

## 4. 研究原则

本项目强调 **Research Discipline**，主要包括：

- 严格区分研究期与鲁棒性期
- 避免未来函数和数据泄漏
- 不使用当前板块成分股反填历史
- 不利用已观察的 2025 数据反向选择模型
- 模拟数据仅用于工程 Smoke Test
- 真实策略结论只基于真实历史数据
- LLM 不直接产生 BUY / SELL / HOLD
- LLM 仅负责文本理解和结构化语义提取
- 最终事件分数由 Python 确定性计算
- 不为了“把收益调正”而无限调参
- 每个新模块必须通过独立数据 Gate 才允许进入主链

---

## 5. 数据源

| 数据类型 | 主要来源 | 用途 |
|---|---|---|
| 全 A 日频横截面 | Tushare / 历史 PIT 数据 | 市场情绪、Breadth、Return、Dispersion |
| 实时 A 股行情 | Eastmoney / AKShare | 盘中实时市场情绪刷新 |
| 半导体板块代理 | 512480 半导体 ETF / Sina-AKShare | 板块情绪与中芯国际联动 |
| 中芯国际日频 | BaoStock | 个股历史日线与 EOD 补充 |
| 中芯国际 5min | BaoStock | V7.9 Intraday Confirmation |
| 中芯国际新闻 | Eastmoney / AKShare | Step C 新闻事件分析 |
| 宽基指数 EOD Bridge | 000300 等指数 | 最新 EOD market-return overlay 续接 |
| LLM API | DeepSeek | Prompt Engineering / Event Structuring |

> 注意：项目不会把 ETF 代理、宽基指数代理或模拟数据伪装成完整全市场历史横截面。

---

## 6. 环境安装

### 6.1 Python 环境

建议使用：

```text
Python 3.10+
```

安装依赖：

```powershell
pip install -r requirements.txt
```

核心依赖包括：

```text
pandas
numpy
PyYAML
matplotlib
tushare
akshare
baostock
openai
pytest
```

VeighNa 属于可选执行环境，不建议与研究环境强行共用同一个 Python 环境。

---

## 7. 配置文件

建议使用：

```text
config/
├── settings.yaml
├── openai_config.example.py
└── tushare_token.example.py
```

使用前将示例文件复制为：

```text
config/openai_config.py
config/tushare_token.py
```

并填入自己的 API Key / Token。

示例：

```python
OPENAI_API_KEY = "YOUR_DEEPSEEK_API_KEY"
OPENAI_BASE_URL = "https://api.deepseek.com"
OPENAI_MODEL = "deepseek-v4-flash"
```

```python
TUSHARE_TOKEN = "YOUR_TUSHARE_TOKEN"
```

> 不应将真实 API Key 或 Token 上传到公开仓库。

---

## 8. 版本演进

### 8.1 V7.6 — A 股情绪周期基础系统

V7.6 建立了：

- A 股情绪周期
- 板块情绪
- 个股联动
- 动态仓位
- 动量突破
- 连续价格与异常断层检查
- 基础研究回测

核心结果：

| 指标 | V7.6 |
|---|---:|
| Annual Return | 7.27% |
| Sharpe | 0.519 |
| Max Drawdown | -31.40% |
| IC5 | 0.0190 |
| 2025 Annual | 17.29% |
| 2025 Sharpe | 1.198 |

---

### 8.2 V7.7 — Frozen Daily Champion

V7.7 在 V7.6 基础上进一步强化状态转换研究与冻结机制，并成为后续版本复用的日频 Champion。

| 区间 | Annual | Sharpe | MaxDD |
|---|---:|---:|---:|
| Full Sample | 9.22% | 0.563 | -32.27% |
| Development 2020–2024 | 7.65% | 0.482 | -32.27% |
| 2025 Robustness | 20.06% | 1.094 | -10.73% |

后续版本默认：

```text
V7.7 股票排序 / 日频信号逻辑保持冻结
```

---

### 8.3 V7.8 — Real A-Share Market Overlay

V7.8 重点研究真实全 A 市场环境对 Frozen Daily Alpha 的增量价值。

研究因子包括：

- Market Return
- Breadth
- Dispersion
- Volume
- Extreme Breadth
- Full Sentiment Fusion
- Sector Overlay

Development-only ablation 最终选择：

```text
v7_8_market_return_only
```

作为 V7.8 Final Challenger。

| 区间 | Annual | Sharpe | MaxDD |
|---|---:|---:|---:|
| Full Sample | 10.56% | 0.638 | -29.74% |
| Development 2020–2024 | 8.60% | 0.538 | -29.74% |
| 2025 Robustness | 15.19% | 0.920 | -11.59% |

项目保留：

```text
Champion   = v7_7_frozen
Challenger = v7_8_market_return_only
```

即使 V7.8 全样本表现更高，也不会自动覆盖已冻结 Champion。

---

## 9. Step A — 真实板块情绪 → 个股联动

Step A 补齐真实历史 Sector → Stock Linkage。

```text
半导体板块
    ↓
Sector Return
Sector Liquidity
    ↓
Sector Strength Score
    ↓
中芯国际 Relative Strength
    +
Sector-Stock Rolling Correlation
    ↓
Sector-Stock Link Score
```

最终数据 Gate：

```text
sector_gate=True
coverage=99.9%
```

Sector 信息主要作为：

```text
Challenger
Diagnostic
Realtime Context
```

不强行替换冻结的最终市场 Overlay。

---

## 10. Step B — 实时情绪周期与动态仓位

入口：

```powershell
python run_realtime_sentiment.py
```

典型输出：

```text
market_gate=True
overlay_gate=True

selected_variant=v7_8_market_return_only

component_score=...
phase=...
transition=...
transition_confidence=...

position_multiplier=...
```

### 盘前 / 集合竞价

如果完整全 A 当日横截面尚不可用：

```text
LATEST_EOD_BROAD_INDEX_RETURN_PROXY
```

用于续接 frozen `market_return_only`。

系统显式保持：

```text
full_market_diagnostic_gate=False
```

以避免把宽基指数代理伪装成完整全 A Breadth 情绪。

### 盘中

当 Eastmoney 全 A 实时横截面正常时：

```text
REALTIME_INTRADAY_POINT_IN_TIME
```

系统可以重新计算：

- Breadth
- Return
- Dispersion
- Volume
- Market Sentiment
- Cycle Phase
- Dynamic Position Multiplier

---

## 11. Step C — Prompt Engineering / LLM Event Structuring

Step C 将新闻与主观事件信息正式接入量化链。

```text
Eastmoney News
      ↓
Text Cleaning
      ↓
Prompt Engineering
      ↓
DeepSeek LLM
      ↓
Structured JSON
      ↓
Deterministic Python Score
      ↓
Event Risk Multiplier
```

LLM 输出示例：

```json
{
  "event_type": "company_fundamental",
  "direction": 1.0,
  "relevance": 0.9,
  "surprise": 0.8,
  "persistence": 0.7,
  "stock_relevance": 0.9,
  "sector_relevance": 0.8,
  "confidence": 0.95,
  "horizon_days": 10,
  "summary": "..."
}
```

### LLM Governance

LLM 不负责：

```text
BUY
SELL
HOLD
目标价
目标仓位
收益预测
```

真正的：

```text
structured_event_score
```

由 Python 根据结构化字段确定性计算。

事件风险乘数：

```text
event_risk_multiplier ∈ [0.85, 1.00]
```

规则：

- 正面新闻不会自动增加杠杆
- 高置信负面事件可以降低风险暴露
- LLM 不能直接控制最终仓位

---

## 12. V7.9 — 中芯国际真实 5min Intraday Confirmation

真实数据：

```text
Symbol: 688981.SH
Period: 2020-07-16 ~ 2025-12-31
Bars: 63,408
Trading Days: 1,321
Median Bars/Day: 48
Daily Signal Coverage: 99.1%
quality=REAL_5MIN_HISTORICAL
intraday_gate=True
```

### Intraday Features

- VWAP Strength
- 5min Momentum
- 15min Momentum
- RVOL
- Opening Range Breakout
- Intraday Volatility Control

### 执行约束

- A 股 T+1
- 当前 5min bar close 生成信号
- 最早下一根 bar open 执行
- 加入交易成本
- 模拟 5min 数据仅用于 smoke test

---

## 13. V7.9 实验结果

### 13.1 Full Sample

| Variant | Annual | Sharpe | MaxDD | Trades |
|---|---:|---:|---:|---:|
| V7.8 Open Baseline | -0.81% | -0.260 | -6.69% | 358 |
| **V7.9 Full** | **-0.49%** | **-0.176** | **-6.11%** | **282** |

研究结论：

> 5min Intraday Confirmation 对中芯国际单股执行具有增量，但不能凭空创造 Daily Alpha。

### 13.2 Development Ablation

主要结论：

- VWAP：明显正贡献
- Breakout：明显正贡献
- Volatility Control：正贡献
- RVOL：小幅正贡献
- Momentum：边际作用较弱，但暂无充分证据删除
- `entry_confirmation_only` 被实验否决
- `v7_9_full` 仍为最佳盘中执行版本

最终保留：

```text
V7.9 Intraday Execution Champion = v7_9_full
```

---

## 14. 最终执行指令

最终研究系统将多个模块压缩为一个统一执行指令。

入口：

```powershell
python build_final_execution_instruction.py --root "."
```

输出：

```text
outputs/v7_final/final_execution_instruction.json
```

格式示例：

```json
{
  "version": "V7_FINAL",
  "symbol": "688981.SH",
  "vt_symbol": "688981.SSE",
  "base_target_weight": 0.0578,
  "cycle_multiplier": 0.7941,
  "event_risk_multiplier": 1.0,
  "intraday_action": "WAIT",
  "final_target_weight": 0.0459,
  "market_gate": true,
  "sector_gate": true,
  "event_gate": true,
  "intraday_gate": true,
  "quality": "V7_FINAL_EXECUTION_READY"
}
```

基本逻辑：

```text
FinalTargetWeight
=
BaseTargetWeight
× CycleMultiplier
× EventRiskMultiplier
```

并结合：

```text
ENTER / WAIT / REDUCE / EXIT
```

控制盘中仓位变化。

---

## 15. VeighNa CTA Execution Adapter

文件：

```text
multi_agent_sentiment_strategy.py
```

核心类：

```python
class MultiAgentSentimentStrategy(CtaTemplate):
    ...
```

Adapter 不重新计算 Alpha。

它只负责：

```text
final_execution_instruction.json
        ↓
Signal Validation
        ↓
Weight → Shares
        ↓
A-share Lot Size
        ↓
T+1 Validation
        ↓
BUY / SELL Intent
        ↓
VeighNa Order Management
```

默认安全模式：

```python
dry_run = True
```

首次导入 VeighNa 时，仅记录模拟委托，不真实发单。

---

## 16. CTA 在本项目中的含义

CTA 原意为：

> Commodity Trading Advisor

量化语境中的传统 CTA 策略通常包括：

- Trend Following
- Momentum
- Breakout
- Volatility Scaling
- Time-Series Strategy

但本项目使用 VeighNa `CtaStrategy` 的主要目的不是构建传统商品 CTA Alpha，而是：

> **利用 VeighNa 事件驱动的行情、订单、成交和策略生命周期框架，作为 V7 Final Research System 的执行适配层。**

因此：

```text
Research Alpha ≠ CTA Alpha

VeighNa CTA
=
Execution Framework
```

---

## 17. 项目目录结构

```text
sentiment_cycle_alpha/
│
├── README.md
├── requirements.txt
├── VERSION
├── .gitignore
│
├── config/
│   ├── settings.yaml
│   ├── openai_config.example.py
│   └── tushare_token.example.py
│
├── data/
│   ├── v7_8/
│   ├── v7_9/
│   │   └── intraday_5m/
│   ├── v7_event_llm/
│   └── v7_realtime/
│
├── src/
│   ├── data/
│   │   ├── v78_sector_proxy_sina.py
│   │   ├── v79_intraday_downloader.py
│   │   ├── realtime_market_snapshot_v7.py
│   │   └── focus_news_fetcher_v7.py
│   │
│   └── v7/
│       ├── sector_linkage_v78.py
│       ├── intraday_features_v79.py
│       ├── intraday_confirmation_v79.py
│       ├── intraday_backtest_v79.py
│       ├── research_pipeline_v79.py
│       ├── realtime_sentiment_v7.py
│       └── llm_event_structurer_v7.py
│
├── outputs/
│   ├── v7_7/
│   ├── v7_8/
│   ├── v7_9/
│   ├── v7_realtime/
│   ├── v7_event_llm/
│   └── v7_final/
│
├── tests/
│
├── run_v7_8.py
├── run_v7_8_sector_data.py
├── run_v7_9.py
├── run_v7_9_data.py
├── run_realtime_sentiment.py
├── run_step_c_event_scoring.py
├── build_final_execution_instruction.py
├── multi_agent_sentiment_strategy.py
├── test_veighna_adapter.py
└── README.md
```

---

## 18. 推荐完整运行顺序

如果核心历史数据已经准备完成，建议按以下顺序运行。

### Step 1 — 测试项目完整性

```powershell
python -m pytest -q
```

### Step 2 — 运行 V7.8 日频研究

```powershell
python run_v7_8.py
```

期望：

```text
market_gate=True
sector_gate=True
selected_variant=v7_8_market_return_only
champion=v7_7_frozen
challenger=v7_8_market_return_only
```

### Step 3 — 准备 / 运行 V7.9 真实 5min

若数据已存在：

```powershell
python run_v7_9.py
```

如需重新下载：

```powershell
python run_v7_9_data.py `
  --source baostock `
  --symbol 688981.SH `
  --start 2020-01-01 `
  --end 2025-12-31
```

然后：

```powershell
python run_v7_9.py
```

### Step 4 — 刷新实时情绪周期

```powershell
python run_realtime_sentiment.py
```

### Step 5 — 更新 LLM 新闻事件评分

```powershell
python run_step_c_event_scoring.py `
  --source eastmoney `
  --symbol 688981.SH `
  --max-events 100
```

> `--max-events` 表示最大处理数量。若数据接口只返回 10 条候选新闻，则只处理 10 条。

### Step 6 — 生成最终执行指令

```powershell
python build_final_execution_instruction.py --root "."
```

核心 Gate：

```text
market_gate=True
sector_gate=True
event_gate=True
intraday_gate=True
```

输出：

```text
outputs/v7_final/final_execution_instruction.json
```

### Step 7 — VeighNa CTA Dry Run

将：

```text
multi_agent_sentiment_strategy.py
```

复制到 VeighNa Trader 运行目录的：

```text
strategies/
```

策略实例：

```text
Class:
MultiAgentSentimentStrategy

Strategy Name:
SMIC_V7_Final

vt_symbol:
688981.SSE
```

建议参数：

```text
strategy_capital = 1000000
lot_size = 100
max_target_weight = 0.50
price_offset = 0.002
signal_stale_seconds = 900
dry_run = True
```

`signal_path` 设置为：

```text
<PROJECT_ROOT>\outputs\v7_final\final_execution_instruction.json
```

首次仅进行 Dry Run。

---

## 19. 研究纪律与防止数据泄漏

### 19.1 Development / Robustness

```text
2020–2024 = Development / Selection
2025      = Robustness Only
```

由于 2025 已在历史研究中被多次观察，因此本项目不再将其描述为完全 blind OOS。

### 19.2 Point-in-Time

项目尽可能保证：

- 日频信号映射到下一交易日
- 5min 信号只使用当前及历史 bar
- 当前 bar close 计算信号
- 下一 bar open 执行
- 不使用 future volume
- 不用当前板块成分股回填历史
- 模拟 5min 数据只用于 smoke test

### 19.3 LLM Governance

LLM 只负责：

```text
Content Understanding
+
Structured Semantic Extraction
```

不负责：

```text
Portfolio Optimization
Order Submission
BUY / SELL
Leverage Decision
```

---

## 20. 主要研究结论

1. 复杂情绪融合并不一定优于简单、稳定的 Market Return Overlay，因此 V7.8 Final 选择 `v7_8_market_return_only`。
2. Sector-Stock Linkage 能提供额外解释与实时上下文信息，但 Development 中未超过 Frozen Market-Return-Only Variant。
3. 5min Intraday Confirmation 能减少交易次数、改善 Sharpe 和降低 Max Drawdown，但不能凭空创造 Daily Alpha。
4. LLM 更适合作为 `Unstructured Information → Structured Features` 的桥梁，而不是直接交易决策器。
5. 本项目最终更适合被理解为一个 **多源信息驱动的量化研究与交易执行框架**。

---

## 21. 当前项目状态

```text
V7.6  ✅
V7.7  ✅ Frozen Daily Champion
V7.8  ✅ Real Market Challenger

Step A ✅ Sector-Stock Linkage
Step B ✅ Realtime Sentiment / Dynamic Position
Step C ✅ Prompt Engineering / Structured Events

V7.9  ✅ Real 5min Intraday Execution Research

Final Execution Instruction ✅

VeighNa CTA Adapter:
    Code Ready ✅
    Builder Ready ✅
    Dry-Run Default ✅
    VeighNa GUI Import / Final Runtime Verification ⏳
```

---

## 22. Final Research Pipeline

```text
Information Acquisition
        ↓
Content Parsing
        ↓
Prompt Engineering
        ↓
Structured Event Score
        ↓
A-Share Market Sentiment
        ↓
Sector Sentiment
        ↓
Sector-Stock Linkage
        ↓
Frozen Daily Alpha
        ↓
Dynamic Position Management
        ↓
5min Intraday Confirmation
        ↓
T+1 / Next-Bar / Cost / Risk
        ↓
Final Execution Instruction
        ↓
VeighNa CTA Adapter
```

---

## 23. 项目局限性

当前项目仍存在以下现实限制：

- V7.9 重点案例仍为单只股票中芯国际
- 实时数据接口存在第三方服务稳定性风险
- Sector 层使用历史代理数据而非完整历史成分股 PIT 重建
- LLM 事件分析受新闻数量、文本质量和 API 可用性影响
- VeighNa 当前以 Dry Run 验证执行适配为主
- 尚未完成真实券商 Gateway 下的账户级联调

---

## 24. Disclaimer

本项目仅用于：

- 量化研究
- 策略开发
- 学术 / 实习项目展示
- 工程验证

不构成任何投资建议。

真实交易环境还需要额外考虑：

- 券商 Gateway
- 真实可用资金
- 账户持仓同步
- 涨跌停
- 停牌
- 集合竞价
- 委托失败
- 部分成交
- 盘口流动性
- 实际手续费
- 滑点
- 实盘延迟
- 数据源中断
- 异常恢复

因此 VeighNa Adapter 默认：

```python
dry_run = True
```

在未完成真实账户级风险验证前，不应直接用于真实资金自动交易。

---

## 25. Project Status

**V7 Final Research System：Research Complete**

**剩余最终执行验证：VeighNa CTA Dry Run**
