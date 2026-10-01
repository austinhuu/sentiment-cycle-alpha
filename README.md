# SentimentCycle Alpha

**Multi-Agent A-Share Sentiment Cycle & Research-to-Execution Framework**  
中文：多智能体 A 股情绪周期与研究—交易执行框架。

> **Publication review draft:** Derived from a private research archive. Do not make this repository public until code/data ownership and internship confidentiality permissions are confirmed. API credentials, licensed/raw market datasets, full output caches and historical LLM responses are intentionally not distributed.

## Overview / 项目简介

A modular research pipeline integrating market/fundamental/technical/news agents, structured Bull–Bear debate, A-share sentiment-cycle and sector-to-stock signals, dynamic risk budgeting, 5-minute intraday confirmation, and a guarded VeighNa execution adapter. The main case study is 688981.SH (SMIC).

**Scope is research and simulated execution infrastructure, not a verified live-trading system.** The adapter defaults to `dry_run=True`; external brokerage gateway integration still requires validation.

## Architecture / 系统架构

```text
Market / Fundamentals / News / Technical Signals
       |
Multi-agent views + evidence-based Bull/Bear debate
       |
A-share market sentiment cycle + sector-to-stock linkage
       |
Frozen daily alpha + dynamic position/risk multiplier
       |
LLM event structuring -> deterministic Python event score
       |
5-minute intraday confirmation (VWAP, RVOL, ORB, volatility)
       |
T+1 + next-bar execution + transaction costs + position cap
       |
final_execution_instruction.json -> VeighNa adapter (dry-run)
```

The LLM is used for **structured interpretation of unstructured information**. It is not authorized to place orders or directly choose final portfolio positions.

## Modules / 主要模块

| Path | Purpose |
|---|---|
| `src/agents/`, `src/debate/` | Technical, fundamental, sentiment, regime, news, risk agents; Bull–Bear debate and controlled decision aggregation |
| `src/v7/` | Market sentiment cycle, sector-stock linkage, V7.7/7.8 overlay and V7.9 intraday research |
| `src/models/`, `src/optimization/` | Adaptive fusion and risk-aware optimization |
| `src/backtest/`, `src/evaluation/`, `src/robustness/` | Portfolio backtest, ablation, walk-forward and robustness experiments |
| `src/execution/`, `multi_agent_sentiment_strategy.py` | Instruction generation, order planning and guarded execution adapter |
| `tests/` | Data, research, event-structuring and execution-oriented tests |

## Quick start: synthetic demo / 合成数据演示

Use Python 3.10+ in a fresh environment. The following path uses **synthetic data**, not the reported historical case-study dataset.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_demo_data.py
python run_pipeline.py
```

The generated CSV and outputs are intentionally ignored by Git. Some optional real-data modules depend on third-party APIs, valid credentials, provider availability and data licenses.

Run static/integration tests locally after installing dependencies:

```powershell
python -m pytest -q
```

## Credentials / 密钥配置

Do not hardcode tokens. The checked-in `config/openai_config.py` and `config/tushare_token.py` read environment variables. PowerShell example:

```powershell
$env:DEEPSEEK_API_KEY = "<your-own-key>"
$env:TUSHARE_TOKEN = "<your-own-token>"
```

`.env.example` documents expected names; it is **not automatically loaded**. `.env` is ignored.

## Research status / 研究状态

- V7.7 is the frozen daily Champion; V7.8 was evaluated as a market-overlay Challenger rather than automatically replacing the baseline.
- LLM event signals are bounded and are converted to deterministic risk multipliers, not direct orders.
- The V7.9 SMIC intraday study uses real historical 5-minute bars; synthetic data is used only for smoke tests.
- 2025 robustness data was observed during research and is **not claimed as fully blind untouched OOS**.
- Full real-data and numerical-result reproduction require separately authorized historical datasets; they are intentionally absent here.
- VeighNa adapter is code-ready / dry-run default; real brokerage gateway and account-level integration remain unverified.

See [Detailed research notes](docs/RESEARCH_NOTES.md) for methodology, historical experiments, limitations, and the full run sequence. Historical output metrics are documented, but the omitted underlying data prevents independent full replication from this source-only package.

## Responsible use / 合规说明

This is a research/educational showcase. It is not investment advice and is not production-ready automated trading. Confirm employer/internship ownership and confidentiality permissions before public release. Obtain market/news data from authorized sources and comply with their redistribution terms. No open-source license is granted until permission/ownership is confirmed and a LICENSE is deliberately selected.
