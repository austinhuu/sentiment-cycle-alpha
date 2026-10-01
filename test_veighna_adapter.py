# -*- coding: utf-8 -*-
from __future__ import annotations
import ast, importlib.util
from pathlib import Path
import pytest
from build_final_execution_instruction import build_instruction

ROOT=Path(__file__).resolve().parent
STRATEGY_FILE=ROOT/'multi_agent_sentiment_strategy.py'

def test_final_instruction_math():
    d=build_instruction(symbol='688981.SH',vt_symbol='688981.SSE',base_target_weight=.30,cycle_multiplier=.80,event_risk_multiplier=.90,intraday_action='ENTER',market_gate=True,sector_gate=True,event_gate=True,intraday_gate=True,max_target_weight=.50,as_of='2026-08-23T10:00:00+08:00')
    assert d['quality']=='V7_FINAL_EXECUTION_READY'; assert d['final_target_weight']==pytest.approx(.216); assert len(d['signal_id'])==20

def test_exit_forces_zero_weight():
    d=build_instruction(symbol='688981.SH',vt_symbol='688981.SSE',base_target_weight=.30,cycle_multiplier=.80,event_risk_multiplier=1.0,intraday_action='EXIT',market_gate=True,sector_gate=True,event_gate=True,intraday_gate=True,max_target_weight=.50,as_of='2026-08-23T10:00:00+08:00')
    assert d['final_target_weight']==0

def test_closed_gate_refuses_execution():
    with pytest.raises(RuntimeError,match='sector_gate'):
        build_instruction(symbol='688981.SH',vt_symbol='688981.SSE',base_target_weight=.30,cycle_multiplier=.80,event_risk_multiplier=1.0,intraday_action='ENTER',market_gate=True,sector_gate=False,event_gate=True,intraday_gate=True,max_target_weight=.50,as_of='2026-08-23T10:00:00+08:00')

def test_strategy_is_valid_python(): ast.parse(STRATEGY_FILE.read_text(encoding='utf-8'))

def test_strategy_defaults_to_dry_run():
    tree=ast.parse(STRATEGY_FILE.read_text(encoding='utf-8')); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MultiAgentSentimentStrategy')
    vals={t.id:n.value for n in cls.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
    assert isinstance(vals['dry_run'],ast.Constant) and vals['dry_run'].value is True

def test_optional_veighna_import_smoke():
    if importlib.util.find_spec('vnpy_ctastrategy') is None: pytest.skip('vnpy_ctastrategy not installed')
    spec=importlib.util.spec_from_file_location('multi_agent_sentiment_strategy',STRATEGY_FILE); mod=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(mod); assert hasattr(mod,'MultiAgentSentimentStrategy')
