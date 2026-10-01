# -*- coding: utf-8 -*-
from __future__ import annotations
import json, math
from datetime import date, datetime
from pathlib import Path
from typing import Any
from vnpy.trader.constant import Direction
from vnpy.trader.object import BarData, OrderData, TickData, TradeData
from vnpy_ctastrategy import CtaTemplate, StopOrder

class MultiAgentSentimentStrategy(CtaTemplate):
    """V7 Final -> VeighNa CTA execution adapter. Long-only, T+1 aware."""
    author='V7 Final Multi-Agent Research'
    strategy_capital=1_000_000.0; lot_size=100; max_target_weight=.50; price_offset=.002; signal_stale_seconds=900; dry_run=True
    signal_path='final_execution_instruction.json'
    parameters=['strategy_capital','lot_size','max_target_weight','price_offset','signal_stale_seconds','dry_run','signal_path']
    target_weight=0.0; target_pos=0; sellable_pos=0; today_bought=0; cycle_multiplier=1.0; event_risk_multiplier=1.0; intraday_action=''; signal_quality=''; signal_time=''; last_signal_id=''; last_action=''; pending_order=False; trading_day=''
    variables=['target_weight','target_pos','sellable_pos','today_bought','cycle_multiplier','event_risk_multiplier','intraday_action','signal_quality','signal_time','last_signal_id','last_action','pending_order','trading_day']

    def __init__(self,cta_engine:Any,strategy_name:str,vt_symbol:str,setting:dict)->None:
        super().__init__(cta_engine,strategy_name,vt_symbol,setting); self._last_signal_mtime_ns=-1; self._latest_price=0.0
    def on_init(self): self.write_log('V7 Final CTA Adapter 初始化'); self._roll_trading_day(datetime.now().date()); self.put_event()
    def on_start(self): self.write_log(f'V7 Final CTA Adapter 启动 | dry_run={self.dry_run} | signal={self.signal_path}'); self.put_event()
    def on_stop(self): self.write_log('V7 Final CTA Adapter 停止'); self.put_event()
    def on_tick(self,tick:TickData):
        if tick.last_price and tick.last_price>0: self._latest_price=float(tick.last_price)
        self._roll_trading_day(tick.datetime.date()); self._process_instruction(self._latest_price)
    def on_bar(self,bar:BarData):
        if bar.close_price and bar.close_price>0: self._latest_price=float(bar.close_price)
        self._roll_trading_day(bar.datetime.date()); self._process_instruction(self._latest_price)
    def on_order(self,order:OrderData):
        try:
            if not order.is_active(): self.pending_order=False
        except Exception: pass
        self.put_event()
    def on_trade(self,trade:TradeData):
        volume=int(round(float(trade.volume)))
        if trade.direction==Direction.LONG: self.today_bought+=volume
        else: self.sellable_pos=max(0,self.sellable_pos-volume)
        self.pending_order=False; self.write_log(f'成交 direction={trade.direction.value} volume={volume} pos={self.pos} sellable={self.sellable_pos} today_bought={self.today_bought}'); self.put_event()
    def on_stop_order(self,stop_order:StopOrder): self.put_event()

    def _roll_trading_day(self,d:date):
        s=d.isoformat()
        if self.trading_day==s: return
        self.trading_day=s; self.sellable_pos=max(0,int(round(self.pos))); self.today_bought=0; self.pending_order=False
        self.write_log(f'新交易日 {s}: pos={self.pos}, sellable_pos={self.sellable_pos}'); self.put_event()
    def _resolve_signal_path(self):
        p=Path(self.signal_path).expanduser(); return p if p.is_absolute() else Path.cwd()/p
    def _load_instruction_if_changed(self):
        p=self._resolve_signal_path()
        if not p.exists(): return None
        try: st=p.stat()
        except OSError: return None
        if st.st_mtime_ns==self._last_signal_mtime_ns: return None
        try: data=json.loads(p.read_text(encoding='utf-8-sig'))
        except Exception as e: self.write_log(f'读取 instruction 失败: {e}'); self._last_signal_mtime_ns=st.st_mtime_ns; return None
        self._last_signal_mtime_ns=st.st_mtime_ns; return data if isinstance(data,dict) else None
    def _validate_instruction(self,data):
        if data.get('quality')!='V7_FINAL_EXECUTION_READY': return False,f"quality={data.get('quality')}"
        if str(data.get('vt_symbol','')).upper()!=self.vt_symbol.upper(): return False,'vt_symbol mismatch'
        for g in ('market_gate','sector_gate','event_gate','intraday_gate'):
            if not bool(data.get(g)): return False,f'{g}=False'
        action=str(data.get('intraday_action','')).upper()
        if action not in {'ENTER','WAIT','REDUCE','EXIT','HOLD'}: return False,f'unknown action={action}'
        try: w=float(data['final_target_weight'])
        except Exception: return False,'invalid final_target_weight'
        if not math.isfinite(w) or not 0<=w<=self.max_target_weight: return False,f'target_weight out of range: {w}'
        try:
            ts=datetime.fromisoformat(str(data['as_of']).replace('Z','+00:00')); now=datetime.now().astimezone(); ts=ts.astimezone() if ts.tzinfo is None else ts.astimezone(); age=abs((now-ts).total_seconds())
        except Exception: return False,'invalid as_of'
        if age>self.signal_stale_seconds: return False,f'signal stale: {age:.0f}s'
        return True,'OK'
    def _weight_to_target_pos(self,w,price):
        if price<=0 or self.strategy_capital<=0: return 0
        return max(0,int(math.floor((self.strategy_capital*w/price)/self.lot_size)*self.lot_size))
    def _limit_price(self,price,is_buy): return max(.01,price*(1+(self.price_offset if is_buy else -self.price_offset)))
    def _process_instruction(self,price):
        if price<=0: return
        data=self._load_instruction_if_changed()
        if data is None: return
        ok,reason=self._validate_instruction(data)
        if not ok: self.last_action=f'REJECTED: {reason}'; self.write_log(self.last_action); self.put_event(); return
        sid=str(data.get('signal_id',''))
        if sid and sid==self.last_signal_id: return
        self.target_weight=float(data['final_target_weight']); self.cycle_multiplier=float(data.get('cycle_multiplier',1)); self.event_risk_multiplier=float(data.get('event_risk_multiplier',1)); self.intraday_action=str(data.get('intraday_action','')).upper(); self.signal_quality=str(data.get('quality','')); self.signal_time=str(data.get('as_of','')); self.target_pos=self._weight_to_target_pos(self.target_weight,price)
        current=max(0,int(round(self.pos))); desired=self.target_pos
        if self.intraday_action in {'WAIT','HOLD','REDUCE'}: desired=min(desired,current)
        delta=desired-current
        self.write_log(f'V7 signal {sid} action={self.intraday_action} weight={self.target_weight:.2%} current={current} desired={desired} sellable={self.sellable_pos}')
        if delta==0: self.last_action='NO_ORDER_TARGET_MET'; self.last_signal_id=sid; self.put_event(); return
        if self.pending_order and not self.dry_run: self.last_action='WAIT_PENDING_ORDER'; self.write_log(self.last_action); self.put_event(); return
        if delta>0:
            vol=int(math.floor(delta/self.lot_size)*self.lot_size)
            if vol<=0: self.last_action='NO_ORDER_BELOW_LOT'
            else: self._submit_buy(self._limit_price(price,True),vol)
        else:
            vol=min(abs(delta),max(0,int(self.sellable_pos))); vol=int(math.floor(vol/self.lot_size)*self.lot_size)
            if vol<=0: self.last_action='T+1_BLOCKED_NO_SELLABLE_SHARES'; self.write_log(self.last_action)
            else: self._submit_sell(self._limit_price(price,False),vol)
        self.last_signal_id=sid; self.put_event()
    def _submit_buy(self,price,volume):
        if self.dry_run: self.last_action=f'DRY_RUN BUY {volume} @ {price:.3f}'; self.write_log(self.last_action); return
        ids=self.buy(price,volume); self.pending_order=bool(ids); self.last_action=f'BUY {volume} @ {price:.3f}'; self.write_log(self.last_action)
    def _submit_sell(self,price,volume):
        if self.dry_run: self.last_action=f'DRY_RUN SELL {volume} @ {price:.3f}'; self.write_log(self.last_action); return
        ids=self.sell(price,volume); self.pending_order=bool(ids); self.last_action=f'SELL {volume} @ {price:.3f}'; self.write_log(self.last_action)
