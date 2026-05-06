import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, time

from core.config import MT5_PATH, SYMBOL_A, SYMBOL_B, DI_SYMBOL, TIMEFRAME
from core.kalman_filter import KalmanBetaFilter
from core.signals import calc_beta_ols, calc_zscore

# ─── Configuration ──────────────────────────────────────────────────────────
WDO_KALMAN_W = 40
WDO_KALMAN_Q = 1e-4
WDO_KALMAN_R = 1e2
DI_KALMAN_W = 40
DI_BETA_REF_BARS = 2240

Z_ENTRY = 2.0
Z_ATTENTION = 1.4

BARS_TO_FETCH = 20000  # Approximately 4 months of 5M bars

SCENARIOS = [
    {"name": "Scalp", "sl": 250, "tp": 300, "max_bars": None, "end_of_day": False},
    {"name": "Tempo de Tela", "sl": 510, "tp": 600, "max_bars": 5, "end_of_day": False},
    {"name": "Carregamento Intraday", "sl": 560, "tp": 99999, "max_bars": None, "end_of_day": True},
]

def fetch_data():
    print(f"Connecting to MT5 at {MT5_PATH}...")
    if not mt5.initialize(path=MT5_PATH):
        print(f"MT5 Init Failed: {mt5.last_error()}")
        return None
    
    for sym in [SYMBOL_A, SYMBOL_B, DI_SYMBOL]:
        mt5.symbol_select(sym, True)
        
    print(f"Fetching {BARS_TO_FETCH} bars for {SYMBOL_A}, {SYMBOL_B}, and {DI_SYMBOL}...")
    win_rates = mt5.copy_rates_from_pos(SYMBOL_A, TIMEFRAME, 0, BARS_TO_FETCH)
    wdo_rates = mt5.copy_rates_from_pos(SYMBOL_B, TIMEFRAME, 0, BARS_TO_FETCH)
    di_rates = mt5.copy_rates_from_pos(DI_SYMBOL, TIMEFRAME, 0, BARS_TO_FETCH)
    
    if win_rates is None or wdo_rates is None or di_rates is None:
        print("Error fetching data. Check symbols and MT5 connection.")
        return None
        
    df_win = pd.DataFrame(win_rates)
    df_win['time'] = pd.to_datetime(df_win['time'], unit='s')
    df_win.set_index('time', inplace=True)
    
    df_wdo = pd.DataFrame(wdo_rates)
    df_wdo['time'] = pd.to_datetime(df_wdo['time'], unit='s')
    df_wdo.set_index('time', inplace=True)
    
    df_di = pd.DataFrame(di_rates)
    df_di['time'] = pd.to_datetime(df_di['time'], unit='s')
    df_di.set_index('time', inplace=True)
    
    # Align data
    df = pd.DataFrame({
        'win': df_win['close'],
        'win_high': df_win['high'],
        'win_low': df_win['low'],
        'wdo': df_wdo['close'],
        'di': df_di['close']
    }).dropna()
    
    print(f"Data aligned: {len(df)} bars.")
    return df

def generate_signals(df):
    print("Generating signals (Kalman for WDO, OLS for DI)...")
    win = df['win'].values
    wdo = df['wdo'].values
    di = df['di'].values
    
    # WDO Kalman Z-Score
    kf = KalmanBetaFilter(initial_beta=-22.5, trans_cov=WDO_KALMAN_Q, obs_cov=WDO_KALMAN_R)
    spreads_wdo = []
    for y, x in zip(win, wdo):
        beta, spread, _ = kf.update(float(y), float(x))
        spreads_wdo.append(spread)
    
    z_wdo = KalmanBetaFilter.rolling_zscore(spreads_wdo, window=WDO_KALMAN_W)
    df['z_wdo'] = z_wdo
    
    # DI OLS Z-Score
    z_di = np.zeros(len(df))
    # We must compute a rolling OLS beta over 2240 bars, but for speed in backtest
    # we can use a simpler rolling window approach or just calc it efficiently
    # To keep it identical to server.py:
    # beta_ref_20d = calc_beta_ols(closes_win[-2240:], closes_di[-2240:])
    # calc_zscore(...)
    
    print("Calculating rolling DI Z-Scores...")
    # Pre-calculate rolling DI beta and zscore
    # This might be slow if done naive loop, so we vectorize
    # For backtest simulation, we will use a rolling window of DI_BETA_REF_BARS for beta
    
    # Simple vectorization for DI beta over rolling window (avoid exact 2240 loop if too slow, but let's try it)
    beta_di_arr = np.zeros(len(df))
    for i in range(len(df)):
        if i < DI_KALMAN_W:
            beta_di_arr[i] = -100 # arbitrary
            continue
            
        start_idx = max(0, i - DI_BETA_REF_BARS)
        w_win = win[start_idx:i+1]
        w_di = di[start_idx:i+1]
        
        # Simple OLS: Cov(X, Y) / Var(X)
        if len(w_win) > 2:
            cov = np.cov(w_di, w_win)[0][1]
            var = np.var(w_di)
            beta = cov / var if var > 0 else -100
        else:
            beta = -100
        beta_di_arr[i] = beta
        
    df['beta_di'] = beta_di_arr
    df['spread_di'] = win - (beta_di_arr * di)
    df['z_di'] = (df['spread_di'] - df['spread_di'].rolling(DI_KALMAN_W).mean()) / df['spread_di'].rolling(DI_KALMAN_W).std()
    
    # Consensus Signal
    # BUY: (z_wdo <= -Z_ENTRY and z_di <= -Z_ATTENTION) or (z_wdo <= -Z_ATTENTION and z_di <= -Z_ENTRY)
    # SELL: (z_wdo >= Z_ENTRY and z_di >= Z_ATTENTION) or (z_wdo >= Z_ATTENTION and z_di >= Z_ENTRY)
    df['sig_wdo'] = np.where(df['z_wdo'] <= -Z_ATTENTION, 1, np.where(df['z_wdo'] >= Z_ATTENTION, -1, 0))
    df['sig_di'] = np.where(df['z_di'] <= -Z_ATTENTION, 1, np.where(df['z_di'] >= Z_ATTENTION, -1, 0))
    
    df['sig_wdo_strong'] = np.where(df['z_wdo'] <= -Z_ENTRY, 1, np.where(df['z_wdo'] >= Z_ENTRY, -1, 0))
    df['sig_di_strong'] = np.where(df['z_di'] <= -Z_ENTRY, 1, np.where(df['z_di'] >= Z_ENTRY, -1, 0))
    
    df['signal'] = 0
    # Buy
    buy_mask = ((df['sig_wdo_strong'] == 1) & (df['sig_di'] == 1)) | ((df['sig_wdo'] == 1) & (df['sig_di_strong'] == 1))
    df.loc[buy_mask, 'signal'] = 1
    # Sell
    sell_mask = ((df['sig_wdo_strong'] == -1) & (df['sig_di'] == -1)) | ((df['sig_wdo'] == -1) & (df['sig_di_strong'] == -1))
    df.loc[sell_mask, 'signal'] = -1
    
    return df

def run_backtest(df, scenario):
    sl_pts = scenario['sl']
    tp_pts = scenario['tp']
    max_bars = scenario['max_bars']
    end_of_day = scenario['end_of_day']
    
    trades = []
    open_trade = None
    
    # Iterate row by row
    for i in range(len(df)):
        row = df.iloc[i]
        dt = df.index[i]
        
        # Check Exits
        if open_trade is not None:
            open_trade['bars_held'] += 1
            
            # Intraday High/Low approximation (using close price here for simplicity, 
            # ideally use High/Low of WIN for accurate SL/TP matching during the bar)
            # We'll use the bar's High/Low to see if SL/TP was hit within the bar.
            
            entry_price = open_trade['entry_price']
            direction = open_trade['direction']
            
            hit_sl = False
            hit_tp = False
            exit_price = None
            exit_reason = ""
            
            if direction == 1: # BUY
                if row['win_low'] <= entry_price - sl_pts:
                    hit_sl, exit_price, exit_reason = True, entry_price - sl_pts, "SL"
                elif row['win_high'] >= entry_price + tp_pts:
                    hit_tp, exit_price, exit_reason = True, entry_price + tp_pts, "TP"
            else: # SELL
                if row['win_high'] >= entry_price + sl_pts:
                    hit_sl, exit_price, exit_reason = True, entry_price + sl_pts, "SL"
                elif row['win_low'] <= entry_price - tp_pts:
                    hit_tp, exit_price, exit_reason = True, entry_price - tp_pts, "TP"
                    
            # Time-based exits
            hit_time = False
            if max_bars and open_trade['bars_held'] >= max_bars:
                hit_time, exit_price, exit_reason = True, row['win'], "TIME_MAX_BARS"
                
            if end_of_day and dt.time() >= time(17, 30):
                hit_time, exit_price, exit_reason = True, row['win'], "TIME_EOD"
                
            if hit_sl or hit_tp or hit_time:
                open_trade['exit_price'] = exit_price
                open_trade['exit_time'] = dt
                open_trade['exit_reason'] = exit_reason
                
                if direction == 1:
                    open_trade['pnl'] = exit_price - entry_price
                else:
                    open_trade['pnl'] = entry_price - exit_price
                    
                trades.append(open_trade)
                open_trade = None
                continue # Trade closed, look for new entries on next bar
                
        # Check Entries
        if open_trade is None and row['signal'] != 0:
            # Avoid opening entries at the very end of the day
            if end_of_day and dt.time() >= time(17, 0):
                continue
                
            open_trade = {
                'entry_time': dt,
                'direction': row['signal'],
                'entry_price': row['win'],
                'bars_held': 0
            }
            
    return trades

def print_report(trades, name):
    if not trades:
        print(f"--- {name} ---")
        print("No trades executed.\n")
        return
        
    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] < 0]
    
    win_rate = len(wins) / len(df_trades) * 100
    gross_profit = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    net_profit = gross_profit - gross_loss
    
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
    payoff = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Calculate Drawdown
    df_trades['cum_pnl'] = df_trades['pnl'].cumsum()
    df_trades['peak'] = df_trades['cum_pnl'].cummax()
    df_trades['drawdown'] = df_trades['peak'] - df_trades['cum_pnl']
    max_dd = df_trades['drawdown'].max()
    
    ret_dd = net_profit / max_dd if max_dd > 0 else float('inf')
    
    print(f"=== Report: {name} ===")
    print(f"Total Trades:   {len(df_trades)}")
    print(f"Win Rate:       {win_rate:.2f}%")
    print(f"Net PnL:        {net_profit:.2f} pts")
    print(f"Max Drawdown:   {max_dd:.2f} pts")
    print(f"Ret/DD:         {ret_dd:.2f}")
    print(f"Pay-off:        {payoff:.2f}")
    print(f"Profit Factor:  {profit_factor:.2f}")
    print("-" * 25)

if __name__ == "__main__":
    df = fetch_data()
    if df is not None:
        df = generate_signals(df)
        print("\nStarting Backtest Simulations...\n")
        
        for sc in SCENARIOS:
            trades = run_backtest(df, sc)
            print_report(trades, sc['name'])
