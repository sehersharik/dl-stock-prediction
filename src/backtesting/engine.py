import pandas as pd
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.reports_dir = self.project_root / 'reports'
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        bt_cfg = self.config.get('backtesting', {})
        self.initial_capital = bt_cfg.get('initial_capital', 100000)
        self.transaction_cost = bt_cfg.get('transaction_cost', 0.001) # 0.1% cost
        self.slippage = bt_cfg.get('slippage', 0.0005) # 0.05% slippage
        
    def run(self, ticker: str, model_name: str, target_col: str = 'Target_Direction', threshold: float = 0.5):
        logger.info(f"Initializing Backtest for {ticker} using {model_name}...")
        logger.warning("DISCLAIMER: This is an experimental financial ML research system.")
        logger.warning("Historical backtest performance does NOT guarantee future returns. Do NOT trade this.")
        
        # Load out-of-sample predictions
        pred_path = self.project_root / 'data' / 'predictions' / f"{ticker}_{model_name}_{target_col}_test_preds.csv"
        if not pred_path.exists():
            logger.error(f"Prediction file not found: {pred_path}")
            return
        
        preds = pd.read_csv(pred_path, index_col=0, parse_dates=True)
        
        # Load original raw/feature data to get precise price movements
        features_path = self.project_root / self.config['data'].get('processed_dir', 'data/processed') / f"{ticker}_features.csv"
        features = pd.read_csv(features_path, index_col=0, parse_dates=True)
        
        # Align indexes perfectly
        df = preds.join(features[['Close']], how='inner')
        
        # Next Day Return = (Close[t+1] / Close[t]) - 1
        # Since we are predicting at Close of T for T+1, we shift Close back by -1 to calculate the return
        df['Next_Close'] = df['Close'].shift(-1)
        df['Next_Day_Return'] = (df['Next_Close'] / df['Close']) - 1
        
        # Drop the last row as we don't know its next day return yet
        df = df.dropna()
        
        # Generate Signals
        # 1 = Long, 0 = Cash
        df['Signal'] = (df['Prob'] >= threshold).astype(int)
        
        # Calculate Returns
        # Strategy return = Signal * Next_Day_Return - TransactionCosts
        df['Position_Change'] = df['Signal'].diff().fillna(df['Signal'].iloc[0]).abs()
        total_friction = self.transaction_cost + self.slippage
        
        df['Friction'] = df['Position_Change'] * total_friction
        df['Strategy_Return'] = (df['Signal'] * df['Next_Day_Return']) - df['Friction']
        df['BnH_Return'] = df['Next_Day_Return'] # Buy and Hold Benchmark
        
        # Calculate Equity Curves
        df['Strategy_Equity'] = self.initial_capital * (1 + df['Strategy_Return']).cumprod()
        df['BnH_Equity'] = self.initial_capital * (1 + df['BnH_Return']).cumprod()
        
        # Compute Drawdowns
        df['Strategy_Peak'] = df['Strategy_Equity'].cummax()
        df['Strategy_Drawdown'] = (df['Strategy_Equity'] - df['Strategy_Peak']) / df['Strategy_Peak']
        
        df['BnH_Peak'] = df['BnH_Equity'].cummax()
        df['BnH_Drawdown'] = (df['BnH_Equity'] - df['BnH_Peak']) / df['BnH_Peak']
        
        metrics = self._calculate_metrics(df)
        self._save_results(df, metrics, ticker, model_name)
        self._plot_results(df, ticker, model_name)
        
        logger.info(f"=== BACKTEST RESULTS: {model_name.upper()} ===")
        logger.info(f"Total Return: {metrics['Strategy_Total_Return']:.2%}")
        logger.info(f"BnH Return:   {metrics['BnH_Total_Return']:.2%}")
        logger.info(f"Sharpe Ratio: {metrics['Strategy_Sharpe']:.2f}")
        logger.info(f"Max Drawdown: {metrics['Strategy_Max_Drawdown']:.2%}")
        logger.info(f"Win Rate:     {metrics['Win_Rate']:.2%}")
        logger.info(f"Total Trades: {metrics['Total_Trades']}")
        
        if metrics['Strategy_Total_Return'] < 0:
            logger.info("The backtest demonstrates NO profitability.")
            
        return metrics

    def _calculate_metrics(self, df: pd.DataFrame):
        days = len(df)
        years = days / 252
        
        strat_returns = df['Strategy_Return']
        bnh_returns = df['BnH_Return']
        
        strat_total = (1 + strat_returns).prod() - 1
        bnh_total = (1 + bnh_returns).prod() - 1
        
        strat_ann = (1 + strat_total) ** (1 / years) - 1 if years > 0 else 0
        bnh_ann = (1 + bnh_total) ** (1 / years) - 1 if years > 0 else 0
        
        strat_vol = strat_returns.std() * np.sqrt(252)
        bnh_vol = bnh_returns.std() * np.sqrt(252)
        
        strat_sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252) if strat_returns.std() != 0 else 0
        bnh_sharpe = bnh_returns.mean() / bnh_returns.std() * np.sqrt(252) if bnh_returns.std() != 0 else 0
        
        max_dd = df['Strategy_Drawdown'].min()
        bnh_max_dd = df['BnH_Drawdown'].min()
        
        trades = int(df['Position_Change'].sum())
        
        # Win rate (excluding flat days where return is purely friction)
        active_days = df[df['Signal'] == 1]
        win_rate = (active_days['Strategy_Return'] > 0).mean() if len(active_days) > 0 else 0
        
        return {
            "Days_Tested": days,
            "Strategy_Total_Return": float(strat_total),
            "Strategy_Annualized": float(strat_ann),
            "Strategy_Volatility": float(strat_vol),
            "Strategy_Sharpe": float(strat_sharpe),
            "Strategy_Max_Drawdown": float(max_dd),
            "BnH_Total_Return": float(bnh_total),
            "BnH_Annualized": float(bnh_ann),
            "BnH_Volatility": float(bnh_vol),
            "BnH_Sharpe": float(bnh_sharpe),
            "BnH_Max_Drawdown": float(bnh_max_dd),
            "Win_Rate": float(win_rate),
            "Total_Trades": trades
        }
        
    def _save_results(self, df: pd.DataFrame, metrics: dict, ticker: str, model_name: str):
        with open(self.reports_dir / f"{ticker}_{model_name}_backtest_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        df.to_csv(self.reports_dir / f"{ticker}_{model_name}_backtest_curve.csv")
        
    def _plot_results(self, df: pd.DataFrame, ticker: str, model_name: str):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity Curve
        ax1.plot(df.index, df['Strategy_Equity'], label=f"Strategy ({model_name})", color='blue', linewidth=2)
        ax1.plot(df.index, df['BnH_Equity'], label="Buy & Hold Benchmark", color='gray', linestyle='--')
        ax1.set_title(f"Backtest Equity Curve: {ticker} (Out-of-Sample)\nNOT A RECOMMENDATION - RESEARCH ONLY")
        ax1.set_ylabel("Equity")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown Curve
        ax2.fill_between(df.index, df['Strategy_Drawdown'], 0, color='red', alpha=0.3, label="Strategy Drawdown")
        ax2.fill_between(df.index, df['BnH_Drawdown'], 0, color='gray', alpha=0.1, label="BnH Drawdown")
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / f"{ticker}_{model_name}_backtest.png")
        plt.close()
