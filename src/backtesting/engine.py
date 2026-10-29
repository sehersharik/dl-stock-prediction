import pandas as pd
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
from pathlib import Path

from src.features.generator import align_trading_calendar

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# Signal at the close of session T (features use data up to and including T).
# Each execution variant defines the return earned by the position decided at T:
EXECUTIONS = {
    # optimistic: trade at the same close the features were computed from
    "close_to_close": "enter/exit at close T, hold to close T+1",
    # executable: the order is placed after the close and filled at the next open
    "next_open": "enter/exit at open T+1, hold to open T+2",
}


class Backtester:
    """Long/cash backtest of out-of-sample direction predictions.

    Conventions (all explicit, see docs/learning-notes.md Phase 4):
    - Returns are computed on the trading-session calendar (align_trading_calendar)
      *before* aligning to prediction dates, so a period is always one session.
    - Cost = transaction_cost + slippage, charged per side: on every entry and every
      exit, including closing an open position at the end of the test period.
    - Buy & hold is always long over exactly the same periods and pays one entry
      and one exit at the same per-side cost.
    - Cash earns nothing. Sharpe uses excess daily returns over a constant annual
      risk-free rate (config backtesting.risk_free_rate_annual).
    """
    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.reports_dir = self.project_root / 'reports'
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        bt_cfg = self.config.get('backtesting', {})
        self.initial_capital = bt_cfg.get('initial_capital', 100000)
        self.transaction_cost = bt_cfg.get('transaction_cost', 0.001) # 0.1% cost
        self.slippage = bt_cfg.get('slippage', 0.0005) # 0.05% slippage
        self.risk_free_rate = bt_cfg.get('risk_free_rate_annual', 0.0)
        self.last_curve = None

    def _session_prices(self, ticker: str) -> pd.DataFrame:
        raw_dir = self.project_root / self.config['data'].get('raw_dir', 'data/raw')
        prices = pd.read_csv(raw_dir / f"{ticker}_10y.csv", index_col=0, parse_dates=True).sort_index()
        nifty = pd.read_csv(raw_dir / "NSEI_10y.csv", index_col=0, parse_dates=True)
        return align_trading_calendar(prices, {"NIFTY": nifty})

    @staticmethod
    def period_returns(prices: pd.DataFrame, execution: str) -> pd.Series:
        """Return earned by a position decided at the close of each session."""
        if execution == "close_to_close":
            return prices['Close'].shift(-1) / prices['Close'] - 1
        if execution == "next_open":
            return prices['Open'].shift(-2) / prices['Open'].shift(-1) - 1
        raise ValueError(f"Unknown execution {execution!r}; expected one of {list(EXECUTIONS)}")

    def run(self, ticker: str, model_name: str, target_col: str = 'Target_Direction', threshold: float = 0.5,
            pred_path: Path = None, save: bool = True, execution: str = "close_to_close", exit_threshold: float = None,
            preds: pd.DataFrame = None):
        logger.info(f"Initializing Backtest for {ticker} using {model_name} ({execution})...")
        logger.warning("DISCLAIMER: This is an experimental financial ML research system.")
        logger.warning("Historical backtest performance does NOT guarantee future returns. Do NOT trade this.")

        # Load out-of-sample predictions
        if preds is None:
            if pred_path is None:
                pred_path = self.project_root / 'data' / 'predictions' / f"{ticker}_{model_name}_{target_col}_test_preds.csv"
            if not pred_path.exists():
                logger.error(f"Prediction file not found: {pred_path}")
                return
            preds = pd.read_csv(pred_path, index_col=0, parse_dates=True)

        df = self.simulate(preds, self._session_prices(ticker), threshold, execution, exit_threshold)
        self.last_curve = df
        metrics = self._calculate_metrics(df, execution)
        metrics["Entry_Threshold"] = float(threshold)
        metrics["Exit_Threshold"] = float(threshold if exit_threshold is None else exit_threshold)
        if save:
            self._save_results(df, metrics, ticker, f"{model_name}_{execution}")
            self._plot_results(df, ticker, f"{model_name} ({execution})")

        logger.info(f"=== BACKTEST RESULTS: {model_name.upper()} [{execution}] ===")
        logger.info(f"Total Return: {metrics['Strategy_Total_Return']:.2%} | BnH: {metrics['BnH_Total_Return']:.2%}")
        logger.info(f"Sharpe (rf {self.risk_free_rate:.2%}): {metrics['Strategy_Sharpe']:.2f} | Max DD: {metrics['Strategy_Max_Drawdown']:.2%}")
        logger.info(f"Trades: {metrics['Trades']} | Win rate per trade: {metrics['Win_Rate_Trade']:.2%} | per day: {metrics['Win_Rate_Day']:.2%}")
        return metrics

    @staticmethod
    def signals(prob: pd.Series, enter: float, exit_: float) -> pd.Series:
        """Long when P(up) >= enter, cash when P(up) < exit_, otherwise keep the current
        position (hysteresis). enter == exit_ gives the plain threshold rule."""
        pos, out = 0, []
        for p in prob.values:
            if p >= enter:
                pos = 1
            elif p < exit_:
                pos = 0
            out.append(pos)
        return pd.Series(out, index=prob.index, dtype=int)

    def simulate(self, preds: pd.DataFrame, prices: pd.DataFrame, threshold: float, execution: str,
                 exit_threshold: float = None) -> pd.DataFrame:
        cost = self.transaction_cost + self.slippage
        df = preds[['Prob']].join(self.period_returns(prices, execution).rename('Period_Return'), how='left')
        missing = df['Period_Return'].isna()
        if missing.any():
            logger.info(f"Dropping {int(missing.sum())} prediction rows without a complete next period: "
                        f"{[str(d.date()) for d in df.index[missing]]}")
        df = df[~missing].copy()

        # 1 = Long, 0 = Cash
        df['Signal'] = self.signals(df['Prob'], threshold, threshold if exit_threshold is None else exit_threshold)
        prev = df['Signal'].shift(1, fill_value=0)
        df['Position_Change'] = (df['Signal'] - prev).abs()
        # Close any open position at the end of the test period
        df['Final_Exit'] = 0
        df.iloc[-1, df.columns.get_loc('Final_Exit')] = int(df['Signal'].iloc[-1])
        df['Friction'] = (df['Position_Change'] + df['Final_Exit']) * cost
        df['Strategy_Return'] = df['Signal'] * df['Period_Return'] - df['Friction']

        bnh_friction = pd.Series(0.0, index=df.index)
        bnh_friction.iloc[0] += cost
        bnh_friction.iloc[-1] += cost
        df['BnH_Return'] = df['Period_Return'] - bnh_friction

        for side in ("Strategy", "BnH"):
            df[f'{side}_Equity'] = self.initial_capital * (1 + df[f'{side}_Return']).cumprod()
            df[f'{side}_Peak'] = df[f'{side}_Equity'].cummax()
            df[f'{side}_Drawdown'] = df[f'{side}_Equity'] / df[f'{side}_Peak'] - 1
        return df

    @staticmethod
    def trade_returns(df: pd.DataFrame) -> list:
        """Net return of each round trip: entry cost, holding days and exit cost.

        A trade is a run of consecutive long periods. Its exit cost sits on the first
        cash period after it (or on its last period if closed at the end)."""
        trades, start = [], None
        sig, net = df['Signal'].values, df['Strategy_Return'].values
        for i, s in enumerate(sig):
            if s == 1 and start is None:
                start = i
            if start is not None and (s == 0 or i == len(sig) - 1):
                # include period i: the first cash period (exit cost) or the final period
                trades.append(float(np.prod(1 + net[start:i + 1]) - 1))
                start = None
        return trades

    def _calculate_metrics(self, df: pd.DataFrame, execution: str = "close_to_close"):
        days = len(df)
        years = days / TRADING_DAYS
        rf_daily = (1 + self.risk_free_rate) ** (1 / TRADING_DAYS) - 1

        def side(r: pd.Series, dd: pd.Series):
            total = (1 + r).prod() - 1
            excess = r - rf_daily
            return {
                "Total_Return": float(total),
                "Annualized": float((1 + total) ** (1 / years) - 1) if years > 0 else 0.0,
                "Volatility": float(r.std() * np.sqrt(TRADING_DAYS)),
                "Sharpe": float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) if excess.std() else 0.0,
                "Sharpe_rf0": float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() else 0.0,
                "Max_Drawdown": float(dd.min()),
            }
        strat = side(df['Strategy_Return'], df['Strategy_Drawdown'])
        bnh = side(df['BnH_Return'], df['BnH_Drawdown'])

        active_days = df[df['Signal'] == 1]
        trades = self.trade_returns(df)
        position_changes = int(df['Position_Change'].sum() + df['Final_Exit'].sum())

        return {
            "Execution": execution,
            "Execution_Rule": EXECUTIONS[execution],
            "First_Signal_Date": str(df.index[0].date()),
            "Last_Signal_Date": str(df.index[-1].date()),
            "Days_Tested": days,
            "Cost_Per_Side": float(self.transaction_cost + self.slippage),
            "Risk_Free_Rate_Annual": float(self.risk_free_rate),
            **{f"Strategy_{k}": v for k, v in strat.items()},
            **{f"BnH_{k}": v for k, v in bnh.items()},
            "Excess_Return_vs_BnH": float(strat["Total_Return"] - bnh["Total_Return"]),
            "In_Market_Days": int(len(active_days)),
            "Cash_Days": int(days - len(active_days)),
            "Exposure": float(len(active_days) / days),
            "Winning_Days": int((active_days['Strategy_Return'] > 0).sum()),
            # Win rate per in-market day (net of costs charged that day)
            "Win_Rate_Day": float((active_days['Strategy_Return'] > 0).mean()) if len(active_days) else 0.0,
            "Trades": len(trades),
            "Winning_Trades": int(sum(t > 0 for t in trades)),
            # Win rate per round-trip trade, net of entry and exit costs
            "Win_Rate_Trade": float(np.mean([t > 0 for t in trades])) if trades else 0.0,
            "Avg_Trade_Return": float(np.mean(trades)) if trades else 0.0,
            "Position_Changes": position_changes,
            # Fraction of capital traded per year (each position change trades 100% of equity)
            "Annual_Turnover": float(position_changes / years) if years > 0 else 0.0,
            "Total_Cost_Paid": float(df['Friction'].sum()),
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
