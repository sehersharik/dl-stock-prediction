# Feature Dictionary

## Group 1: Raw
| Feature Name | Category | Formula / Definition | Lookback | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `Open`, `High`, `Low`, `Close` | Raw | Unaltered daily price levels | 0 | Baseline price structure |
| `Volume` | Raw | Unaltered daily trading volume | 0 | Baseline liquidity |
| `Returns` | Raw | `(Close_t - Close_t-1) / Close_t-1` | 1 | Daily percentage change |

## Group 2: Trend
| Feature Name | Category | Formula / Definition | Lookback | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `SMA_20` | Trend | Simple Moving Average (20-day) | 20 | Short-term trend direction |
| `SMA_50` | Trend | Simple Moving Average (50-day) | 50 | Medium-term trend direction |
| `SMA_200` | Trend | Simple Moving Average (200-day) | 200 | Long-term macroeconomic trend |
| `EMA_12` | Trend | Exponential Moving Average (12-day) | 12 | Fast, weight-decayed short trend |
| `EMA_26` | Trend | Exponential Moving Average (26-day) | 26 | Medium-fast weight-decayed trend |
| `EMA_50` | Trend | Exponential Moving Average (50-day) | 50 | Medium weight-decayed trend |

## Group 3: Momentum
| Feature Name | Category | Formula / Definition | Lookback | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `RSI_14` | Momentum | Relative Strength Index (14-day) | 14 | Measures overbought (>70) or oversold (<30) conditions |
| `MACD` | Momentum | `EMA_12 - EMA_26` | 26 | Relationship between two moving averages |
| `MACD_Signal` | Momentum | 9-day EMA of MACD | 35 (26+9) | Trigger line for MACD signals |
| `MACD_Hist` | Momentum | `MACD - MACD_Signal` | 35 | Divergence between MACD and Signal |
| `ROC` | Momentum | Rate of Change: `(Close_t - Close_{t-n}) / Close_{t-n}` | 12 (default) | Pure momentum/speed of price change |
| `Momentum` | Momentum | `Close_t - Close_{t-n}` | 10 (default) | Absolute price change momentum |

## Group 4: Volatility
| Feature Name | Category | Formula / Definition | Lookback | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `ATR` | Volatility | Average True Range | 14 | Absolute measure of daily price volatility |
| `Rolling_Vol` | Volatility | Standard deviation of returns (20-day) | 20 | Statistical historical volatility |
| `BB_Upper` | Volatility | `SMA_20 + (2 * std_20)` | 20 | Upper boundary of expected price band |
| `BB_Middle` | Volatility | `SMA_20` | 20 | Middle band (same as SMA_20) |
| `BB_Lower` | Volatility | `SMA_20 - (2 * std_20)` | 20 | Lower boundary of expected price band |
| `BB_Width` | Volatility | `(BB_Upper - BB_Lower) / BB_Middle` | 20 | Normalized band width; contraction/expansion |

## Group 5: Volume
| Feature Name | Category | Formula / Definition | Lookback | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `Vol_Change` | Volume | `(Volume_t - Volume_t-1) / Volume_t-1` | 1 | Velocity of liquidity changes |
| `Vol_SMA_20` | Volume | 20-day SMA of Volume | 20 | Baseline average liquidity |
| `Vol_Ratio` | Volume | `Volume / Vol_SMA_20` | 20 | Volume surge/drop relative to norm |
| `OBV` | Volume | On-Balance Volume | Cumulative | Cumulative buying/selling pressure |

## Group 6: Market Context
| Feature Name | Category | Formula / Definition | Lookback | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `NIFTY_Return` | Context | NIFTY 50 Daily Return | 1 | Macro market direction |
| `NIFTY_Vol` | Context | NIFTY 50 20-day rolling volatility | 20 | Macro market stability |
| `India_VIX` | Context | India VIX Close | 0 | Implied market volatility / fear index |
