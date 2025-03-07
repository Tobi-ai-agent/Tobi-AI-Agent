
# Trading environment parameters
LOOKBACK_WINDOW = 20  # Number of past candles to consider
TRADING_FEE = 0.001   # Trading fee as a fraction (0.1%)
INITIAL_BALANCE = 10000  # Initial balance for backtesting

# Action space
# 0: Hold, 1: Buy, 2: Sell
ACTION_SPACE = 3

# Reward settings
USE_SHARPE_RATIO = True  # Use Sharpe ratio component in reward
RISK_FREE_RATE = 0.0     # Risk-free rate for Sharpe ratio calculation