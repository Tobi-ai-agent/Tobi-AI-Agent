#!/usr/bin/env python
# scripts/live_crypto_trader.py
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageGrab
import torch
import logging
import json
import threading
import ccxt
import datetime
from tqdm import tqdm
from colorama import Fore, Style, init
import getpass

# Initialize colorama
init(autoreset=True)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.agent import DQNAgent
from src.utils.visualization import plot_trade_performance, generate_performance_report
from config.hyperparameters import *
from config.trading_config import *
from src.wallet.crypto_wallet import EthereumWallet, SolanaWallet

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LiveCryptoTrader")

class LiveCryptoTrader:
    """Live cryptocurrency trading system using trained DQN agent."""
    
    def __init__(self, model_path, exchange_id='binance', symbols=None, timeframe='1h', 
                wallet_type=None, wallet_path=None, wallet_password=None,
                usdc_trading=True, paper_trading=True, max_position=1.0,
                stop_loss_pct=2.0, take_profit_pct=4.0, update_interval=60,
                device=None, capture_mode=None, capture_region=None):
        """Initialize the live trading system.
        
        Args:
            model_path (str): Path to the trained model
            exchange_id (str): Exchange ID (e.g., 'binance', 'coinbase')
            symbols (list): List of symbols to trade
            timeframe (str): Timeframe to use (e.g., '1h', '4h')
            wallet_type (str): Wallet type ('ethereum', 'solana')
            wallet_path (str): Path to wallet file
            wallet_password (str): Wallet password
            usdc_trading (bool): Whether to trade with USDC pairs
            paper_trading (bool): Whether to use paper trading
            max_position (float): Maximum position size
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
            update_interval (int): Chart update interval in seconds
            device (torch.device): Device to use for inference
            capture_mode (str): Chart capture mode ('screenshot', 'api')
            capture_region (tuple): Screen region to capture (left, top, right, bottom)
        """
        self.model_path = model_path
        self.exchange_id = exchange_id
        self.symbols = symbols or ['BTC/USDC', 'ETH/USDC', 'SOL/USDC', 'TRX/USDC']
        self.timeframe = timeframe
        self.wallet_type = wallet_type
        self.wallet_path = wallet_path
        self.wallet_password = wallet_password
        self.usdc_trading = usdc_trading
        self.paper_trading = paper_trading
        self.max_position = max_position
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.update_interval = update_interval
        self.capture_mode = capture_mode or ('api' if paper_trading else 'screenshot')
        self.capture_region = capture_region
        
        # State variables
        self.running = False
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.entry_prices = {symbol: 0 for symbol in self.symbols}
        self.stop_losses = {symbol: 0 for symbol in self.symbols}
        self.take_profits = {symbol: 0 for symbol in self.symbols}
        self.balances = {'USDC': 0}
        self.trade_history = []
        self.pnl = {'daily': 0, 'total': 0}
        
        # Performance tracking
        self.portfolio_values = []
        self.timestamps = []
        
        # Set device
        if device:
            self.device = device
        elif USE_MPS and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) device")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            logger.info("Using CUDA device")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device")
        
        # Initialize agent, exchange, and wallet
        self._init_agent()
        self._connect_exchange()
        
        if not self.paper_trading:
            self._init_wallet()
        
        # Create output directory
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join('trading_results', f"{exchange_id}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save configuration
        self.config = {
            'model_path': model_path,
            'exchange_id': exchange_id,
            'symbols': self.symbols,
            'timeframe': timeframe,
            'wallet_type': wallet_type,
            'usdc_trading': usdc_trading,
            'paper_trading': paper_trading,
            'max_position': max_position,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'update_interval': update_interval,
            'capture_mode': self.capture_mode,
            'timestamp': timestamp
        }
        
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _init_agent(self):
        """Initialize the DQN agent with the trained model."""
        # Define state size based on INPUT_CHANNELS and IMAGE_SIZE
        state_size = (INPUT_CHANNELS, *IMAGE_SIZE)
        action_size = ACTION_SPACE
        
        # Create complete config dictionary with all required parameters
        config = {
            'INPUT_CHANNELS': INPUT_CHANNELS,
            'IMAGE_SIZE': IMAGE_SIZE,
            'BATCH_SIZE': BATCH_SIZE,
            'LEARNING_RATE': LEARNING_RATE,
            'GAMMA': GAMMA,
            'EPSILON_START': EPSILON_START,
            'EPSILON_END': EPSILON_END,
            'EPSILON_DECAY': EPSILON_DECAY,
            'MEMORY_SIZE': MEMORY_SIZE,
            'TARGET_UPDATE': TARGET_UPDATE,
            'UPDATE_EVERY': UPDATE_EVERY,
            'USE_MPS': USE_MPS
        }
        
        # Initialize agent
        self.agent = DQNAgent(state_size, action_size, config, self.device)
        
        # Load trained model
        self.agent.load(self.model_path)
        logger.info(f"Loaded trained model from {self.model_path}")
    
    def _connect_exchange(self):
        """Connect to cryptocurrency exchange."""
        try:
            # Initialize exchange
            if self.paper_trading:
                # Paper trading uses exchange API for data but not for trading
                self.exchange = getattr(ccxt, self.exchange_id)({
                    'enableRateLimit': True,
                })
                logger.info(f"Connected to {self.exchange_id} in paper trading mode")
            else:
                # Live trading requires API keys
                api_key = os.environ.get(f"{self.exchange_id.upper()}_API_KEY")
                api_secret = os.environ.get(f"{self.exchange_id.upper()}_API_SECRET")
                
                if not api_key or not api_secret:
                    api_key = input(f"Enter {self.exchange_id} API key: ")
                    api_secret = getpass.getpass(f"Enter {self.exchange_id} API secret: ")
                
                self.exchange = getattr(ccxt, self.exchange_id)({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                })
                logger.info(f"Connected to {self.exchange_id} for live trading")
            
            # Load markets
            self.exchange.load_markets()
            
            # Verify all symbols are valid
            for symbol in self.symbols:
                if symbol not in self.exchange.markets:
                    logger.warning(f"Symbol {symbol} not available on {self.exchange_id}")
            
            # Get initial balances
            if not self.paper_trading:
                self._update_balances()
            else:
                # Initialize with paper trading balance
                self.balances = {'USDC': 10000}  # Default paper trading balance
                for symbol in self.symbols:
                    base = symbol.split('/')[0]
                    self.balances[base] = 0
        
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {e}")
            raise
    
    def _init_wallet(self):
        """Initialize cryptocurrency wallet."""
        if not self.wallet_type:
            return
        
        try:
            if self.wallet_type.lower() == 'ethereum':
                self.wallet = EthereumWallet(wallet_name=self.wallet_path)
            elif self.wallet_type.lower() == 'solana':
                self.wallet = SolanaWallet(wallet_name=self.wallet_path)
            else:
                logger.error(f"Unsupported wallet type: {self.wallet_type}")
                return
            
            # Load wallet
            if self.wallet_path and os.path.exists(self.wallet_path):
                wallet_data = self.wallet.load_wallet(self.wallet_password)
                if wallet_data:
                    logger.info(f"Loaded {self.wallet_type} wallet: {self.wallet.get_address()}")
                else:
                    logger.error(f"Failed to load wallet from {self.wallet_path}")
            else:
                create_new = input("Wallet not found. Create new wallet? (y/n): ")
                if create_new.lower() == 'y':
                    if not self.wallet_password:
                        self.wallet_password = getpass.getpass("Enter new wallet password: ")
                    
                    wallet_data = self.wallet.create_wallet(self.wallet_password)
                    if wallet_data:
                        logger.info(f"Created new {self.wallet_type} wallet: {self.wallet.get_address()}")
                    else:
                        logger.error("Failed to create new wallet")
                else:
                    import_key = input("Import private key? (y/n): ")
                    if import_key.lower() == 'y':
                        private_key = getpass.getpass("Enter private key: ")
                        if not self.wallet_password:
                            self.wallet_password = getpass.getpass("Enter wallet password: ")
                        
                        wallet_data = self.wallet.import_wallet(private_key, self.wallet_password)
                        if wallet_data:
                            logger.info(f"Imported {self.wallet_type} wallet: {self.wallet.get_address()}")
                        else:
                            logger.error("Failed to import wallet")
        
        except Exception as e:
            logger.error(f"Failed to initialize wallet: {e}")
    
    def _update_balances(self):
        """Update account balances."""
        try:
            # Get exchange balances
            if self.paper_trading:
                return
            
            balances = self.exchange.fetch_balance()
            
            # Update balances
            self.balances = {}
            for symbol in self.symbols:
                base, quote = symbol.split('/')
                
                if base in balances['total']:
                    self.balances[base] = balances['total'][base]
                else:
                    self.balances[base] = 0
                
                if quote in balances['total']:
                    self.balances[quote] = balances['total'][quote]
                else:
                    self.balances[quote] = 0
            
            # Make sure USDC balance is tracked
            if 'USDC' not in self.balances:
                if 'USDC' in balances['total']:
                    self.balances['USDC'] = balances['total']['USDC']
                else:
                    self.balances['USDC'] = 0
            
            logger.info(f"Updated balances: {self.balances}")
        
        except Exception as e:
            logger.error(f"Failed to update balances: {e}")
    
    def _get_current_price(self, symbol):
        """Get current price for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            float: Current price
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None
    
    def _capture_chart(self, symbol):
        """Capture chart for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            numpy.array: Chart image as numpy array
        """
        try:
            if self.capture_mode == 'screenshot':
                # Capture screen region
                if self.capture_region:
                    screenshot = ImageGrab.grab(bbox=self.capture_region)
                else:
                    screenshot = ImageGrab.grab()
                
                return np.array(screenshot)
            
            elif self.capture_mode == 'api':
                # Generate chart image from API data
                now = datetime.datetime.now()
                start_time = now - datetime.timedelta(days=7)
                
                # Fetch candles
                candles = self.exchange.fetch_ohlcv(
                    symbol, 
                    self.timeframe,
                    since=int(start_time.timestamp() * 1000),
                    limit=100
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Create figure
                plt.figure(figsize=(8, 6))
                
                # Plot candlesticks
                width = 0.6
                width2 = width * 0.8
                
                up = df[df.close >= df.open]
                down = df[df.close < df.open]
                
                # Candlestick bodies
                plt.bar(up.index, up.close - up.open, width, bottom=up.open, color='green')
                plt.bar(down.index, down.close - down.open, width, bottom=down.open, color='red')
                
                # Candlestick wicks
                plt.bar(up.index, up.high - up.close, width2, bottom=up.close, color='green')
                plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color='green')
                plt.bar(down.index, down.high - down.open, width2, bottom=down.open, color='red')
                plt.bar(down.index, down.low - down.close, width2, bottom=down.close, color='red')
                
                # Add title and styling
                plt.title(f"{symbol} ({self.timeframe}) - {df.iloc[-1]['close']:.2f}")
                plt.grid(True, color='#333333', linestyle='-', linewidth=0.5, alpha=0.3)
                plt.xticks(range(0, len(df), 10), df['timestamp'].iloc[::10].dt.strftime('%m-%d'), rotation=45)
                
                # Save to buffer
                from io import BytesIO
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                plt.close()
                buf.seek(0)
                
                # Convert to numpy array
                chart_img = np.array(Image.open(buf))
                
                return chart_img
        
        except Exception as e:
            logger.error(f"Failed to capture chart for {symbol}: {e}")
            return None
    
    def _execute_trade(self, symbol, action, price=None):
        """Execute trade for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            action (int): Action (0: Hold, 1: Buy, 2: Sell)
            price (float, optional): Current price
            
        Returns:
            dict: Trade result
        """
        try:
            # Map action to string
            action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
            action_str = action_map[action]
            
            # Get current price if not provided
            if not price:
                price = self._get_current_price(symbol)
                if not price:
                    return None
            
            base, quote = symbol.split('/')
            
            # Skip if already in position for buy or no position for sell
            if action == 1 and self.positions[symbol] > 0:
                logger.info(f"Already in position for {symbol}, skipping buy")
                return None
            
            if action == 2 and self.positions[symbol] == 0:
                logger.info(f"No position for {symbol}, skipping sell")
                return None
            
            # Execute trade
            if action == 1:  # Buy
                if self.paper_trading:
                    # Paper trading
                    # Calculate amount to buy based on max position and available balance
                    usdc_amount = min(self.balances['USDC'], self.max_position * price)
                    amount = usdc_amount / price
                    
                    # Update balances
                    self.balances['USDC'] -= usdc_amount
                    self.balances[base] += amount
                    
                    # Update position
                    self.positions[symbol] = amount
                    self.entry_prices[symbol] = price
                    
                    # Set stop loss and take profit
                    self.stop_losses[symbol] = price * (1 - self.stop_loss_pct / 100)
                    self.take_profits[symbol] = price * (1 + self.take_profit_pct / 100)
                    
                    trade_result = {
                        'symbol': symbol,
                        'action': action_str,
                        'price': price,
                        'amount': amount,
                        'total': usdc_amount,
                        'timestamp': datetime.datetime.now()
                    }
                
                else:
                    # Live trading
                    # Calculate amount to buy based on max position and available balance
                    usdc_balance = self.balances['USDC']
                    usdc_amount = min(usdc_balance, self.max_position * price)
                    amount = usdc_amount / price
                    
                    # Check minimum order amount
                    market = self.exchange.market(symbol)
                    if 'limits' in market and 'amount' in market['limits']:
                        min_amount = market['limits']['amount']['min']
                        if amount < min_amount:
                            logger.warning(f"Order amount {amount} below minimum {min_amount} for {symbol}")
                            return None
                    
                    # Execute order
                    order = self.exchange.create_market_buy_order(symbol, amount)
                    
                    # Update balances
                    self._update_balances()
                    
                    # Update position
                    self.positions[symbol] = amount
                    self.entry_prices[symbol] = price
                    
                    # Set stop loss and take profit
                    self.stop_losses[symbol] = price * (1 - self.stop_loss_pct / 100)
                    self.take_profits[symbol] = price * (1 + self.take_profit_pct / 100)
                    
                    trade_result = {
                        'symbol': symbol,
                        'action': action_str,
                        'price': price,
                        'amount': amount,
                        'total': usdc_amount,
                        'timestamp': datetime.datetime.now(),
                        'order_id': order['id']
                    }
            
            elif action == 2:  # Sell
                amount = self.positions[symbol]
                
                if self.paper_trading:
                    # Paper trading
                    # Calculate total
                    usdc_amount = amount * price
                    
                    # Update balances
                    self.balances['USDC'] += usdc_amount
                    self.balances[base] -= amount
                    
                    # Calculate profit/loss
                    entry_value = amount * self.entry_prices[symbol]
                    exit_value = amount * price
                    pnl = exit_value - entry_value
                    
                    # Update position
                    self.positions[symbol] = 0
                    
                    trade_result = {
                        'symbol': symbol,
                        'action': action_str,
                        'price': price,
                        'amount': amount,
                        'total': usdc_amount,
                        'pnl': pnl,
                        'pnl_percent': (pnl / entry_value * 100) if entry_value > 0 else 0,
                        'timestamp': datetime.datetime.now()
                    }
                    
                    # Update PnL
                    self.pnl['total'] += pnl
                    self.pnl['daily'] += pnl
                
                else:
                    # Live trading
                    # Execute order
                    order = self.exchange.create_market_sell_order(symbol, amount)
                    
                    # Calculate profit/loss
                    entry_value = amount * self.entry_prices[symbol]
                    exit_value = amount * price
                    pnl = exit_value - entry_value
                    
                    # Update balances
                    self._update_balances()
                    
                    # Update position
                    self.positions[symbol] = 0
                    
                    trade_result = {
                        'symbol': symbol,
                        'action': action_str,
                        'price': price,
                        'amount': amount,
                        'total': amount * price,
                        'pnl': pnl,
                        'pnl_percent': (pnl / entry_value * 100) if entry_value > 0 else 0,
                        'timestamp': datetime.datetime.now(),
                        'order_id': order['id']
                    }
                    
                    # Update PnL
                    self.pnl['total'] += pnl
                    self.pnl['daily'] += pnl
            
            else:  # Hold
                return None
            
            # Add to trade history
            self.trade_history.append(trade_result)
            
            # Print trade info
            if action == 1:
                logger.info(f"{Fore.GREEN}BUY {symbol}: {amount:.6f} @ ${price:.2f} = ${amount * price:.2f}")
                print(f"{Fore.GREEN}BUY {symbol}: {amount:.6f} @ ${price:.2f} = ${amount * price:.2f}")
            elif action == 2:
                pnl_str = f"PnL: ${trade_result['pnl']:.2f} ({trade_result['pnl_percent']:.2f}%)"
                logger.info(f"{Fore.RED}SELL {symbol}: {amount:.6f} @ ${price:.2f} = ${amount * price:.2f} | {pnl_str}")
                print(f"{Fore.RED}SELL {symbol}: {amount:.6f} @ ${price:.2f} = ${amount * price:.2f} | {pnl_str}")
            
            # Save trade to file
            with open(os.path.join(self.output_dir, 'trades.jsonl'), 'a') as f:
                f.write(json.dumps(trade_result) + '\n')
            
            return trade_result
        
        except Exception as e:
            logger.error(f"Failed to execute {action_map[action]} for {symbol}: {e}")
            return None
    
    def _check_stop_loss_take_profit(self, symbol, price):
        """Check if stop loss or take profit is triggered.
        
        Args:
            symbol (str): Trading pair symbol
            price (float): Current price
            
        Returns:
            bool: Whether a trade was executed
        """
        try:
            # Skip if no position
            if self.positions[symbol] == 0:
                return False
            
            # Check stop loss
            if price <= self.stop_losses[symbol]:
                logger.info(f"Stop loss triggered for {symbol} at ${price:.2f}")
                self._execute_trade(symbol, 2, price)
                return True
            
            # Check take profit
            if price >= self.take_profits[symbol]:
                logger.info(f"Take profit triggered for {symbol} at ${price:.2f}")
                self._execute_trade(symbol, 2, price)
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to check stop loss/take profit for {symbol}: {e}")
            return False
    
    def _calculate_portfolio_value(self):
        """Calculate total portfolio value.
        
        Returns:
            float: Total portfolio value
        """
        try:
            total = self.balances['USDC']
            
            for symbol in self.symbols:
                base = symbol.split('/')[0]
                if base in self.balances and self.balances[base] > 0:
                    price = self._get_current_price(symbol)
                    if price:
                        total += self.balances[base] * price
            
            return total
        
        except Exception as e:
            logger.error(f"Failed to calculate portfolio value: {e}")
            return sum(self.balances.values())
    
    def _display_status(self):
        """Display current trading status."""
        try:
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value()
            
            # Add to history
            self.portfolio_values.append(portfolio_value)
            self.timestamps.append(datetime.datetime.now())
            
            # Clear terminal
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Print header
            print(f"\n{Fore.CYAN}===== Live Crypto Trader =====")
            print(f"{Fore.CYAN}Model: {self.model_path}")
            print(f"{Fore.CYAN}Exchange: {self.exchange_id} ({'Paper Trading' if self.paper_trading else 'Live Trading'})")
            print(f"{Fore.CYAN}Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{Fore.CYAN}==============================\n")
            
            # Print portfolio value
            initial_value = self.portfolio_values[0] if self.portfolio_values else portfolio_value
            pct_change = (portfolio_value - initial_value) / initial_value * 100 if initial_value > 0 else 0
            
            print(f"{Fore.YELLOW}Portfolio Value: ${portfolio_value:.2f} ({pct_change:+.2f}%)")
            print(f"{Fore.YELLOW}Daily PnL: ${self.pnl['daily']:.2f}")
            print(f"{Fore.YELLOW}Total PnL: ${self.pnl['total']:.2f}\n")
            
            # Print balances
            print(f"{Fore.WHITE}Balances:")
            for currency, balance in self.balances.items():
                if balance > 0:
                    print(f"  {currency}: {balance:.6f}")
            
            # Print positions
            print(f"\n{Fore.WHITE}Active Positions:")
            for symbol, amount in self.positions.items():
                if amount > 0:
                    price = self._get_current_price(symbol)
                    if price:
                        value = amount * price
                        entry_value = amount * self.entry_prices[symbol]
                        pnl = value - entry_value
                        pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                        
                        print(f"  {symbol}: {amount:.6f} @ ${self.entry_prices[symbol]:.2f} "
                              f"(Current: ${price:.2f}, PnL: ${pnl:.2f} / {pnl_pct:+.2f}%)")
                        print(f"     Stop Loss: ${self.stop_losses[symbol]:.2f}, "
                              f"Take Profit: ${self.take_profits[symbol]:.2f}")
            
            # Print recent trades
            print(f"\n{Fore.WHITE}Recent Trades:")
            for trade in reversed(self.trade_history[-5:]):
                timestamp = trade['timestamp'].strftime('%H:%M:%S') if isinstance(trade['timestamp'], datetime.datetime) else trade['timestamp']
                if trade['action'] == 'buy':
                    print(f"  [{timestamp}] {Fore.GREEN}BUY {trade['symbol']}: "
                          f"{trade['amount']:.6f} @ ${trade['price']:.2f} = ${trade['total']:.2f}")
                elif trade['action'] == 'sell':
                    print(f"  [{timestamp}] {Fore.RED}SELL {trade['symbol']}: "
                          f"{trade['amount']:.6f} @ ${trade['price']:.2f} = ${trade['total']:.2f} "
                          f"(PnL: ${trade.get('pnl', 0):.2f} / {trade.get('pnl_percent', 0):+.2f}%)")
        
        except Exception as e:
            logger.error(f"Failed to display status: {e}")
    
    def _save_results(self):
        """Save trading results."""
        try:
            # Calculate final portfolio value
            portfolio_value = self._calculate_portfolio_value()
            
            # Save trade history
            if self.trade_history:
                # Convert to DataFrame and save
                trades_df = pd.DataFrame(self.trade_history)
                trades_df.to_csv(os.path.join(self.output_dir, 'trade_history.csv'), index=False)
            
            # Save portfolio value history
            if self.portfolio_values:
                portfolio_df = pd.DataFrame({
                    'timestamp': self.timestamps,
                    'value': self.portfolio_values
                })
                portfolio_df.to_csv(os.path.join(self.output_dir, 'portfolio_history.csv'), index=False)
            
            # Save summary
            summary = {
                'start_time': self.timestamps[0].strftime('%Y-%m-%d %H:%M:%S') if self.timestamps else None,
                'end_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'initial_value': self.portfolio_values[0] if self.portfolio_values else None,
                'final_value': portfolio_value,
                'total_pnl': self.pnl['total'],
                'total_trades': len(self.trade_history),
                'symbols': self.symbols,
                'exchange': self.exchange_id,
                'paper_trading': self.paper_trading
            }
            
            with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Generate performance report
            if self.portfolio_values:
                plot_trade_performance(
                    self.trade_history,
                    self.portfolio_values,
                    self.portfolio_values[0] if self.portfolio_values else portfolio_value,
                    save_path=os.path.join(self.output_dir, 'performance.png')
                )
            
            if self.trade_history:
                generate_performance_report(
                    self.trade_history,
                    self.portfolio_values,
                    self.portfolio_values[0] if self.portfolio_values else portfolio_value,
                    portfolio_value,
                    save_path=os.path.join(self.output_dir, 'performance_report.txt')
                )
            
            logger.info(f"Saved trading results to {self.output_dir}")
        
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _trading_loop(self):
        """Main trading loop."""
        try:
            # Initial status update
            self._display_status()
            
            # Trading loop
            while self.running:
                # Process each symbol
                for symbol in self.symbols:
                    try:
                        # Get current price
                        price = self._get_current_price(symbol)
                        if not price:
                            continue
                        
                        # Check stop loss and take profit first
                        if self._check_stop_loss_take_profit(symbol, price):
                            continue  # Skip further processing if trade was executed
                        
                        # Capture chart
                        chart_img = self._capture_chart(symbol)
                        if chart_img is None:
                            continue
                        
                        # Get action from agent
                        action = self.agent.act(chart_img, epsilon=0)  # No exploration in live trading
                        
                        # Execute trade
                        self._execute_trade(symbol, action, price)
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                # Update and display status
                self._display_status()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
                # Reset daily PnL at midnight
                now = datetime.datetime.now()
                if now.hour == 0 and now.minute == 0:
                    self.pnl['daily'] = 0
        
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            self.running = False
            self._save_results()
    
    def start(self):
        """Start the trading system."""
        logger.info("Starting trading system")
        
        # Initialize portfolio value
        initial_value = self._calculate_portfolio_value()
        self.portfolio_values.append(initial_value)
        self.timestamps.append(datetime.datetime.now())
        
        # Start trading loop
        self.running = True
        self._trading_loop()
    
    def stop(self):
        """Stop the trading system."""
        logger.info("Stopping trading system")
        self.running = False

def main():
    """Main function to run the live trading system."""
    parser = argparse.ArgumentParser(description='Live Crypto Trading with DCNN')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange ID (e.g., binance, coinbase)')
    parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDC', 'ETH/USDC', 'SOL/USDC', 'TRX/USDC'], 
                        help='Symbols to trade')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe to use')
    parser.add_argument('--wallet', type=str, choices=['ethereum', 'solana'], help='Wallet type')
    parser.add_argument('--wallet_path', type=str, help='Path to wallet file')
    parser.add_argument('--no_usdc', action='store_false', dest='usdc_trading', help='Disable USDC trading')
    parser.add_argument('--live', action='store_true', dest='live_trading', help='Enable live trading')
    parser.add_argument('--max_position', type=float, default=1.0, help='Maximum position size')
    parser.add_argument('--stop_loss', type=float, default=2.0, help='Stop loss percentage')
    parser.add_argument('--take_profit', type=float, default=4.0, help='Take profit percentage')
    parser.add_argument('--interval', type=int, default=60, help='Update interval in seconds')
    parser.add_argument('--capture', type=str, choices=['screenshot', 'api'], help='Chart capture mode')
    parser.add_argument('--region', type=str, help='Screen region to capture as "left,top,right,bottom"')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Parse screen region if provided
    capture_region = None
    if args.region:
        try:
            capture_region = tuple(map(int, args.region.split(',')))
            if len(capture_region) != 4:
                raise ValueError("Region must be 4 integers: left,top,right,bottom")
        except Exception as e:
            logger.error(f"Invalid region format: {e}")
            return
    
    # Set device
    device = None
    if args.cpu:
        device = torch.device("cpu")
        logger.info("Forcing CPU usage as requested")
    
    # Initialize trader
    trader = LiveCryptoTrader(
        model_path=args.model,
        exchange_id=args.exchange,
        symbols=args.symbols,
        timeframe=args.timeframe,
        wallet_type=args.wallet,
        wallet_path=args.wallet_path,
        usdc_trading=args.usdc_trading,
        paper_trading=not args.live_trading,
        max_position=args.max_position,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        update_interval=args.interval,
        device=device,
        capture_mode=args.capture,
        capture_region=capture_region
    )
    
    # Start trading
    trader.start()

if __name__ == "__main__":
    main()