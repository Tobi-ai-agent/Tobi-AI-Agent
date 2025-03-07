# scripts/live_trading.py
import os
import argparse
import time
import datetime
import torch
import numpy as np
from PIL import Image, ImageGrab
import matplotlib.pyplot as plt
import json
import logging
import sys
from threading import Thread
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.agent import DQNAgent
from config.hyperparameters import *
from config.trading_config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LiveTrading")


class LiveTradingSystem:
    """Live trading system using trained DQN agent."""
    
    def __init__(self, model_path, capture_region=None, interval=60, 
                trading_enabled=False, api_key=None, api_secret=None,
                symbol="BTCUSD", timeframe="15m", max_position=1.0,
                stop_loss_pct=2.0, take_profit_pct=4.0, device=None):
        """Initialize the live trading system.
        
        Args:
            model_path (str): Path to the trained model
            capture_region (tuple): Screen region to capture (left, top, right, bottom)
            interval (int): Time between trades in seconds
            trading_enabled (bool): Whether to execute real trades
            api_key (str): Trading API key
            api_secret (str): Trading API secret
            symbol (str): Trading symbol
            timeframe (str): Chart timeframe
            max_position (float): Maximum position size
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
            device (torch.device): Device to use for inference
        """
        self.model_path = model_path
        self.capture_region = capture_region
        self.interval = interval
        self.trading_enabled = trading_enabled
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_position = max_position
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Trading state
        self.position = 0
        self.entry_price = 0
        self.balance = INITIAL_BALANCE  # From trading_config.py
        self.trade_history = []
        self.last_action = "INITIAL"
        self.last_price = 0
        self.stop_loss_price = 0
        self.take_profit_price = 0
        
        # Performance tracking
        self.portfolio_values = [self.balance]
        self.prices = []
        self.actions = []
        self.timestamps = []
        
        # Setup device
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
        
        # Initialize agent
        self._init_agent()
        
        # Connect to trading API if enabled
        if trading_enabled:
            self._connect_trading_api()
        
        # Create output directory for screenshots and results
        self.output_dir = os.path.join("live_results", 
                                      f"{symbol}_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save configuration
        self.config = {
            "model_path": model_path,
            "interval": interval,
            "trading_enabled": trading_enabled,
            "symbol": symbol,
            "timeframe": timeframe,
            "max_position": max_position,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "start_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
    
    def _init_agent(self):
        """Initialize the DQN agent with the trained model."""
        # Define state size based on INPUT_CHANNELS and IMAGE_SIZE from hyperparameters.py
        state_size = (INPUT_CHANNELS, *IMAGE_SIZE)
        action_size = ACTION_SPACE  # From trading_config.py
        
        # Create config dictionary
        config = {
            'INPUT_CHANNELS': INPUT_CHANNELS,
            'IMAGE_SIZE': IMAGE_SIZE,
            'BATCH_SIZE': BATCH_SIZE,
            'LEARNING_RATE': LEARNING_RATE,
            'USE_MPS': USE_MPS
        }
        
        # Initialize agent
        self.agent = DQNAgent(state_size, action_size, config, self.device)
        
        # Load trained model
        self.agent.load(self.model_path)
        logger.info(f"Loaded trained model from {self.model_path}")
    
    def _connect_trading_api(self):
        """Connect to trading API for executing real trades."""
        if not self.api_key or not self.api_secret:
            logger.error("API key and secret required for live trading")
            self.trading_enabled = False
            return
            
        try:
            # This is a placeholder for actual trading API integration
            # You would replace this with code for your specific broker/exchange
            
            # Example for a hypothetical trading API:
            # from trading_api import TradingAPI
            # self.api = TradingAPI(self.api_key, self.api_secret)
            # self.api.connect()
            
            logger.info("Connected to trading API")
            
            # Get account balance
            # self.balance = self.api.get_balance()
            # logger.info(f"Current account balance: ${self.balance:.2f}")
            
            # Get current position
            # self.position = self.api.get_position(self.symbol)
            # if self.position > 0:
            #     self.entry_price = self.api.get_position_entry_price(self.symbol)
            #     logger.info(f"Current position: {self.position} {self.symbol} at ${self.entry_price:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to connect to trading API: {e}")
            self.trading_enabled = False
    
    def capture_chart(self):
        """Capture chart image from screen or trading platform.
        
        Returns:
            numpy.array: Chart image as numpy array
        """
        try:
            if self.capture_region:
                # Capture specific screen region
                screenshot = ImageGrab.grab(bbox=self.capture_region)
            else:
                # Capture full screen
                screenshot = ImageGrab.grab()
            
            # Convert to numpy array
            img_array = np.array(screenshot)
            
            # Save screenshot
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            screenshot_path = os.path.join(self.output_dir, f"chart_{timestamp}.png")
            screenshot.save(screenshot_path)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error capturing chart: {e}")
            return None
    
    def get_current_price(self):
        """Get current price from trading platform or OCR from chart.
        
        Returns:
            float: Current price
        """
        # In a real implementation, you would:
        # 1. Get price from trading API if available
        # 2. Or extract price from chart image using OCR
        # 3. Or use websocket/API to get real-time price data
        
        # This is a placeholder for actual price retrieval
        if self.trading_enabled:
            try:
                # Example with hypothetical trading API:
                # return self.api.get_price(self.symbol)
                
                # For now, simulate price movement based on previous price
                if not self.prices:
                    price = 100.0  # Initial price if no history
                else:
                    # Random walk with slight upward bias
                    last_price = self.prices[-1]
                    price = last_price * (1 + np.random.normal(0.0001, 0.001))
            except Exception as e:
                logger.error(f"Error getting price from API: {e}")
                if self.prices:
                    price = self.prices[-1]
                else:
                    price = 100.0
        else:
            # Simulate price for testing
            if not self.prices:
                price = 100.0
            else:
                last_price = self.prices[-1]
                price = last_price * (1 + np.random.normal(0.0001, 0.001))
        
        return price
    
    def execute_trade(self, action):
        """Execute trade based on model prediction.
        
        Args:
            action (int): Action from agent (0: Hold, 1: Buy, 2: Sell)
            
        Returns:
            bool: Whether the trade was executed successfully
        """
        current_price = self.get_current_price()
        timestamp = datetime.datetime.now()
        
        # Map action to string for logging
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_str = action_map[action]
        
        # Log action
        logger.info(f"Action: {action_str}, Price: ${current_price:.2f}, Position: {self.position}")
        
        # Check if we need to execute a trade
        execute_trade = False
        trade_type = None
        trade_amount = 0
        
        if action == 1 and self.position < self.max_position:  # Buy
            # Calculate position size (using all available balance for simplicity)
            # In a real system, you'd use proper position sizing
            if self.trading_enabled:
                trade_amount = self.balance / current_price
                trade_amount = min(trade_amount, self.max_position - self.position)
            else:
                trade_amount = 0.1  # For simulation
            
            execute_trade = True
            trade_type = "BUY"
            
            # Set stop loss and take profit
            self.entry_price = current_price
            self.stop_loss_price = current_price * (1 - self.stop_loss_pct/100)
            self.take_profit_price = current_price * (1 + self.take_profit_pct/100)
            
        elif action == 2 and self.position > 0:  # Sell
            trade_amount = self.position
            execute_trade = True
            trade_type = "SELL"
        
        # Execute trade if needed
        if execute_trade:
            if self.trading_enabled:
                try:
                    # Example with hypothetical trading API:
                    # if trade_type == "BUY":
                    #     self.api.place_order(self.symbol, "BUY", trade_amount)
                    # else:
                    #     self.api.place_order(self.symbol, "SELL", trade_amount)
                    
                    logger.info(f"Executed {trade_type} of {trade_amount} {self.symbol} at ${current_price:.2f}")
                    
                    # Update balance and position
                    if trade_type == "BUY":
                        self.balance -= trade_amount * current_price
                        self.position += trade_amount
                    else:
                        self.balance += trade_amount * current_price
                        self.position = 0
                    
                except Exception as e:
                    logger.error(f"Error executing trade: {e}")
                    return False
            else:
                # Simulate trade for testing
                if trade_type == "BUY":
                    self.balance -= trade_amount * current_price
                    self.position += trade_amount
                    logger.info(f"[SIMULATION] Bought {trade_amount} {self.symbol} at ${current_price:.2f}")
                else:
                    self.balance += trade_amount * current_price
                    self.position = 0
                    logger.info(f"[SIMULATION] Sold {trade_amount} {self.symbol} at ${current_price:.2f}")
            
            # Record trade
            self.trade_history.append((trade_type, current_price, trade_amount, timestamp))
        
        # Update state
        self.last_action = action_str
        self.last_price = current_price
        
        # Update tracking
        portfolio_value = self.balance + (self.position * current_price)
        self.portfolio_values.append(portfolio_value)
        self.prices.append(current_price)
        self.actions.append(action)
        self.timestamps.append(timestamp)
        
        # Check stop loss and take profit
        if self.position > 0:
            if current_price <= self.stop_loss_price:
                logger.info(f"Stop loss triggered at ${current_price:.2f}")
                self.execute_trade(2)  # Sell
            elif current_price >= self.take_profit_price:
                logger.info(f"Take profit triggered at ${current_price:.2f}")
                self.execute_trade(2)  # Sell
        
        return True
    
    def visualize_state(self):
        """Visualize current trading state."""
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Get timestamps for x-axis
        x = range(len(self.portfolio_values))
        if self.timestamps:
            time_labels = [t.strftime('%H:%M:%S') for t in self.timestamps]
            if len(time_labels) < len(self.portfolio_values):
                time_labels = ["Start"] + time_labels
        else:
            time_labels = [str(i) for i in x]
        
        # Plot portfolio value
        ax1.plot(x, self.portfolio_values)
        ax1.set_title('Portfolio Value')
        ax1.set_ylabel('Value ($)')
        ax1.grid(True)
        
        if len(x) > 10:
            tick_indices = np.linspace(0, len(x)-1, 10, dtype=int)
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels([time_labels[i] for i in tick_indices], rotation=45)
        
        # Plot price and actions
        if self.prices:
            ax2.plot(range(len(self.prices)), self.prices)
            ax2.set_title(f'{self.symbol} Price')
            ax2.set_ylabel('Price ($)')
            ax2.grid(True)
            
            # Mark buy and sell points
            for i, action in enumerate(self.actions):
                if action == 1:  # Buy
                    ax2.scatter(i, self.prices[i], color='green', marker='^', s=100)
                elif action == 2:  # Sell
                    ax2.scatter(i, self.prices[i], color='red', marker='v', s=100)
            
            if len(self.prices) > 10:
                tick_indices = np.linspace(0, len(self.prices)-1, 10, dtype=int)
                ax2.set_xticks(tick_indices)
                ax2.set_xticklabels([time_labels[i+1] for i in tick_indices], rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(self.output_dir, f"state_{timestamp}.png"))
        plt.close()
    
    def save_results(self):
        """Save trading results and statistics."""
        # Calculate performance metrics
        if self.prices and self.portfolio_values:
            initial_value = self.portfolio_values[0]
            final_value = self.portfolio_values[-1]
            profit = final_value - initial_value
            profit_pct = (profit / initial_value) * 100
            
            trade_count = len([a for a in self.actions if a in [1, 2]])
            buy_count = len([a for a in self.actions if a == 1])
            sell_count = len([a for a in self.actions if a == 2])
            
            # Calculate win/loss ratio
            profitable_trades = 0
            for i in range(len(self.trade_history) - 1):
                if self.trade_history[i][0] == "BUY" and self.trade_history[i+1][0] == "SELL":
                    if self.trade_history[i+1][1] > self.trade_history[i][1]:
                        profitable_trades += 1
            
            win_rate = profitable_trades / buy_count if buy_count > 0 else 0
            
            results = {
                "initial_value": float(initial_value),
                "final_value": float(final_value),
                "profit": float(profit),
                "profit_pct": float(profit_pct),
                "trade_count": trade_count,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "win_rate": float(win_rate),
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "duration": len(self.actions) * self.interval,
                "start_time": self.timestamps[0].strftime('%Y-%m-%d %H:%M:%S') if self.timestamps else "",
                "end_time": self.timestamps[-1].strftime('%Y-%m-%d %H:%M:%S') if self.timestamps else "",
            }
            
            # Save results
            with open(os.path.join(self.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
            
            # Save trading history
            history_df = pd.DataFrame({
                "timestamp": self.timestamps[1:] if len(self.timestamps) > len(self.actions) else self.timestamps,
                "action": [action_map[a] for a in self.actions],
                "price": self.prices,
                "portfolio_value": self.portfolio_values[1:] if len(self.portfolio_values) > len(self.actions) else self.portfolio_values
            })
            
            history_df.to_csv(os.path.join(self.output_dir, "trading_history.csv"), index=False)
            
            logger.info(f"Results saved to {self.output_dir}")
            logger.info(f"Final profit: ${profit:.2f} ({profit_pct:.2f}%)")
            logger.info(f"Win rate: {win_rate*100:.2f}%")
            
            return results
    
    def run(self, duration=None, max_iterations=None):
        """Run the live trading system.
        
        Args:
            duration (int): Duration to run in seconds (None for indefinite)
            max_iterations (int): Maximum number of iterations (None for indefinite)
        """
        logger.info("Starting live trading system")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Symbol: {self.symbol}, Timeframe: {self.timeframe}")
        logger.info(f"Trading enabled: {self.trading_enabled}")
        
        iteration = 0
        start_time = time.time()
        
        try:
            while True:
                # Check termination conditions
                if duration and (time.time() - start_time) > duration:
                    logger.info(f"Reached maximum duration of {duration} seconds")
                    break
                
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached maximum iterations: {max_iterations}")
                    break
                
                # Capture chart image
                logger.info(f"Iteration {iteration}: Capturing chart...")
                chart_img = self.capture_chart()
                
                if chart_img is None:
                    logger.error("Failed to capture chart. Retrying...")
                    time.sleep(self.interval / 2)
                    continue
                
                # Get action from agent
                logger.info("Getting action from agent...")
                action = self.agent.act(chart_img, epsilon=0)  # No exploration in live trading
                
                # Execute trade
                success = self.execute_trade(action)
                
                if not success:
                    logger.error("Failed to execute trade. Continuing...")
                
                # Visualize state periodically
                if iteration % 10 == 0:
                    self.visualize_state()
                
                # Wait for next interval
                iteration += 1
                logger.info(f"Waiting for {self.interval} seconds until next action...")
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            # Save final results
            self.visualize_state()
            self.save_results()
            logger.info("Trading session ended")


def main():
    """Main function to run live trading."""
    parser = argparse.ArgumentParser(description='Run live trading with trained DQN agent')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--interval', type=int, default=60, help='Time between trades in seconds')
    parser.add_argument('--region', type=str, help='Screen region to capture as "left,top,right,bottom"')
    parser.add_argument('--symbol', type=str, default='BTCUSD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='15m', help='Chart timeframe')
    parser.add_argument('--live', action='store_true', help='Enable live trading (default is simulation)')
    parser.add_argument('--api_key', type=str, help='Trading API key')
    parser.add_argument('--api_secret', type=str, help='Trading API secret')
    parser.add_argument('--max_position', type=float, default=1.0, help='Maximum position size')
    parser.add_argument('--stop_loss', type=float, default=2.0, help='Stop loss percentage')
    parser.add_argument('--take_profit', type=float, default=4.0, help='Take profit percentage')
    parser.add_argument('--duration', type=int, help='Duration to run in seconds')
    parser.add_argument('--iterations', type=int, help='Maximum number of iterations')
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
    
    # Check if trading is enabled but API credentials are missing
    if args.live and (not args.api_key or not args.api_secret):
        logger.warning("Live trading enabled but API credentials missing. Falling back to simulation mode.")
        args.live = False
    
    # Initialize trading system
    trading_system = LiveTradingSystem(
        model_path=args.model,
        capture_region=capture_region,
        interval=args.interval,
        trading_enabled=args.live,
        api_key=args.api_key,
        api_secret=args.api_secret,
        symbol=args.symbol,
        timeframe=args.timeframe,
        max_position=args.max_position,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        device=device
    )
    
    # Run trading system
    trading_system.run(duration=args.duration, max_iterations=args.iterations)


if __name__ == "__main__":
    main()