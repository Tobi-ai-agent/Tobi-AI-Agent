# src/environment/trading_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import os
from PIL import Image
import pandas as pd


class TradingEnvironment(gym.Env):
    """Custom trading environment for reinforcement learning based on chart images."""
    
    def __init__(self, data_dir, metadata_path=None, lookback_window=20, 
                 initial_balance=10000, commission=0.001, use_sharpe=True):
        """Initialize the trading environment.
        
        Args:
            data_dir (str): Directory containing chart images
            metadata_path (str): Path to metadata CSV (optional)
            lookback_window (int): Number of past candles to consider
            initial_balance (float): Initial account balance
            commission (float): Trading commission as a fraction
            use_sharpe (bool): Whether to include Sharpe ratio in reward
        """
        super(TradingEnvironment, self).__init__()
        
        # Load chart images and metadata
        self.chart_images, self.metadata = self._load_data(data_dir, metadata_path)
        
        # Environment parameters
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.commission = commission
        self.use_sharpe = use_sharpe
        
        # Define action and observation spaces
        # Actions: 0 (Hold), 1 (Buy), 2 (Sell)
        self.action_space = spaces.Discrete(3)
        
        # Image shape from the first image
        sample_image = self.chart_images[0]
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=sample_image.shape, 
            dtype=np.uint8
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None):
        """Reset the environment to initial state.
        
        Returns:
            numpy.array: Initial state
            dict: Info dictionary
        """
        super().reset(seed=seed)
        
        # Reset trading variables
        self.balance = self.initial_balance
        self.position = 0  # 0 units of asset
        self.current_step = self.lookback_window
        self.trade_history = []
        self.asset_value = 0
        self.total_reward = 0
        self.portfolio_values = [self.initial_balance]
        self.returns = []
        
        # Get initial observation
        observation = self._next_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Take an action in the environment.
        
        Args:
            action (int): Action to take (0: Hold, 1: Buy, 2: Sell)
            
        Returns:
            tuple: (next_state, reward, done, truncated, info)
        """
        # Get current price
        current_price = self._get_current_price()
        
        # Process action
        reward = 0
        done = False
        truncated = False
        info = {}
        
        # Track portfolio value before action
        portfolio_value_before = self.balance + (self.position * current_price)
        
        if action == 1:  # Buy
            if self.balance > 0:
                # Calculate max units we can buy with current balance
                max_units = self.balance / (current_price * (1 + self.commission))
                self.position += max_units
                self.balance -= max_units * current_price * (1 + self.commission)
                self.trade_history.append(('buy', current_price, max_units, self.current_step))
        
        elif action == 2:  # Sell
            if self.position > 0:
                # Sell all units
                self.balance += self.position * current_price * (1 - self.commission)
                self.trade_history.append(('sell', current_price, self.position, self.current_step))
                self.position = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value after action
        portfolio_value_after = self.balance + (self.position * current_price)
        self.portfolio_values.append(portfolio_value_after)
        
        # Calculate simple return
        if len(self.portfolio_values) > 1:
            daily_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
            self.returns.append(daily_return)
        
        # Calculate reward (change in portfolio value)
        reward = portfolio_value_after - portfolio_value_before
        
        # Add Sharpe ratio component to reward if enabled
        if self.use_sharpe and len(self.returns) > 1:
            sharpe = self._calculate_sharpe_ratio()
            reward += sharpe * 0.1  # Scale Sharpe component
        
        self.total_reward += reward
        
        # Check if episode is done
        if self.current_step >= len(self.chart_images) - 1:
            done = True
            info['final_balance'] = self.balance + self.position * current_price
            info['initial_balance'] = self.initial_balance
            info['profit_pct'] = ((self.balance + self.position * current_price) / self.initial_balance - 1) * 100
            info['total_trades'] = len(self.trade_history)
            
            # Calculate win rate
            if len(self.trade_history) > 1:
                buy_sell_pairs = []
                buy_orders = [t for t in self.trade_history if t[0] == 'buy']
                sell_orders = [t for t in self.trade_history if t[0] == 'sell']
                profitable_trades = 0
                
                for buy, sell in zip(buy_orders, sell_orders):
                    if sell[1] > buy[1]:  # If sell price > buy price
                        profitable_trades += 1
                
                if len(buy_orders) > 0:
                    info['win_rate'] = profitable_trades / len(buy_orders)
                else:
                    info['win_rate'] = 0
        
        # Return next observation, reward, done flag, truncated flag, and info
        return self._next_observation(), reward, done, truncated, info
    
    def _next_observation(self):
        """Get the next observation (chart image).
        
        Returns:
            numpy.array: Chart image as observation
        """
        return self.chart_images[self.current_step]
    
    def _get_current_price(self):
        """Extract current price from metadata or estimate it.
        
        Returns:
            float: Current price
        """
        if self.metadata is not None and 'price' in self.metadata.columns:
            price = self.metadata.iloc[self.current_step]['price']
            if pd.notna(price):
                return price
        
        # Fallback: generate a simulated price if no real price data is available
        # In a real-world scenario, you'd extract this from chart data or metadata
        base_price = 100
        time_component = np.sin(self.current_step / 20) * 10
        trend_component = self.current_step / 100
        return base_price + time_component + trend_component
    
   # Update the trading_env.py file with this improved _load_data method

    def _load_data(self, data_dir, metadata_path=None):
        """Load chart images and metadata.
        
        Args:
            data_dir (str): Directory containing chart images
            metadata_path (str): Path to metadata CSV (optional)
            
        Returns:
            tuple: (list of images, metadata dataframe)
        """
        # Load metadata if provided
        metadata = None
        if metadata_path and os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            print(f"Loaded metadata with {len(metadata)} records")
            
            # Process image files from metadata
            image_files = []
            
            if 'processed_path' in metadata.columns:
                # First try the paths as they are in the metadata
                original_paths = metadata['processed_path'].tolist()
                valid_paths = [path for path in original_paths if os.path.exists(path)]
                
                if len(valid_paths) == 0:
                    print("WARNING: No valid paths found in metadata 'processed_path' column")
                    
                    # Try using just the filename with the data_dir
                    filenames = [os.path.basename(path) for path in original_paths]
                    for filename in filenames:
                        path = os.path.join(data_dir, filename)
                        if os.path.exists(path):
                            image_files.append(path)
                    
                    # If still no valid paths, try looking in subdirectories
                    if len(image_files) == 0:
                        print("Searching for images in subdirectories...")
                        for root, dirs, files in os.walk(data_dir):
                            for file in files:
                                if file in filenames:
                                    image_files.append(os.path.join(root, file))
                else:
                    image_files = valid_paths
                
            elif 'path' in metadata.columns:
                # Try the 'path' column
                original_paths = metadata['path'].tolist()
                valid_paths = [path for path in original_paths if os.path.exists(path)]
                
                if len(valid_paths) == 0:
                    print("WARNING: No valid paths found in metadata 'path' column")
                    # Try with data_dir prepended
                    for path in original_paths:
                        full_path = os.path.join(data_dir, os.path.basename(path))
                        if os.path.exists(full_path):
                            image_files.append(full_path)
                else:
                    image_files = valid_paths
                
            elif 'filename' in metadata.columns:
                # Try constructing paths from filenames
                for filename in metadata['filename']:
                    # Try direct path
                    path = os.path.join(data_dir, filename)
                    if os.path.exists(path):
                        image_files.append(path)
                    else:
                        # Try subdirectories if we have symbol information
                        if 'symbol' in metadata.columns:
                            symbols = metadata['symbol'].unique()
                            for symbol in symbols:
                                symbol_dir = os.path.join(data_dir, symbol)
                                if os.path.exists(symbol_dir):
                                    symbol_path = os.path.join(symbol_dir, filename)
                                    if os.path.exists(symbol_path):
                                        image_files.append(symbol_path)
        else:
            # Get image files from directory
            print("No metadata provided, scanning directory for images...")
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        
        # Load images
        images = []
        print(f"Loading {len(image_files)} images...")
        
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                # Ensure image is in RGB mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img)
                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        print(f"Loaded {len(images)} images")
        
        # Handle the case where no images are found
        if not images:
            raise ValueError("No images could be loaded! Please check your data directory and metadata.")
        
        return images, metadata
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0.0, window=20):
        """Calculate rolling Sharpe ratio.
        
        Args:
            risk_free_rate (float): Risk-free rate
            window (int): Window for rolling calculation
            
        Returns:
            float: Sharpe ratio
        """
        if len(self.returns) < 2:
            return 0
        
        # Use the most recent window of returns
        recent_returns = self.returns[-window:] if len(self.returns) > window else self.returns
        
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if std_return == 0:
            return 0
        
        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe * np.sqrt(252)  # Annualized Sharpe ratio
    
    def render(self, mode='human'):
        """Render the environment.
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            # Display current chart image with portfolio value
            plt.figure(figsize=(12, 8))
            
            # Plot chart image
            plt.subplot(2, 1, 1)
            plt.imshow(self.chart_images[self.current_step])
            plt.title(f'Step: {self.current_step}, ' +
                      f'Balance: ${self.balance:.2f}, ' +
                      f'Position: {self.position:.4f}, ' +
                      f'Asset Value: ${self.position * self._get_current_price():.2f}, ' +
                      f'Total: ${(self.balance + self.position * self._get_current_price()):.2f}')
            
            # Plot portfolio value history
            plt.subplot(2, 1, 2)
            plt.plot(self.portfolio_values)
            plt.title('Portfolio Value History')
            plt.xlabel('Step')
            plt.ylabel('Value ($)')
            
            plt.tight_layout()
            plt.show()