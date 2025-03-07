# src/train.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import time
import pandas as pd
from datetime import datetime

from src.models.agent import DQNAgent
from src.environment.trading_env import TradingEnvironment
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.hyperparameters import *
from config.trading_config import *


def train(config, trading_config, data_dir, metadata_path, model_dir, log_dir, 
          symbols=None, num_episodes=1000, render_interval=100, save_interval=50, device=None):
    """Train the DQN agent.
    
    Args:
        config (dict): Hyperparameters configuration
        trading_config (dict): Trading environment configuration
        data_dir (str): Directory containing chart images
        metadata_path (str): Path to metadata CSV (optional)
        model_dir (str): Directory to save trained models
        log_dir (str): Directory to save training logs
        symbols (list): List of symbols to filter on
        num_episodes (int): Number of training episodes
        render_interval (int): How often to render (in episodes)
        save_interval (int): How often to save model (in episodes)
        device (torch.device): Device to use for training
        
    Returns:
        tuple: (agent, training history)
    """
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Filter metadata by symbols if specified
    filtered_metadata_path = None
 # Replace this section in the train function:
    if symbols and (metadata_path or os.path.exists(os.path.join(data_dir, 'metadata.csv'))):
        # Use provided metadata path or default
        metadata_file = metadata_path or os.path.join(data_dir, 'metadata.csv')
        
        # Read metadata
        metadata_df = pd.read_csv(metadata_file)
        
        # Filter by symbols
        print(f"\nFiltering data to include only symbols: {symbols}")
        original_count = len(metadata_df)
        metadata_df = metadata_df[metadata_df['symbol'].isin(symbols)]
        filtered_count = len(metadata_df)
        print(f"Filtered from {original_count} to {filtered_count} images")
        
        # Check if filtered paths exist
        path_column = 'processed_path' if 'processed_path' in metadata_df.columns else 'path'
        if path_column in metadata_df.columns:
            valid_paths = []
            for path in metadata_df[path_column]:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    # Try an alternative path construction
                    filename = os.path.basename(path)
                    alt_path = os.path.join(data_dir, filename)
                    if os.path.exists(alt_path):
                        # Update the path in metadata
                        metadata_df.loc[metadata_df[path_column] == path, path_column] = alt_path
                        valid_paths.append(alt_path)
            
            print(f"Found {len(valid_paths)} valid image paths out of {filtered_count}")
            if len(valid_paths) == 0:
                print("No valid image paths found! Training on all data instead.")
                # Fall back to using all data
                metadata_path = metadata_file
            else:
                # Keep only rows with valid paths
                metadata_df = metadata_df[metadata_df[path_column].isin(valid_paths)]
                # Save filtered metadata
                filtered_metadata_path = os.path.join(data_dir, 'filtered_metadata.csv')
                metadata_df.to_csv(filtered_metadata_path, index=False)
                print(f"Saved filtered metadata with {len(metadata_df)} valid entries to {filtered_metadata_path}")
                metadata_path = filtered_metadata_path
    
    # Initialize environment
    env = TradingEnvironment(
        data_dir=data_dir,
        metadata_path=metadata_path,
        lookback_window=trading_config['LOOKBACK_WINDOW'],
        initial_balance=trading_config['INITIAL_BALANCE'],
        commission=trading_config['TRADING_FEE'],
        use_sharpe=trading_config['USE_SHARPE_RATIO']
    )
    
    # Initialize agent
    sample_obs, _ = env.reset()
    state_size = (config['INPUT_CHANNELS'], *config['IMAGE_SIZE'])
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, config, device)
    
    # Initialize tracking variables
    scores = []
    win_rates = []
    epsilon = config['EPSILON_START']
    epsilon_history = []
    best_score = -np.inf
    start_time = time.time()
    
    # Training loop
    print(f"Starting training for {num_episodes} episodes...")
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        score = 0
        done = False
        
        # Episode loop
        while not done:
            # Get action from agent
            action = agent.act(state, epsilon)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Add experience to agent's memory and learn
            agent.step(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            if truncated:
                break
            
            # Render if needed
            if episode % render_interval == 0 and episode > num_episodes - 5:
                env.render()
        
        # Decay epsilon
        epsilon = max(config['EPSILON_END'], config['EPSILON_DECAY'] * epsilon)
        
        # Update tracking variables
        scores.append(score)
        epsilon_history.append(epsilon)
        
        # Track win rate if available
        if 'win_rate' in info:
            win_rates.append(info['win_rate'])
        
        # Print progress
        elapsed_time = time.time() - start_time
        print(f"Episode {episode}/{num_episodes} | " +
              f"Score: {score:.2f} | " +
              f"Epsilon: {epsilon:.4f} | " +
              f"Time: {elapsed_time:.1f}s", end="")
        
        if 'profit_pct' in info:
            print(f" | Profit: {info['profit_pct']:.2f}%", end="")
        
        if 'win_rate' in info:
            print(f" | Win Rate: {info['win_rate']*100:.1f}%")
        else:
            print()
        
        # Save model if it's the best so far
        if score > best_score:
            best_score = score
            agent.save(os.path.join(model_dir, 'dqn_best.pth'))
            print(f"New best model saved with score: {best_score:.2f}")
        
        # Save model periodically
        if episode % save_interval == 0:
            agent.save(os.path.join(model_dir, f'dqn_episode_{episode}.pth'))
            
            # Plot training progress
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot scores with moving average
            ax1.plot(scores)
            if len(scores) > 10:
                moving_avg = np.convolve(scores, np.ones(10)/10, mode='valid')
                ax1.plot(range(9, len(scores)), moving_avg, 'r-')
            ax1.set_title('Episode Scores')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Score')
            
            # Plot epsilon
            ax2.plot(epsilon_history)
            ax2.set_title('Epsilon Decay')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Epsilon')
            
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f'training_progress_episode_{episode}.png'))
            plt.close()
            
            # Also plot win rate if available
            if win_rates:
                plt.figure(figsize=(10, 6))
                plt.plot(win_rates)
                plt.title('Win Rate Progression')
                plt.xlabel('Episode with Win Rate Data')
                plt.ylabel('Win Rate')
                plt.ylim(0, 1)
                plt.savefig(os.path.join(log_dir, f'win_rate_episode_{episode}.png'))
                plt.close()
    
    # Save final model
    agent.save(os.path.join(model_dir, 'dqn_final.pth'))
    
    # Calculate final statistics
    avg_score = np.mean(scores[-100:])
    avg_win_rate = np.mean(win_rates[-100:]) if win_rates else 0
    
    # Save training history
    history = {
        'scores': scores,
        'epsilon_history': epsilon_history,
        'win_rates': win_rates,
        'avg_score': float(avg_score),
        'avg_win_rate': float(avg_win_rate),
        'training_time': time.time() - start_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'symbols': symbols
    }
    
    with open(os.path.join(log_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\nTraining completed!")
    print(f"Average score (last 100 episodes): {avg_score:.2f}")
    print(f"Average win rate (last 100 episodes): {avg_win_rate*100:.1f}%")
    print(f"Total training time: {history['training_time']:.1f} seconds")
    
    return agent, history


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train DQN agent for trading')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing chart images')
    parser.add_argument('--metadata', type=str, help='Path to metadata CSV (optional)')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to train on (e.g., BTC_USD,ETH_USD)')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--render_interval', type=int, default=100, help='How often to render (in episodes)')
    parser.add_argument('--save_interval', type=int, default=50, help='How often to save model (in episodes)')
    parser.add_argument('--cpu', action='store_true', help='Force using CPU even if GPU is available')
    args = parser.parse_args()
    
    # Create dictionaries from configuration modules
    hyper_config = {k: v for k, v in globals().items() 
                   if k.isupper() and not k.startswith('__') 
                   and k not in ['LOOKBACK_WINDOW', 'TRADING_FEE', 'INITIAL_BALANCE', 
                                'ACTION_SPACE', 'USE_SHARPE_RATIO', 'RISK_FREE_RATE']}
    
    trade_config = {k: v for k, v in globals().items() 
                   if k.isupper() and not k.startswith('__') 
                   and k in ['LOOKBACK_WINDOW', 'TRADING_FEE', 'INITIAL_BALANCE', 
                            'ACTION_SPACE', 'USE_SHARPE_RATIO', 'RISK_FREE_RATE']}
    
    # Set device
    device = None
    if args.cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
    
    # Process symbols if provided
    symbols = None
    if args.symbols:
        symbols = args.symbols.split(',')
    
    # Print configuration
    print("\nTraining with the following configuration:")
    print("\nHyperparameters:")
    for key, value in hyper_config.items():
        print(f"  {key}: {value}")
    
    print("\nTrading Environment Settings:")
    for key, value in trade_config.items():
        print(f"  {key}: {value}")
    
    if symbols:
        print("\nFiltering on symbols:", symbols)
    
    # Train agent
    agent, history = train(
        hyper_config,
        trade_config,
        args.data_dir,
        args.metadata,
        args.model_dir,
        args.log_dir,
        symbols,
        args.episodes,
        args.render_interval,
        args.save_interval,
        device
    )


if __name__ == "__main__":
    main()