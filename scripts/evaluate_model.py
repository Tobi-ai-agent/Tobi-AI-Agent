# scripts/evaluate_model.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.agent import DQNAgent
from src.environment.trading_env import TradingEnvironment
from config.hyperparameters import *
from config.trading_config import *


def evaluate(agent, env, num_episodes=10, render=True, render_interval=5):
    """Evaluate a trained agent.
    
    Args:
        agent (DQNAgent): Trained agent
        env (TradingEnvironment): Trading environment
        num_episodes (int): Number of evaluation episodes
        render (bool): Whether to render the environment
        render_interval (int): How often to render (in steps)
        
    Returns:
        dict: Evaluation metrics
    """
    # Metrics to track
    total_rewards = []
    profits = []
    trade_counts = []
    win_rates = []
    portfolio_histories = []
    
    # Evaluation loop
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        # Store portfolio value history for this episode
        portfolio_history = []
        
        # Episode loop
        while not done:
            # Get action from agent (no exploration)
            action = agent.act(state, epsilon=0.0)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Track reward
            total_reward += reward
            
            # Store portfolio value (balance + position value)
            current_price = env._get_current_price()
            portfolio_value = env.balance + (env.position * current_price)
            portfolio_history.append(portfolio_value)
            
            # Update state and step count
            state = next_state
            step_count += 1
            
            # Render if requested
            if render and step_count % render_interval == 0:
                env.render()
            
            if truncated:
                break
        
        # Store portfolio history for this episode
        portfolio_histories.append(portfolio_history)
        
        # Store metrics
        total_rewards.append(total_reward)
        
        # Calculate profit percentage
        if 'profit_pct' in info:
            profits.append(info['profit_pct'])
        
        # Count trades
        if 'total_trades' in info:
            trade_counts.append(info['total_trades'])
        else:
            trade_counts.append(len(env.trade_history))
        
        # Store win rate if available
        if 'win_rate' in info:
            win_rates.append(info['win_rate'])
        
        # Print episode summary
        print(f"Episode {episode}/{num_episodes} | " +
              f"Total Reward: {total_reward:.2f} | " +
              f"Trades: {trade_counts[-1]}")
        
        if 'profit_pct' in info:
            print(f"  Profit: {info['profit_pct']:.2f}%")
        
        if 'win_rate' in info:
            print(f"  Win Rate: {info['win_rate']*100:.2f}%")
    
    # Calculate overall metrics
    avg_reward = np.mean(total_rewards) if total_rewards else 0
    avg_profit = np.mean(profits) if profits else 0
    avg_trades = np.mean(trade_counts) if trade_counts else 0
    avg_win_rate = np.mean(win_rates) if win_rates else 0
    
    evaluation_metrics = {
        'avg_reward': float(avg_reward),
        'avg_profit': float(avg_profit),
        'avg_trades': float(avg_trades),
        'avg_win_rate': float(avg_win_rate),
        'episode_rewards': total_rewards,
        'episode_profits': profits,
        'episode_trade_counts': trade_counts,
        'episode_win_rates': win_rates
    }
    
    return evaluation_metrics, portfolio_histories


def main():
    """Main function to evaluate a trained agent."""
    parser = argparse.ArgumentParser(description='Evaluate DQN agent for trading')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing chart images')
    parser.add_argument('--metadata', type=str, help='Path to metadata CSV')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='evaluation', help='Directory to save evaluation results')
    parser.add_argument('--episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--no_render', action='store_false', dest='render', help='Disable rendering')
    parser.add_argument('--render_interval', type=int, default=10, help='How often to render (in steps)')
    parser.add_argument('--cpu', action='store_true', help='Force using CPU even if GPU is available')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Initialize environment
    env = TradingEnvironment(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        lookback_window=trade_config['LOOKBACK_WINDOW'],
        initial_balance=trade_config['INITIAL_BALANCE'],
        commission=trade_config['TRADING_FEE'],
        use_sharpe=trade_config['USE_SHARPE_RATIO']
    )
    
    # Initialize agent
    sample_obs, _ = env.reset()
    state_size = (hyper_config['INPUT_CHANNELS'], *hyper_config['IMAGE_SIZE'])
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, hyper_config, device)
    
    # Load trained model
    agent.load(args.model_path)
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate agent
    print(f"Evaluating agent for {args.episodes} episodes...")
    metrics, portfolio_histories = evaluate(
        agent, env, args.episodes, args.render, args.render_interval
    )
    
    # Print overall metrics
    print("\nEvaluation Results:")
    print(f"Average Reward: {metrics['avg_reward']:.2f}")
    print(f"Average Profit: {metrics['avg_profit']:.2f}%")
    print(f"Average Trades per Episode: {metrics['avg_trades']:.2f}")
    print(f"Average Win Rate: {metrics['avg_win_rate'] * 100:.2f}%")
    
    # Calculate average win rate and profit to see if we're approaching the 95% target
    if metrics['avg_win_rate'] > 0:
        print(f"\nCurrent win rate: {metrics['avg_win_rate'] * 100:.2f}% (Target: 95%)")
        print(f"Progress towards target: {(metrics['avg_win_rate'] * 100) / 95:.2f}%")
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot results
    # 1. Profit by episode
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(metrics['episode_profits']) + 1), metrics['episode_profits'])
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Profit (%)')
    plt.title('Profit by Episode')
    plt.savefig(os.path.join(args.output_dir, 'profits_by_episode.png'))
    plt.close()
    
    # 2. Win rate by episode
    if metrics['episode_win_rates']:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(metrics['episode_win_rates']) + 1), 
                [rate * 100 for rate in metrics['episode_win_rates']])
        plt.axhline(y=50, color='orange', linestyle='--', label='50% Win Rate')
        plt.axhline(y=95, color='r', linestyle='--', label='Target: 95% Win Rate')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate by Episode')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'win_rates_by_episode.png'))
        plt.close()
    
    # 3. Portfolio value over time (for each episode)
    plt.figure(figsize=(12, 8))
    for i, history in enumerate(portfolio_histories):
        plt.plot(history, label=f'Episode {i+1}')
    plt.axhline(y=trade_config['INITIAL_BALANCE'], color='r', linestyle='--', 
                label='Initial Balance')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'portfolio_values.png'))
    plt.close()
    
    print(f"Saved evaluation results to {args.output_dir}")


if __name__ == "__main__":
    main()