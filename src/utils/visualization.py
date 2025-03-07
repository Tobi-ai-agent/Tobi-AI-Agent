# src/utils/visualization.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import torch


def plot_training_history(history, save_path=None):
    """Plot training history.
    
    Args:
        history (dict): Training history dictionary
        save_path (str): Path to save the plot (optional)
    """
    plt.figure(figsize=(15, 10))
    
    # Plot scores
    plt.subplot(2, 2, 1)
    plt.plot(history['scores'])
    if len(history['scores']) > 20:
        window_size = min(20, len(history['scores']) // 5)
        moving_avg = np.convolve(history['scores'], np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(history['scores'])), moving_avg, 'r-')
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    # Plot epsilon
    plt.subplot(2, 2, 2)
    plt.plot(history['epsilon_history'])
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    # Plot win rate if available
    if 'win_rates' in history and history['win_rates']:
        plt.subplot(2, 2, 3)
        plt.plot(history['win_rates'])
        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.axhline(y=0.95, color='g', linestyle='--')
        plt.title('Win Rate')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
        plt.grid(True)
    
    # Plot loss if available
    if 'losses' in history and history['losses']:
        plt.subplot(2, 2, 4)
        plt.plot(history['losses'])
        plt.title('Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_trade_performance(trade_history, portfolio_values, initial_balance,
                          metadata=None, save_path=None):
    """Plot trading performance.
    
    Args:
        trade_history (list): List of trade tuples (action, price, amount, step)
        portfolio_values (list): List of portfolio values over time
        initial_balance (float): Initial account balance
        metadata (pd.DataFrame): Metadata dataframe (optional)
        save_path (str): Path to save the plot (optional)
    """
    plt.figure(figsize=(15, 12))
    
    # Extract buy and sell points
    buy_points = [(t[3], t[1]) for t in trade_history if t[0] == 'buy']
    sell_points = [(t[3], t[1]) for t in trade_history if t[0] == 'sell']
    
    # Plot portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.axhline(y=initial_balance, color='r', linestyle='--', label='Initial Balance')
    
    # Add buy and sell markers if available
    if buy_points:
        buy_indices, buy_values = zip(*[(i, portfolio_values[i]) for i, _ in buy_points if i < len(portfolio_values)])
        plt.scatter(buy_indices, buy_values, marker='^', color='g', label='Buy')
    
    if sell_points:
        sell_indices, sell_values = zip(*[(i, portfolio_values[i]) for i, _ in sell_points if i < len(portfolio_values)])
        plt.scatter(sell_indices, sell_values, marker='v', color='r', label='Sell')
    
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Step')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Calculate returns
    if len(portfolio_values) > 1:
        returns = [(portfolio_values[i] / portfolio_values[i-1]) - 1 for i in range(1, len(portfolio_values))]
        
        # Plot returns
        plt.subplot(2, 2, 3)
        plt.plot(returns)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Returns')
        plt.xlabel('Step')
        plt.ylabel('Return (%)')
        plt.grid(True)
        
        # Plot return distribution
        plt.subplot(2, 2, 4)
        sns.histplot(returns, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Return Distribution')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_q_values(model, state, device):
    """Visualize Q-values for a given state.
    
    Args:
        model (nn.Module): Q-network model
        state (numpy.array): State to evaluate
        device (torch.device): Device to use for computation
    """
    # Convert state to tensor
    if isinstance(state, np.ndarray):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    else:
        state_tensor = state.unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get Q-values
    with torch.no_grad():
        q_values = model(state_tensor).cpu().numpy()[0]
    
    # Plot Q-values
    plt.figure(figsize=(10, 6))
    actions = ['Hold', 'Buy', 'Sell']
    colors = ['blue', 'green', 'red']
    
    plt.bar(actions, q_values, color=colors)
    plt.title('Q-values for Current State')
    plt.xlabel('Action')
    plt.ylabel('Q-value')
    plt.grid(True, axis='y')
    
    # Add value labels
    for i, v in enumerate(q_values):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    
    plt.show()


def generate_performance_report(trade_history, portfolio_values, initial_balance,
                               final_balance, save_path=None):
    """Generate a performance report.
    
    Args:
        trade_history (list): List of trade tuples (action, price, amount, step)
        portfolio_values (list): List of portfolio values over time
        initial_balance (float): Initial account balance
        final_balance (float): Final account balance
        save_path (str): Path to save the report (optional)
    
    Returns:
        dict: Performance metrics
    """
    # Calculate performance metrics
    total_trades = len(trade_history)
    buy_trades = len([t for t in trade_history if t[0] == 'buy'])
    sell_trades = len([t for t in trade_history if t[0] == 'sell'])
    
    profit = final_balance - initial_balance
    profit_pct = (profit / initial_balance) * 100
    
    # Calculate returns and volatility
    returns = []
    if len(portfolio_values) > 1:
        returns = [(portfolio_values[i] / portfolio_values[i-1]) - 1 for i in range(1, len(portfolio_values))]
        
    volatility = np.std(returns) * 100 if returns else 0
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
    
    # Calculate drawdown
    max_drawdown = 0
    if portfolio_values:
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
    
    max_drawdown_pct = max_drawdown * 100
    
    # Assemble metrics
    metrics = {
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'profit': profit,
        'profit_pct': profit_pct,
        'total_trades': total_trades,
        'buy_trades': buy_trades,
        'sell_trades': sell_trades,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown_pct
    }
    
    # Generate report
    report = "\n===== Trading Performance Report =====\n\n"
    report += f"Initial Balance: ${initial_balance:.2f}\n"
    report += f"Final Balance: ${final_balance:.2f}\n"
    report += f"Profit/Loss: ${profit:.2f} ({profit_pct:.2f}%)\n\n"
    
    report += f"Total Trades: {total_trades}\n"
    report += f"  - Buy Trades: {buy_trades}\n"
    report += f"  - Sell Trades: {sell_trades}\n\n"
    
    report += f"Performance Metrics:\n"
    report += f"  - Volatility: {volatility:.2f}%\n"
    report += f"  - Sharpe Ratio: {sharpe:.2f}\n"
    report += f"  - Maximum Drawdown: {max_drawdown_pct:.2f}%\n"
    
    # Print or save report
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    else:
        print(report)
    
    return metrics