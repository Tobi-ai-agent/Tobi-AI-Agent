#!/usr/bin/env python
# scripts/crypto_data_fetcher.py
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time
from PIL import Image
import ccxt
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoDataFetcher")

# Mapping of timeframe strings between different systems
TIMEFRAME_MAP = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '2h': '2h',
    '4h': '4h',
    '6h': '6h',
    '8h': '8h',
    '12h': '12h',
    '1d': '1d',
    '3d': '3d',
    '1w': '1w',
    '1M': '1M'
}

def fetch_historical_ohlcv(exchange_id, symbol, timeframe, start_date, end_date, rate_limit=True):
    """Fetch historical OHLCV data from an exchange.
    
    Args:
        exchange_id (str): Exchange ID (e.g., 'binance', 'coinbase')
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
        timeframe (str): Candle timeframe (e.g., '1h', '1d')
        start_date (str): Start date in ISO format (e.g., '2020-01-01T00:00:00Z')
        end_date (str): End date in ISO format
        rate_limit (bool): Whether to respect exchange rate limits
        
    Returns:
        pd.DataFrame: OHLCV data with columns: timestamp, open, high, low, close, volume
    """
    logger.info(f"Fetching data for {symbol} on {exchange_id} ({timeframe}) from {start_date} to {end_date}")
    
    try:
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': rate_limit,
        })
        
        # Convert dates to milliseconds timestamp
        since = exchange.parse8601(start_date)
        until = exchange.parse8601(end_date)
        
        # Some exchanges have limits on how many candles can be fetched at once
        # We'll fetch in chunks and combine
        all_candles = []
        current_since = since
        
        with tqdm(total=int((until - since) / (exchange.parse_timeframe(timeframe) * 1000))) as pbar:
            while current_since < until:
                try:
                    candles = exchange.fetch_ohlcv(symbol, timeframe, current_since)
                    if not candles:
                        break
                    
                    all_candles.extend(candles)
                    
                    # Update progress
                    num_candles = len(candles)
                    pbar.update(num_candles)
                    
                    # Set new since to the timestamp of the last candle + 1
                    current_since = candles[-1][0] + 1
                    
                    # Respect rate limits
                    if rate_limit:
                        time.sleep(exchange.rateLimit / 1000)  # Convert ms to seconds
                except Exception as e:
                    logger.error(f"Error fetching candles: {e}")
                    time.sleep(10)  # Wait before retrying
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        logger.info(f"Successfully fetched {len(df)} candles")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def generate_chart_image(df, output_path, symbol, timeframe, width=800, height=600, theme='dark'):
    """Generate a chart image from OHLCV data.
    
    Args:
        df (pd.DataFrame): OHLCV data
        output_path (str): Path to save the image
        symbol (str): Symbol name for the title
        timeframe (str): Timeframe for the title
        width (int): Image width
        height (int): Image height
        theme (str): Chart theme ('dark' or 'light')
        
    Returns:
        str: Path to the saved image
    """
    try:
        # Set theme
        if theme == 'dark':
            plt.style.use('dark_background')
            candle_up_color = 'green'
            candle_down_color = 'red'
            volume_up_color = 'green'
            volume_down_color = 'red'
            grid_color = '#333333'
        else:
            plt.style.use('default')
            candle_up_color = 'green'
            candle_down_color = 'red'
            volume_up_color = 'green'
            volume_down_color = 'red'
            grid_color = '#dddddd'
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width/100, height/100), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot candlesticks
        df.reset_index(inplace=True)
        
        # Get date format based on timeframe
        if timeframe in ['1m', '5m', '15m', '30m']:
            date_format = '%H:%M'
        elif timeframe in ['1h', '2h', '4h', '6h', '8h', '12h']:
            date_format = '%m-%d %H:%M'
        else:
            date_format = '%Y-%m-%d'
        
        # Plot candlesticks
        width = 0.6
        width2 = width * 0.8
        
        up = df[df.close >= df.open]
        down = df[df.close < df.open]
        
        # Candlestick bodies
        ax1.bar(up.index, up.close - up.open, width, bottom=up.open, color=candle_up_color)
        ax1.bar(down.index, down.close - down.open, width, bottom=down.open, color=candle_down_color)
        
        # Candlestick wicks
        ax1.bar(up.index, up.high - up.close, width2, bottom=up.close, color=candle_up_color)
        ax1.bar(up.index, up.low - up.open, width2, bottom=up.open, color=candle_up_color)
        ax1.bar(down.index, down.high - down.open, width2, bottom=down.open, color=candle_down_color)
        ax1.bar(down.index, down.low - down.close, width2, bottom=down.close, color=candle_down_color)
        
        # Format date
        if len(df) > 0:
            price = df.iloc[-1].close
        else:
            price = 0
        
        # Plot volume
        ax2.bar(up.index, up.volume, width, color=volume_up_color, alpha=0.8)
        ax2.bar(down.index, down.volume, width, color=volume_down_color, alpha=0.8)
        
        # Format axes
        ax1.set_title(f"{symbol} ({timeframe}) - Price: ${price:.2f}")
        ax1.set_ylabel('Price')
        ax1.grid(True, color=grid_color, linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax2.set_ylabel('Volume')
        ax2.grid(True, color=grid_color, linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Format x-axis labels
        plt.xticks(rotation=45)
        
        # Ensure x-axis labels are readable
        max_ticks = 15
        step = max(1, len(df) // max_ticks)
        
        ax1.set_xticks(range(0, len(df), step))
        ax1.set_xticklabels(df['timestamp'].iloc[::step].dt.strftime(date_format))
        
        ax2.set_xticks(range(0, len(df), step))
        ax2.set_xticklabels(df['timestamp'].iloc[::step].dt.strftime(date_format))
        
        # Layout
        plt.tight_layout()
        
        # Save image
        plt.savefig(output_path, dpi=100)
        plt.close(fig)
        
        # Create filename with price info
        date_str = df.iloc[-1]['timestamp'].strftime('%Y%m%d')
        time_str = df.iloc[-1]['timestamp'].strftime('%H%M%S')
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating chart image: {e}")
        return None

def fetch_and_generate_charts(exchange_id, symbols, timeframes, start_date, end_date, output_dir, 
                             image_width=800, image_height=600, theme='dark'):
    """Fetch data and generate chart images for multiple symbols and timeframes.
    
    Args:
        exchange_id (str): Exchange ID
        symbols (list): List of symbols
        timeframes (list): List of timeframes
        start_date (str): Start date
        end_date (str): End date
        output_dir (str): Output directory
        image_width (int): Image width
        image_height (int): Image height
        theme (str): Chart theme
        
    Returns:
        dict: Metadata for generated images
    """
    # Create output directories
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Initialize metadata
    metadata = []
    
    # Process each symbol and timeframe
    for symbol in symbols:
        for timeframe in timeframes:
            # Create symbol-specific directory
            symbol_dir = os.path.join(raw_dir, symbol.replace('/', '_'))
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Fetch data
            df = fetch_historical_ohlcv(exchange_id, symbol, timeframe, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No data found for {symbol} ({timeframe})")
                continue
            
            # Save raw data
            csv_path = os.path.join(symbol_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")
            df.to_csv(csv_path)
            logger.info(f"Saved raw data to {csv_path}")
            
            # Generate chart images at various intervals
            logger.info(f"Generating chart images for {symbol} ({timeframe})")
            
            # We'll generate images for different segments of the data
            # to get a good variety of market conditions
            num_candles = len(df)
            window_size = min(100, num_candles)  # Number of candles to include in each chart
            
            # Generate approximately 100 images per symbol/timeframe
            if num_candles <= window_size:
                # If we have less data than window_size, just generate one image
                step = 1
            else:
                step = max(1, (num_candles - window_size) // 100)
            
            for i in range(0, num_candles - window_size, step):
                # Get window of data
                window_df = df.iloc[i:i+window_size].copy()
                
                # Generate chart image
                timestamp = window_df.index[-1].strftime('%Y%m%d_%H%M%S')
                price = window_df['close'].iloc[-1]
                clean_symbol = symbol.replace('/', '')
                
                # Create filename
                filename = f"{clean_symbol}_{timeframe}_{timestamp}_price_{price:.2f}.png"
                image_path = os.path.join(symbol_dir, filename)
                
                # Generate and save chart
                generate_chart_image(
                    window_df.reset_index(), 
                    image_path, 
                    symbol, 
                    timeframe,
                    width=image_width,
                    height=image_height,
                    theme=theme
                )
                
                # Add to metadata
                metadata.append({
                    'filename': filename,
                    'path': image_path,
                    'symbol': clean_symbol,
                    'timeframe': timeframe,
                    'date': timestamp.split('_')[0],
                    'time': timestamp.split('_')[1],
                    'price': price,
                    'exchange': exchange_id
                })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(output_dir, "raw", "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata with {len(metadata_df)} entries to {metadata_path}")
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Fetch historical crypto data and generate chart images')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange ID (e.g., binance, coinbase)')
    parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'TRX/USDT'], 
                        help='Symbols to fetch (e.g., BTC/USDT ETH/USDT)')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['1h', '4h', '1d'], 
                        help='Timeframes to fetch (e.g., 1h 4h 1d)')
    parser.add_argument('--days', type=int, default=365, 
                        help='Number of days of historical data to fetch')
    parser.add_argument('--output_dir', type=str, default='data', 
                        help='Output directory')
    parser.add_argument('--width', type=int, default=800, 
                        help='Image width')
    parser.add_argument('--height', type=int, default=600, 
                        help='Image height')
    parser.add_argument('--theme', type=str, default='dark', choices=['dark', 'light'], 
                        help='Chart theme')
    
    args = parser.parse_args()
    
    # Calculate date range
    end_date = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    start_date = (datetime.utcnow() - timedelta(days=args.days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Fetch and generate charts
    fetch_and_generate_charts(
        args.exchange,
        args.symbols,
        args.timeframes,
        start_date,
        end_date,
        args.output_dir,
        image_width=args.width,
        image_height=args.height,
        theme=args.theme
    )
    
    logger.info("Done!")

if __name__ == "__main__":
    main()