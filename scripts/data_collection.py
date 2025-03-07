# scripts/data_collection.py
import os
import time
import argparse
import datetime
import json
import pandas as pd
from PIL import Image


def save_screenshot_with_metadata(image, output_dir, symbol, timeframe, price=None, timestamp=None):
    """Save screenshot with metadata encoded in the filename.
    
    Args:
        image (PIL.Image): Screenshot image
        output_dir (str): Directory to save the screenshot
        symbol (str): Trading symbol (e.g., 'BTCUSD')
        timeframe (str): Chart timeframe (e.g., '15m', '1h', '4h')
        price (float, optional): Current price at the time of screenshot
        timestamp (datetime, optional): Timestamp of the screenshot
    
    Returns:
        str: Path to the saved image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use current time if timestamp not provided
    if timestamp is None:
        timestamp = datetime.datetime.now()
    
    # Format timestamp
    date_str = timestamp.strftime('%Y%m%d')
    time_str = timestamp.strftime('%H%M%S')
    
    # Create filename
    if price is not None:
        filename = f"{symbol}_{timeframe}_{date_str}_{time_str}_price_{price:.2f}.png"
    else:
        filename = f"{symbol}_{timeframe}_{date_str}_{time_str}.png"
    
    # Save image
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)
    
    print(f"Saved screenshot to {output_path}")
    return output_path


def print_instructions():
    """Print instructions for manual data collection."""
    print("\n===== TradingView Chart Data Collection Instructions =====")
    print("\n1. Open TradingView in your browser")
    print("2. Navigate to the chart you want to capture")
    print("3. For each chart:")
    print("   a. Take a screenshot (Command+Shift+4 on Mac)")
    print("   b. Save the screenshot to your collection directory")
    print("   c. Use a consistent naming format:")
    print("      SYMBOL_TIMEFRAME_YYYYMMDD_HHMMSS_price_XXXX.XX.png")
    print("      Example: BTCUSD_1h_20240301_143000_price_63245.50.png")
    print("\n4. Make sure to:")
    print("   - Capture a consistent chart area")
    print("   - Include price information in the chart")
    print("   - Maintain the same chart settings (indicators, colors)")
    print("   - Collect a diverse set of market conditions")
    print("\n5. For automated collection with a browser extension:")
    print("   - Use a screenshot extension with scheduling")
    print("   - Set up to capture at regular intervals")
    print("   - Export price data alongside images if possible")
    print("\nRemember: Higher quality and consistency in your dataset")
    print("will lead to better model performance.")
    print("\n========================================================")


def create_sample_dataset(output_dir, symbols=None, timeframes=None, 
                        num_samples=100, start_date=None, price_range=None):
    """Create a sample dataset for testing.
    
    This function creates synthetic chart images with metadata for testing purposes.
    
    Args:
        output_dir (str): Directory to save sample images
        symbols (list, optional): List of symbols to use
        timeframes (list, optional): List of timeframes to use
        num_samples (int): Number of samples to generate
        start_date (datetime, optional): Starting date for generated data
        price_range (tuple, optional): Price range (min, max)
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    # Set defaults if not provided
    if symbols is None:
        symbols = ['BTCUSD', 'ETHUSD', 'EURUSD', 'AAPL', 'MSFT']
    
    if timeframes is None:
        timeframes = ['1m', '5m', '15m', '1h', '4h', 'D']
    
    if start_date is None:
        start_date = datetime.datetime.now() - datetime.timedelta(days=30)
    
    if price_range is None:
        price_range = (100, 1000)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track metadata
    metadata = []
    
    print(f"Generating {num_samples} sample chart images...")
    for i in range(num_samples):
        # Randomly select symbol and timeframe
        symbol = random.choice(symbols)
        timeframe = random.choice(timeframes)
        
        # Generate random date and time
        days_offset = random.randint(0, 30)
        hours_offset = random.randint(0, 23)
        minutes_offset = random.randint(0, 59)
        current_date = start_date + datetime.timedelta(
            days=days_offset, hours=hours_offset, minutes=minutes_offset
        )
        
        # Generate random price
        price = random.uniform(price_range[0], price_range[1])
        
        # Create a simple chart image (black background with price line)
        img = Image.new('RGB', (800, 600), color='black')
        draw = ImageDraw.Draw(img)
        
        # Draw a random chart line
        points = []
        x_step = 800 / 100
        x = 0
        last_y = random.randint(100, 500)
        
        for j in range(100):
            y_change = random.randint(-20, 20)
            new_y = max(50, min(550, last_y + y_change))
            points.append((int(x), int(new_y)))
            last_y = new_y
            x += x_step
        
        # Draw the line
        if len(points) >= 2:
            draw.line(points, fill='green' if points[0][1] > points[-1][1] else 'red', width=2)
        
        # Add text with symbol, timeframe, and price
        try:
            draw.text((20, 20), f"{symbol} ({timeframe})", fill="white")
            draw.text((20, 50), f"Price: ${price:.2f}", fill="white")
            draw.text((20, 80), current_date.strftime("%Y-%m-%d %H:%M:%S"), fill="white")
        except Exception as e:
            print(f"Warning: Couldn't add text to image: {e}")
        
        # Save image with metadata
        date_str = current_date.strftime('%Y%m%d')
        time_str = current_date.strftime('%H%M%S')
        filename = f"{symbol}_{timeframe}_{date_str}_{time_str}_price_{price:.2f}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        
        # Add to metadata
        metadata.append({
            'filename': filename,
            'path': filepath,
            'symbol': symbol,
            'timeframe': timeframe,
            'date': date_str,
            'time': time_str,
            'price': price
        })
        
        if i % 10 == 0:
            print(f"Generated {i+1}/{num_samples} images...")
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"Done! Generated {num_samples} sample images and saved metadata to {metadata_path}")


def main():
    """Main function to run the data collection script."""
    parser = argparse.ArgumentParser(description='Trading chart data collection helper')
    parser.add_argument('--mode', type=str, choices=['instructions', 'sample'], default='instructions',
                      help='Mode to run (instructions or sample data generation)')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                      help='Output directory for collected or sample data')
    parser.add_argument('--samples', type=int, default=100,
                      help='Number of sample images to generate in sample mode')
    args = parser.parse_args()
    
    if args.mode == 'instructions':
        print_instructions()
    elif args.mode == 'sample':
        create_sample_dataset(args.output_dir, num_samples=args.samples)
    

if __name__ == "__main__":
    main()