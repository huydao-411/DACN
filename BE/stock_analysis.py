"""
Financial Data Analysis Script for Apple (AAPL) Stock
======================================================
Script này phân tích dữ liệu cổ phiếu Apple trong 5 năm
với các chỉ báo kỹ thuật: MA, RSI, MACD, Bollinger Bands
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Không display, chỉ lưu file
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def load_stock_data(ticker, period_years=5):
    """
    Tải dữ liệu lịch sử cổ phiếu từ yfinance
    
    Args:
        ticker (str): Mã cổ phiếu (VD: AAPL)
        period_years (int): Số năm dữ liệu cần tải
    
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu OHLCV
    """
    try:
        print(f"[INFO] Đang tải dữ liệu cho {ticker}...")
        
        # Tính toán ngày bắt đầu
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*period_years)
        
        # Tải dữ liệu từ yfinance
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='1d',
            progress=False
        )
        
        # Flatten multi-level columns nếu có
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        print(f"[SUCCESS] Tải thành công {len(data)} bản ghi dữ liệu")
        return data
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi tải dữ liệu: {str(e)}")
        return None


def handle_missing_data(df):
    """
    Xử lý dữ liệu bị thiếu (missing values)
    - Forward fill cho giá (Open, High, Low, Close)
    - 0 fill cho Volume
    
    Args:
        df (pd.DataFrame): DataFrame gốc
    
    Returns:
        pd.DataFrame: DataFrame sau khi xử lý
    """
    try:
        print("\n[INFO] Kiểm tra missing values...")
        
        # Hiển thị missing values ban đầu
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print("Missing values trước xử lý:")
            print(missing_counts[missing_counts > 0])
        else:
            print("Không có missing values")
        
        # Xử lý missing data
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')  # Forward fill
        
        # Volume = 0 fill
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
        
        # Xóa các hàng còn lại có NaN
        df = df.dropna()
        
        print(f"[SUCCESS] Xử lý xong, còn {len(df)} bản ghi hợp lệ")
        return df
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi xử lý missing data: {str(e)}")
        return df


def calculate_moving_averages(df):
    """
    Tính Simple Moving Average (MA)
    - MA 10, MA 20, MA 50
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu giá
    
    Returns:
        pd.DataFrame: DataFrame với thêm cột MA
    """
    try:
        print("[INFO] Tính Moving Averages...")
        
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        print("[SUCCESS] Tính MA thành công")
        return df
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi tính MA: {str(e)}")
        return df


def calculate_rsi(df, period=14):
    """
    Tính Relative Strength Index (RSI)
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu giá
        period (int): Chu kỳ RSI (mặc định 14)
    
    Returns:
        pd.DataFrame: DataFrame với thêm cột RSI
    """
    try:
        print("[INFO] Tính RSI...")
        
        # Tính sự thay đổi giá
        delta = df['Close'].diff()
        
        # Tách gain và loss
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Tính RS và RSI
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        print("[SUCCESS] Tính RSI thành công")
        return df
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi tính RSI: {str(e)}")
        return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Tính MACD (Moving Average Convergence Divergence)
    - MACD line, Signal line, Histogram
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu giá
        fast (int): Chu kỳ EMA nhanh (mặc định 12)
        slow (int): Chu kỳ EMA chậm (mặc định 26)
        signal (int): Chu kỳ Signal line (mặc định 9)
    
    Returns:
        pd.DataFrame: DataFrame với thêm cột MACD
    """
    try:
        print("[INFO] Tính MACD...")
        
        # Tính EMA
        ema_fast = df['Close'].ewm(span=fast).mean()
        ema_slow = df['Close'].ewm(span=slow).mean()
        
        # MACD line
        df['MACD'] = ema_fast - ema_slow
        
        # Signal line
        df['Signal'] = df['MACD'].ewm(span=signal).mean()
        
        # MACD Histogram
        df['MACD_Histogram'] = df['MACD'] - df['Signal']
        
        print("[SUCCESS] Tính MACD thành công")
        return df
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi tính MACD: {str(e)}")
        return df


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    Tính Bollinger Bands
    - Middle Band (MA 20)
    - Upper Band (MA + 2*StdDev)
    - Lower Band (MA - 2*StdDev)
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu giá
        period (int): Chu kỳ Moving Average (mặc định 20)
        std_dev (int): Độ lệch chuẩn (mặc định 2)
    
    Returns:
        pd.DataFrame: DataFrame với thêm cột Bollinger Bands
    """
    try:
        print("[INFO] Tính Bollinger Bands...")
        
        # Middle band = SMA
        df.loc[:, 'BB_Middle'] = df['Close'].rolling(window=period).mean()
        
        print("[SUCCESS] Tính Bollinger Bands thành công")
        return df
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi tính Bollinger Bands: {str(e)}")
        return df


def display_data_info(df):
    """
    Hiển thị thông tin cơ bản về DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame cần hiển thị
    """
    try:
        print("\n" + "="*80)
        print("THÔNG TIN CƠ BẢN VỀ DỮ LIỆU")
        print("="*80)
        
        print("\n[DataFrame Info]")
        print(df.info())
        
        print("\n[Thống kê mô tả]")
        print(df.describe())
        
        print(f"\n[Shape] {df.shape[0]} hàng × {df.shape[1]} cột")
        print(f"[Date Range] {df.index[0].date()} đến {df.index[-1].date()}")
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi hiển thị thông tin: {str(e)}")


def plot_price_with_ma_and_bb(df, ticker):
    """
    Vẽ biểu đồ giá với Moving Averages và Bollinger Bands
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        ticker (str): Mã cổ phiếu
    """
    try:
        print("\n[INFO] Vẽ biểu đồ giá, MA...")
        
        plt.figure(figsize=(14, 7))
        
        # Vẽ giá Close
        plt.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=2)
        
        # Vẽ MA
        plt.plot(df.index, df['MA10'], label='MA10', alpha=0.7, linewidth=1.5)
        plt.plot(df.index, df['MA20'], label='MA20', alpha=0.7, linewidth=1.5)
        plt.plot(df.index, df['MA50'], label='MA50', alpha=0.7, linewidth=1.5)
        
        # Vẽ Bollinger Bands nếu có
        if 'BB_Middle' in df.columns:
            bb_std = df['Close'].rolling(window=20).std()
            bb_upper = df['BB_Middle'] + (2 * bb_std)
            bb_lower = df['BB_Middle'] - (2 * bb_std)
            
            plt.plot(df.index, bb_upper, label='BB Upper', 
                    color='red', alpha=0.5, linestyle='--', linewidth=1)
            plt.plot(df.index, bb_lower, label='BB Lower', 
                    color='blue', alpha=0.5, linestyle='--', linewidth=1)
            plt.fill_between(df.index, bb_upper, bb_lower, 
                             alpha=0.1, color='gray')
        
        plt.title(f'{ticker} - Giá với Moving Averages và Bollinger Bands', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Ngày', fontsize=12)
        plt.ylabel('Giá (USD)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('price_ma_bb.png', dpi=100, bbox_inches='tight')
        print("[SUCCESS] Lưu biểu đồ: price_ma_bb.png")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi vẽ biểu đồ giá: {str(e)}")


def plot_rsi_and_macd(df, ticker):
    """
    Vẽ biểu đồ RSI và MACD
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        ticker (str): Mã cổ phiếu
    """
    try:
        print("[INFO] Vẽ biểu đồ RSI và MACD...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Biểu đồ RSI
        axes[0].plot(df.index, df['RSI'], label='RSI (14)', color='purple', linewidth=2)
        axes[0].axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        axes[0].axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        axes[0].fill_between(df.index, 70, 100, alpha=0.1, color='red')
        axes[0].fill_between(df.index, 0, 30, alpha=0.1, color='green')
        axes[0].set_title('RSI (14 periods)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('RSI', fontsize=11)
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 100])
        
        # Biểu đồ MACD
        axes[1].plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=1.5)
        axes[1].plot(df.index, df['Signal'], label='Signal', color='red', linewidth=1.5)
        axes[1].bar(df.index, df['MACD_Histogram'], label='Histogram', 
                   alpha=0.3, color='gray')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_title('MACD (12, 26, 9)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Ngày', fontsize=11)
        axes[1].set_ylabel('MACD', fontsize=11)
        axes[1].legend(loc='best', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{ticker} - Chỉ báo kỹ thuật RSI và MACD', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('rsi_macd.png', dpi=100, bbox_inches='tight')
        print("[SUCCESS] Lưu biểu đồ: rsi_macd.png")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi vẽ biểu đồ RSI/MACD: {str(e)}")


def save_processed_data(df, filename='stock_data_processed.csv'):
    """
    Lưu dữ liệu đã xử lý thành file CSV
    
    Args:
        df (pd.DataFrame): DataFrame cần lưu
        filename (str): Tên file CSV
    """
    try:
        print(f"\n[INFO] Lưu dữ liệu vào {filename}...")
        
        # Chọn các cột quan trọng
        cols_to_save = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA10', 'MA20', 'MA50',
            'RSI', 'MACD', 'Signal', 'MACD_Histogram',
            'BB_Middle'
        ]
        
        # Lọc các cột tồn tại
        cols_to_save = [col for col in cols_to_save if col in df.columns]
        
        df_save = df[cols_to_save].copy()
        df_save.to_csv(filename)
        
        print(f"[SUCCESS] Lưu thành công {len(df_save)} bản ghi vào {filename}")
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi lưu dữ liệu: {str(e)}")


def main():
    """
    Hàm main - điểm vào chính của chương trình
    """
    print("\n" + "="*80)
    print("PHÂN TÍCH DỮ LIỆU CỔ PHIẾU APPLE (AAPL) - 5 NĂM")
    print("="*80)
    
    try:
        # 1. Tải dữ liệu
        df = load_stock_data(ticker='AAPL', period_years=5)
        if df is None or df.empty:
            print("[ERROR] Không thể tải dữ liệu, chương trình dừng")
            return
        
        # 2. Xử lý missing data
        df = handle_missing_data(df)
        
        # 3. Tính các chỉ báo kỹ thuật
        df = calculate_moving_averages(df)
        df = calculate_rsi(df, period=14)
        df = calculate_macd(df, fast=12, slow=26, signal=9)
        df = calculate_bollinger_bands(df, period=20, std_dev=2)
        
        # 4. Hiển thị thông tin
        display_data_info(df)
        
        # 5. Vẽ biểu đồ
        plot_price_with_ma_and_bb(df, 'AAPL')
        plot_rsi_and_macd(df, 'AAPL')
        
        # 6. Lưu dữ liệu
        save_processed_data(df, filename='stock_data_processed.csv')
        
        print("\n" + "="*80)
        print("[SUCCESS] HOÀN THÀNH PHÂN TÍCH!")
        print("="*80)
        print("Files đã tạo:")
        print("  - stock_data_processed.csv (dữ liệu xử lý)")
        print("  - price_ma_bb.png (biểu đồ giá + MA + BB)")
        print("  - rsi_macd.png (biểu đồ RSI + MACD)")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Lỗi nghiêm trọng: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
