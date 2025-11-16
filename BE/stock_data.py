import numpy as np
import pandas as pd
from vnstock import Vnstock
import pandas_ta as ta


class StockDataService:
    """
    Service chuyên xử lý dữ liệu 1 mã cổ phiếu:
      - Fetch dữ liệu lịch sử từ VNStock
      - Thêm các chỉ báo kỹ thuật bằng pandas_ta
      - Chuẩn bị dữ liệu dạng JSON-friendly cho FE vẽ chart
    """

    def __init__(self, symbol: str, source: str = "VCI"):
        self.symbol = symbol
        self.source = source

        self.df_raw: pd.DataFrame | None = None   # dữ liệu gốc (OHLCV)
        self.df_feat: pd.DataFrame | None = None  # có thêm indicators

    # ==============================
    # LẤY DỮ LIỆU TỪ VNSTOCK
    # ==============================
    def fetch_stock_data(self,
                         start: str = "2018-01-01",
                         end: str = "2025-01-01") -> pd.DataFrame:
        stock = Vnstock().stock(symbol=self.symbol, source=self.source)
        df = stock.quote.history(start=start, end=end, interval='1D')

        # Đảm bảo có cột date dạng datetime
        if "time" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        self.df_raw = df
        return df

    # ==============================
    # THÊM CHỈ BÁO BẰNG PANDAS_TA
    # ==============================
    def add_indicators(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Nếu df=None thì dùng self.df_raw.
        Trả về df_feat (OHLCV + indicators) và lưu vào self.df_feat
        """
        if df is None:
            if self.df_raw is None:
                raise RuntimeError("Chưa có dữ liệu. Hãy gọi fetch_stock_data trước.")
            df = self.df_raw

        df = df.copy()

        # EMA
        df.ta.ema(length=5, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)

        # MACD
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        # RSI
        df.ta.rsi(length=14, append=True)

        # ROC, MOM
        df.ta.roc(length=10, append=True)
        df.ta.mom(length=10, append=True)

        # ADX
        df.ta.adx(length=14, append=True)

        # Bollinger Bands
        df.ta.bbands(length=20, std=2, append=True)

        # ATR
        df.ta.atr(length=14, append=True)

        # OBV
        df.ta.obv(append=True)

        # MFI
        df.ta.mfi(length=14, append=True)

        # VWMA
        df.ta.vwma(length=20, append=True)

        close = df["close"]
        volume = df["volume"]

        # Returns & volatility
        df["ret1"] = close.pct_change(1)
        df["ret3"] = close.pct_change(3)
        df["ret5"] = close.pct_change(5)
        df["ret10"] = close.pct_change(10)

        df["volatility_5"] = close.pct_change().rolling(5).std()
        df["volatility_10"] = close.pct_change().rolling(10).std()

        # Volume ratio
        df["vol_ma20"] = volume.rolling(20).mean()
        df["vol_ratio"] = volume / df["vol_ma20"]

        # Bỏ các dòng đầu bị NaN (do rolling)
        df = df.dropna().reset_index(drop=True)

        self.df_feat = df
        return df

    # ==============================
    # HÀM TIỆN LỢI: LẤY + INDICATOR 1 PHÁT
    # ==============================
    def load_with_indicators(self,
                             start: str = "2018-01-01",
                             end: str = "2025-01-01") -> pd.DataFrame:
        """
        Gộp: fetch_stock_data + add_indicators
        Dùng cho những chỗ chỉ cần 1 câu gọi.
        """
        df_raw = self.fetch_stock_data(start=start, end=end)
        df_feat = self.add_indicators(df_raw)
        return df_feat

    # ==============================
    # CHUẨN BỊ DATA CHO FRONTEND VẼ CHART
    # ==============================
    def get_chart_data(self) -> dict:
        """
        Trả về dữ liệu JSON-friendly cho FE:
          - price: OHLC + volume
          - rsi: RSI 14
          - macd: MACD line, signal, histogram
          - bbands: Bollinger Bands
        FE có thể dùng trực tiếp cho Chart.js / ECharts / Highcharts...
        """
        if self.df_feat is None:
            raise RuntimeError("Chưa có df_feat. Hãy gọi load_with_indicators hoặc add_indicators trước.")

        df = self.df_feat

        # Đảm bảo date là string ISO để FE parse dễ
        date_str = df["date"].dt.strftime("%Y-%m-%d")

        # Tự tìm tên cột MACD, BBands, EMA... do pandas_ta đặt tên
        macd_cols = [c for c in df.columns if c.startswith("MACD_")]
        macdh_cols = [c for c in df.columns if c.startswith("MACDh_")]
        macds_cols = [c for c in df.columns if c.startswith("MACDs_")]
        rsi_cols = [c for c in df.columns if c.startswith("RSI_")]
        bb_upper_cols = [c for c in df.columns if c.startswith("BBU_")]
        bb_middle_cols = [c for c in df.columns if c.startswith("BBM_")]
        bb_lower_cols = [c for c in df.columns if c.startswith("BBL_")]

        chart_data = {
            "symbol": self.symbol,
            "price": [
                {
                    "date": d,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v),
                }
                for d, o, h, l, c, v in zip(
                    date_str,
                    df["open"],
                    df["high"],
                    df["low"],
                    df["close"],
                    df["volume"],
                )
            ],
            "rsi": [],
            "macd": [],
            "bbands": [],
        }

        # RSI (lấy cột đầu tiên)
        if rsi_cols:
            rsi_col = rsi_cols[0]
            chart_data["rsi"] = [
                {"date": d, "rsi": float(r)}
                for d, r in zip(date_str, df[rsi_col])
            ]

        # MACD
        if macd_cols and macdh_cols and macds_cols:
            macd_col = macd_cols[0]
            macdh_col = macdh_cols[0]
            macds_col = macds_cols[0]
            chart_data["macd"] = [
                {
                    "date": d,
                    "macd": float(m),
                    "signal": float(s),
                    "hist": float(h),
                }
                for d, m, s, h in zip(
                    date_str,
                    df[macd_col],
                    df[macds_col],
                    df[macdh_col],
                )
            ]

        # Bollinger Bands
        if bb_upper_cols and bb_middle_cols and bb_lower_cols:
            bbu = bb_upper_cols[0]
            bbm = bb_middle_cols[0]
            bbl = bb_lower_cols[0]
            chart_data["bbands"] = [
                {
                    "date": d,
                    "upper": float(u),
                    "middle": float(m),
                    "lower": float(l),
                }
                for d, u, m, l in zip(
                    date_str,
                    df[bbu],
                    df[bbm],
                    df[bbl],
                )
            ]

        return chart_data
