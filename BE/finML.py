import numpy as np
import pandas as pd

from vnstock import Vnstock
import pandas_ta as ta

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from math import sqrt

from stock_data import StockDataService

# ==============================
# CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN
# ==============================
def prepare_dataset(df: pd.DataFrame,
                    horizon: int = 3,
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1,
                    smooth_target: bool = False):
    """
    Tạo target = giá close(t + horizon).
    Nếu smooth_target=True thì dùng rolling mean để giảm noise.
    Chia train/val/test theo tỷ lệ thời gian.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Target: giá tương lai
    df["raw_target"] = df["close"].shift(-horizon)

    if smooth_target:
        df["target"] = df["raw_target"].rolling(3).mean()
    else:
        df["target"] = df["raw_target"]

    # Xóa các dòng cuối cùng không đủ tương lai + smoothing NaN
    df = df.dropna(subset=["target"])

    # Chọn feature numeric
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Loại bỏ target và raw_target ra khỏi features
    for col in ["target", "raw_target"]:
        if col in feature_cols:
            feature_cols.remove(col)

    X = df[feature_cols].values
    y = df["target"].values

    n = len(df)
    if n < 100:
        raise ValueError("Dữ liệu quá ít, cần >= 100 mẫu để train cho đàng hoàng.")

    idx_train_end = int(n * train_ratio)
    idx_val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:idx_train_end]
    y_train = y[:idx_train_end]

    X_val = X[idx_train_end:idx_val_end]
    y_val = y[idx_train_end:idx_val_end]

    X_test = X[idx_val_end:]
    y_test = y[idx_val_end:]

    # Scale (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        "df": df,
        "feature_cols": feature_cols,
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler
    }



# ==============================
# HUẤN LUYỆN & CHỌN MÔ HÌNH TỐT NHẤT
# ==============================
def evaluate_regression(y_true, y_pred, prefix=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{prefix}MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_select_model(X_train, y_train, X_val, y_val):
    candidates = {}

    # ---- XGBoost ----
    xgb = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.2,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    pred_val_xgb = xgb.predict(X_val)
    metrics_xgb = evaluate_regression(y_val, pred_val_xgb, prefix="[XGBoost VAL] ")
    candidates["xgboost"] = {"model": xgb, "metrics": metrics_xgb}

    # ---- RandomForest ----
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred_val_rf = rf.predict(X_val)
    metrics_rf = evaluate_regression(y_val, pred_val_rf, prefix="[RF VAL]      ")
    candidates["random_forest"] = {"model": rf, "metrics": metrics_rf}

    # Chọn model có MAE nhỏ nhất
    best_name = None
    best_entry = None
    best_mae = float("inf")
    for name, entry in candidates.items():
        mae = entry["metrics"]["mae"]
        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_entry = entry

    print(f"\n👉 Best model on VAL: {best_name} (MAE={best_mae:.4f})")
    return best_name, best_entry["model"], best_entry["metrics"]


# ==============================
# FINML WRAPPER CLASS
# ==============================
class FinMLModel:
    """
    Wrapper cho FinML:
      - dùng StockDataService để fetch + build indicators
      - train + chọn model
      - predict tương lai + confidence
    """

    def __init__(self, symbol: str, horizon: int = 3, smooth_target: bool = False, source: str = "VCI"):
        self.symbol = symbol
        self.horizon = horizon
        self.smooth_target = smooth_target

        self.feature_cols = None
        self.scaler = None
        self.model = None
        self.val_metrics = None
        self.test_metrics = None

        # service đọc dữ liệu & indicators
        self.data_service = StockDataService(symbol=symbol, source=source)

    def fit(self, start: str = "2018-01-01", end: str = "2025-01-01"):
        # 1 + 2. Lấy dữ liệu + thêm chỉ báo qua service
        df_feat = self.data_service.load_with_indicators(start=start, end=end)

        # 3. Chuẩn bị dữ liệu
        data = prepare_dataset(
            df_feat,
            horizon=self.horizon,
            smooth_target=self.smooth_target
        )

        self.feature_cols = data["feature_cols"]
        self.scaler = data["scaler"]

        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]

        # 4. Train + chọn model tốt nhất trên validation
        best_name, best_model, val_metrics = train_and_select_model(
            X_train, y_train, X_val, y_val
        )
        self.model = best_model
        self.val_metrics = val_metrics

        # 5. Đánh giá trên test (out-of-sample)
        y_pred_test = self.model.predict(X_test)
        self.test_metrics = evaluate_regression(y_test, y_pred_test, prefix="[TEST]      ")

        print(f"\n✅ FinML model ({self.symbol}) trained with best={best_name}")
        return self


    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("Model chưa được train.")
        obj = {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "smooth_target": self.smooth_target,
            "feature_cols": self.feature_cols,
            "scaler": self.scaler,
            "model": self.model,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
        }
        dump(obj, path)
        print(f"💾 Saved FinML model to {path}")

    @staticmethod
    def load(path: str):
        obj = load(path)
        inst = FinMLModel(symbol=obj["symbol"],
                          horizon=obj["horizon"],
                          smooth_target=obj["smooth_target"])
        inst.feature_cols = obj["feature_cols"]
        inst.scaler = obj["scaler"]
        inst.model = obj["model"]
        inst.val_metrics = obj["val_metrics"]
        inst.test_metrics = obj["test_metrics"]
        print(f"📂 Loaded FinML model from {path}")
        return inst

    def _confidence_score(self):
        """
        Confidence đơn giản từ R2 validation (0–1).
        Có thể tinh chỉnh về sau (dùng residual distribution...).
        """
        if not self.val_metrics:
            return 0.0
        r2 = self.val_metrics.get("r2", 0.0)
        # clamp về [0, 1]
        return float(max(0.0, min(1.0, r2)))

    def predict_next(self, df_latest: pd.DataFrame):
        """
        Dự đoán giá close tương lai sau horizon ngày,
        từ một DataFrame đã có đầy đủ feature (chỉ báo).
        Thường sẽ dùng df mới nhất (vd: output của add_indicators trên toàn bộ dữ liệu).
        """
        if self.model is None or self.scaler is None or self.feature_cols is None:
            raise RuntimeError("Model chưa được train hoặc load.")

        df_latest = df_latest.sort_values("date").reset_index(drop=True)
        last_row = df_latest.iloc[[-1]]  # giữ dạng 2D
        X_last = last_row[self.feature_cols].values
        X_last_scaled = self.scaler.transform(X_last)

        pred_price = float(self.model.predict(X_last_scaled)[0])
        conf = self._confidence_score()

        return {
            "symbol": self.symbol,
            "horizon_days": self.horizon,
            "predicted_price": pred_price,
            "confidence": conf
        }

