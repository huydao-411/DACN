# app.py
import os
from datetime import date

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from stock_data import StockDataService
from finML import FinMLModel

# ==========================
# Flask app & CORS
# ==========================
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # FE Vite

# Cache model trong RAM để không train lại mỗi lần
MODEL_CACHE: dict[tuple[str, int], FinMLModel] = {}


def get_or_train_model(symbol: str, horizon: int) -> FinMLModel:
    """
    Trả về FinMLModel cho (symbol, horizon).
    - Ưu tiên: load trong RAM
    - Nếu chưa có: thử load .joblib
    - Nếu chưa có .joblib: train mới rồi lưu
    """
    key = (symbol.upper(), horizon)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"finml_{symbol.upper()}_h{horizon}.joblib")

    if os.path.exists(model_path):
        # Load model đã train
        model = FinMLModel.load(model_path)
        MODEL_CACHE[key] = model
        return model

    # Train mới
    print(f"[FinML] Training new model for {symbol} (h={horizon})...")
    today_str = date.today().isoformat()

    model = FinMLModel(
        symbol=symbol.upper(),
        horizon=horizon,
        smooth_target=True,   # bạn có thể chỉnh
        source="VCI",
    )
    model.fit(start="2018-01-01", end=today_str)
    model.save(model_path)
    MODEL_CACHE[key] = model
    return model


# ==========================
# Helpers lấy indicator từ df
# ==========================
def extract_latest_indicators(df_feat: pd.DataFrame) -> dict:
    """
    Lấy một số chỉ báo tại dòng cuối cùng để trả cho FE hiển thị.
    """
    last = df_feat.sort_values("date").iloc[-1]

    # RSI
    rsi_cols = [c for c in df_feat.columns if c.startswith("RSI_")]
    rsi_val = float(last[rsi_cols[0]]) if rsi_cols else None

    # MACD line
    macd_cols = [c for c in df_feat.columns if c.startswith("MACD_")]
    macd_val = float(last[macd_cols[0]]) if macd_cols else None

    # EMA 20, 50
    ema20_col = "EMA_20" if "EMA_20" in df_feat.columns else None
    ema50_col = "EMA_50" if "EMA_50" in df_feat.columns else None
    ema20_val = float(last[ema20_col]) if ema20_col else None
    ema50_val = float(last[ema50_col]) if ema50_col else None

    # Bollinger Bands
    bbu_cols = [c for c in df_feat.columns if c.startswith("BBU_")]
    bbl_cols = [c for c in df_feat.columns if c.startswith("BBL_")]
    bb_upper = float(last[bbu_cols[0]]) if bbu_cols else None
    bb_lower = float(last[bbl_cols[0]]) if bbl_cols else None

    return {
        "rsi_14": rsi_val,
        "macd": macd_val,
        "ema_20": ema20_val,
        "ema_50": ema50_val,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
    }


# ==========================
# ROUTES
# ==========================

@app.route("/")
def health_check():
    return jsonify({"message": "Fin Flask API is running"})


# ---------- FinML: /finml/predict ----------
@app.route("/finml/predict", methods=["POST"])
def finml_predict():
    """
    Body JSON:
      {
        "symbol": "VCB",
        "horizon": 7
      }

    Trả về JSON khớp FE:
      {
        "symbol": "...",
        "horizon": ...,
        "history": { "dates": [...], "prices": [...] },
        "forecast": { "dates": [...], "prices": [...] },
        "metrics": { "mae": ..., "rmse": ..., "r2": ... },
        "indicators": { "rsi_14": ..., "macd": ..., "ema_20": ..., "ema_50": ..., "bb_upper": ..., "bb_lower": ... }
      }
    """
    data = request.get_json(force=True)
    symbol = data.get("symbol", "").upper()
    horizon = int(data.get("horizon", 1))

    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    try:
        # 1. Lấy hoặc train model
        model = get_or_train_model(symbol, horizon)

        # 2. Lấy dữ liệu mới nhất + indicators cho history & predict
        today_str = date.today().isoformat()
        # lấy tầm 3 năm gần nhất cho chart cho nhẹ
        start_default = "2022-01-01"

        data_service = StockDataService(symbol=symbol, source="VCI")
        df_feat = data_service.load_with_indicators(start=start_default, end=today_str)

        # 3. History cho FE: dùng cột date & close
        df_feat_sorted = df_feat.sort_values("date")
        chart_data = data_service.get_chart_data()
        history_dates = df_feat_sorted["date"].dt.strftime("%Y-%m-%d").tolist()
        history_prices = df_feat_sorted["close"].astype(float).tolist()

        # 4. Dự đoán giá tương lai từ df_feat
        pred_info = model.predict_next(df_feat_sorted)
        pred_price = pred_info["predicted_price"]

        last_date = df_feat_sorted["date"].iloc[-1]
        future_date = (last_date + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d")

        # 5. Metrics: ưu tiên test_metrics, fallback sang val_metrics
        metrics = model.test_metrics or model.val_metrics or {"mae": None, "rmse": None, "r2": None}
        metrics_clean = {
            "mae": float(metrics.get("mae")) if metrics.get("mae") is not None else None,
            "rmse": float(metrics.get("rmse")) if metrics.get("rmse") is not None else None,
            "r2": float(metrics.get("r2")) if metrics.get("r2") is not None else None,
        }

        # 6. Indicators tại thời điểm cuối
        indicators = extract_latest_indicators(df_feat_sorted)

        # 7. Ghép thành response JSON cho FE
        resp = {
            "symbol": symbol,
            "horizon": horizon,
            "history": {
                "dates": history_dates,
                "prices": history_prices,
            },
            "forecast": {
                "dates": [future_date],
                "prices": [pred_price],
            },
            "metrics": metrics_clean,
            "indicators": indicators,
            "chart": chart_data,   # 👈 thêm dòng này
        }
        return jsonify(resp)

    except Exception as e:
        print("[ERROR] /finml/predict:", e)
        return jsonify({"error": str(e)}), 500


# ---------- FinNLP: /finnlp/sentiment ----------
@app.route("/finnlp/sentiment", methods=["POST"])
def finnlp_sentiment():
    """
    Tạm thời fake logic NLP.
    Sau này bạn thay bằng model thật (VD: transformer, underthesea, v.v.).
    Body:
      { "text": "..." }
    Trả:
      { "text": "...", "score": 0.75, "label": "positive" }
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    # Fake sentiment: nếu có từ "tăng" coi là positive, có "giảm" coi là negative
    text_lower = text.lower()
    if "giảm" in text_lower:
        label = "negative"
        score = -0.6
    elif "tăng" in text_lower or "tích cực" in text_lower:
        label = "positive"
        score = 0.7
    else:
        label = "neutral"
        score = 0.0

    return jsonify({
        "text": text,
        "score": float(score),
        "label": label
    })


# ---------- FinNLP: /finnlp/summarize ----------
@app.route("/finnlp/summarize", methods=["POST"])
def finnlp_summarize():
    """
    Tạm thời tóm tắt kiểu đơn giản:
    - Cắt bớt text cho ngắn.
    Sau này bạn thay bằng model tóm tắt thật (VD: BART, T5).
    Body:
      { "text": "...", "max_sentences": 3 }
    Trả:
      { "text": "...", "summary": "...", "max_sentences": 3 }
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    max_sentences = int(data.get("max_sentences", 3))

    if not text:
        return jsonify({"error": "text is required"}), 400

    # Fake summarize: cắt ngắn khoảng 200 ký tự
    summary = text
    if len(text) > 200:
        summary = text[:200] + "..."

    return jsonify({
        "text": text,
        "summary": summary,
        "max_sentences": max_sentences
    })


if __name__ == "__main__":
    # Chạy Flask dev server
    app.run(host="0.0.0.0", port=8000, debug=True)
