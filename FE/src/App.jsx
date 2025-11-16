import { useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Tooltip,
  Legend
);

// Đậm màu chữ mặc định
ChartJS.defaults.color = "#111";
ChartJS.defaults.font.size = 12;

const API_BASE = "http://localhost:8000";

export default function App() {
  // --- States ---
  const [symbol, setSymbol] = useState("VCB");
  const [horizon, setHorizon] = useState(7);
  const [rangeDays, setRangeDays] = useState(30);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [predictError, setPredictError] = useState("");
  const [finmlData, setFinmlData] = useState(null);

  const [nlpText, setNlpText] = useState("");
  const [nlpError, setNlpError] = useState("");
  const [loadingSentiment, setLoadingSentiment] = useState(false);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [sentimentResult, setSentimentResult] = useState(null);
  const [summaryResult, setSummaryResult] = useState(null);

  // --- Helpers ---
  const sliceLast = (arr) => {
    if (!arr) return [];
    const n = arr.length;
    const k = Math.min(rangeDays, n);
    return arr.slice(n - k);
  };

  // --- API Call ---
  const handlePredict = async () => {
    if (!symbol.trim()) {
      setPredictError("Vui lòng nhập mã chứng khoán.");
      return;
    }

    setPredictError("");
    setLoadingPredict(true);
    setFinmlData(null);

    try {
      const resp = await fetch(`${API_BASE}/finml/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: symbol.trim().toUpperCase(),
          horizon: Number(horizon),
        }),
      });

      if (!resp.ok) throw new Error("API error");

      const data = await resp.json();
      setFinmlData(data);
    } catch (err) {
      console.error(err);
      setPredictError("Không gọi được API FinML.");
    } finally {
      setLoadingPredict(false);
    }
  };

  const handleSentiment = async () => {
    if (!nlpText.trim()) {
      setNlpError("Nhập văn bản để phân tích.");
      return;
    }
    setNlpError("");
    setLoadingSentiment(true);

    try {
      const resp = await fetch(`${API_BASE}/finnlp/sentiment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: nlpText }),
      });
      const data = await resp.json();
      setSentimentResult(data);
    } catch {
      setNlpError("Không gọi API sentiment.");
    } finally {
      setLoadingSentiment(false);
    }
  };

  const handleSummarize = async () => {
    if (!nlpText.trim()) {
      setNlpError("Nhập văn bản để tóm tắt.");
      return;
    }

    setNlpError("");
    setLoadingSummary(true);

    try {
      const resp = await fetch(`${API_BASE}/finnlp/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: nlpText, max_sentences: 3 }),
      });
      const data = await resp.json();
      setSummaryResult(data);
    } catch {
      setNlpError("Không gọi API summarize.");
    } finally {
      setLoadingSummary(false);
    }
  };

  // --- Extract ---
  const metrics = finmlData?.metrics ?? {};
  const indicators = finmlData?.indicators ?? {};
  const chart = finmlData?.chart;

  // --- Chart: Price ---
  let priceChartData = null;
  if (chart?.price) {
    const priceSlice = sliceLast(chart.price);
    const historyLabels = priceSlice.map((p) => p.date);
    const historyClose = priceSlice.map((p) => p.close);

    const forecastDates = finmlData?.forecast?.dates ?? [];
    const forecastPrices = finmlData?.forecast?.prices ?? [];
    const labels = [...historyLabels, ...forecastDates];

    priceChartData = {
      labels,
      datasets: [
        {
          label: "Giá đóng cửa (History)",
          data: [...historyClose, ...Array(forecastPrices.length).fill(null)],
          borderColor: "#2563eb",
          pointRadius: 3,
          borderWidth: 2,
          tension: 0.2,
        },
        {
          label: "Giá dự đoán",
          data: [
            ...Array(historyClose.length).fill(null),
            ...forecastPrices,
          ],
          borderColor: "#16a34a",
          borderDash: [6, 3],
          pointRadius: 4,
          borderWidth: 2,
          tension: 0.2,
        },
      ],
    };
  }

  // --- RSI Chart ---
  let rsiChartData = null;
  if (chart?.rsi?.length) {
    const rsiSlice = sliceLast(chart.rsi);
    rsiChartData = {
      labels: rsiSlice.map((p) => p.date),
      datasets: [
        {
          label: "RSI(14)",
          data: rsiSlice.map((p) => p.rsi),
          borderColor: "#7c3aed",
          borderWidth: 2,
          tension: 0.2,
        },
      ],
    };
  }

  // --- MACD Chart ---
  let macdChartData = null;
  if (chart?.macd?.length) {
    const macdSlice = sliceLast(chart.macd);
    macdChartData = {
      labels: macdSlice.map((p) => p.date),
      datasets: [
        {
          type: "line",
          label: "MACD",
          data: macdSlice.map((p) => p.macd),
          borderColor: "#dc2626",
          borderWidth: 2,
        },
        {
          type: "line",
          label: "Signal",
          data: macdSlice.map((p) => p.signal),
          borderColor: "#0ea5e9",
          borderWidth: 2,
          borderDash: [6, 3],
        },
        {
          type: "bar",
          label: "Histogram",
          data: macdSlice.map((p) => p.hist),
          backgroundColor: "#94a3b8",
        },
      ],
    };
  }

  // --- BBands Chart ---
  let bbChartData = null;
  if (chart?.bbands?.length) {
    const bbs = sliceLast(chart.bbands);
    bbChartData = {
      labels: bbs.map((p) => p.date),
      datasets: [
        {
          label: "Upper Band",
          data: bbs.map((p) => p.upper),
          borderColor: "#f97316",
          borderWidth: 2,
        },
        {
          label: "Middle Band",
          data: bbs.map((p) => p.middle),
          borderColor: "#6b7280",
          borderWidth: 2,
        },
        {
          label: "Lower Band",
          data: bbs.map((p) => p.lower),
          borderColor: "#0ea5e9",
          borderWidth: 2,
        },
      ],
    };
  }

  return (
    <div className="bg-light min-vh-100">
      <div className="container-fluid py-3">
        {/* HEADER: full width */}
        <header className="mb-3 d-flex justify-content-between align-items-center">
          <div>
            <h3 className="mb-0">Fin Dashboard</h3>
            <small className="text-muted">
              Dự đoán & phân tích kỹ thuật (FinML) + NLP (FinNLP)
            </small>
          </div>
          <span className="badge text-bg-success">
            Flask API + React (Vite)
          </span>
        </header>

        {/* ===== 3 CỘT: 20% - 60% - 20% ===== */}
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
          }}
        >
          {/* ==== 20% TRÁI: 3 CARD XẾP DỌC ==== */}
          <div
            style={{
              width: "20%",
              paddingRight: "12px",
              minWidth: "220px",
            }}
          >
            {/* CARD 1: Thiết lập dự đoán */}
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h5>Thiết lập dự đoán</h5>

                <label className="form-label mt-2">Mã chứng khoán</label>
                <input
                  className="form-control"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                />

                <label className="form-label mt-3">Horizon dự đoán</label>
                <select
                  className="form-select"
                  value={horizon}
                  onChange={(e) => setHorizon(e.target.value)}
                >
                  <option value="1">1 ngày</option>
                  <option value="2">2 ngày</option>
                  <option value="14">14 ngày</option>
                  <option value="30">30 ngày</option>
                </select>

                <label className="form-label mt-3">
                  Khoảng thời gian hiển thị
                </label>
                <select
                  className="form-select"
                  value={rangeDays}
                  onChange={(e) => setRangeDays(Number(e.target.value))}
                >
                  <option value="30">1 tháng</option>
                  <option value="90">3 tháng</option>
                  <option value="180">6 tháng</option>
                  <option value="365">1 năm</option>
                </select>

                <button
                  className="btn btn-primary w-100 mt-3"
                  onClick={handlePredict}
                  disabled={loadingPredict}
                >
                  {loadingPredict ? "Đang dự đoán..." : "Chạy dự đoán"}
                </button>

                {predictError && (
                  <div className="alert alert-danger mt-2">
                    {predictError}
                  </div>
                )}
              </div>
            </div>

            {/* CARD 2: Độ chính xác */}
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h5>Độ chính xác mô hình</h5>
                {finmlData ? (
                  <ul className="mt-2 small">
                    <li>MAE: {metrics.mae?.toFixed?.(4)}</li>
                    <li>RMSE: {metrics.rmse?.toFixed?.(4)}</li>
                    <li>R²: {metrics.r2?.toFixed?.(4)}</li>
                  </ul>
                ) : (
                  <p className="text-muted small">Chưa có kết quả.</p>
                )}
              </div>
            </div>

            {/* CARD 3: Chỉ báo hiện tại */}
            <div className="card shadow-sm">
              <div className="card-body">
                <h5>Chỉ báo hiện tại</h5>
                {finmlData ? (
                  <ul className="mt-2 small">
                    <li>RSI(14): {indicators.rsi_14?.toFixed?.(2)}</li>
                    <li>MACD: {indicators.macd?.toFixed?.(4)}</li>
                    <li>EMA20: {indicators.ema_20}</li>
                    <li>EMA50: {indicators.ema_50}</li>
                  </ul>
                ) : (
                  <p className="text-muted small">Chưa có dữ liệu.</p>
                )}
              </div>
            </div>
          </div>

          {/* ==== 60% GIỮA: CHART + NLP ==== */}
          <div
            style={{
              width: "60%",
              paddingInline: "12px",
            }}
          >
            {/* PRICE CHART */}
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h5 className="mb-3">
                  Giá đóng cửa{" "}
                  {finmlData?.symbol ? `- ${finmlData.symbol}` : ""}
                </h5>
                <div style={{ height: 350 }}>
                  {priceChartData ? (
                    <Line
                      data={priceChartData}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                      }}
                    />
                  ) : (
                    <p className="text-muted">Chưa có dữ liệu.</p>
                  )}
                </div>
              </div>
            </div>

            {/* RSI + MACD */}
            <div className="row g-3 mb-3">
              <div className="col-md-6">
                <div className="card shadow-sm">
                  <div className="card-body">
                    <h6>RSI(14)</h6>
                    <p className="text-muted small">
                      RSI cho biết giá đang quá cao hay quá thấp. Trên 70 = quá mua; dưới 30 = quá bán.
                    </p>
                    <div style={{ height: 220 }}>
                      {rsiChartData ? (
                        <Line
                          data={rsiChartData}
                          options={{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: { y: { min: 0, max: 100 } },
                          }}
                        />
                      ) : (
                        <p className="text-muted small">Chưa có RSI.</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              <div className="col-md-6">
                <div className="card shadow-sm">
                  <div className="card-body">
                    <h6>MACD</h6>
                    <p className="text-muted small">
                      MACD cho biết xu hướng mạnh hay yếu. Giao cắt Signal = tín hiệu mua/bán.
                    </p>
                    <div style={{ height: 220 }}>
                      {macdChartData ? (
                        <Bar
                          data={macdChartData}
                          options={{
                            responsive: true,
                            maintainAspectRatio: false,
                          }}
                        />
                      ) : (
                        <p className="text-muted small">Chưa có MACD.</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* BBANDS */}
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6>Bollinger Bands</h6>
                <p className="text-muted small">
                  Bollinger Bands đo lường biến động giá. Giá chạm dải trên = quá mua; chạm dải dưới = quá bán.
                </p>
                <div style={{ height: 240 }}>
                  {bbChartData ? (
                    <Line
                      data={bbChartData}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                      }}
                    />
                  ) : (
                    <p className="text-muted small">Chưa có dữ liệu.</p>
                  )}
                </div>
              </div>
            </div>

            {/* FinNLP */}
            <div className="card shadow-sm mb-5">
              <div className="card-body">
                <h5 className="mb-3">FinNLP – Phân tích văn bản</h5>

                <textarea
                  className="form-control mb-3"
                  rows={3}
                  value={nlpText}
                  onChange={(e) => setNlpText(e.target.value)}
                />

                <div className="d-flex gap-2 mb-3 flex-wrap">
                  <button
                    className="btn btn-outline-primary"
                    disabled={loadingSentiment}
                    onClick={handleSentiment}
                  >
                    {loadingSentiment ? "Đang phân tích..." : "Sentiment"}
                  </button>

                  <button
                    className="btn btn-outline-secondary"
                    disabled={loadingSummary}
                    onClick={handleSummarize}
                  >
                    {loadingSummary ? "Đang tóm tắt..." : "Tóm tắt"}
                  </button>
                </div>

                {nlpError && (
                  <div className="alert alert-danger py-2">{nlpError}</div>
                )}

                <div className="row small mt-3">
                  <div className="col-md-4">
                    <h6>Sentiment</h6>
                    {sentimentResult ? (
                      <>
                        <div>Label: {sentimentResult.label}</div>
                        <div>
                          Score: {sentimentResult.score?.toFixed?.(4)}
                        </div>
                      </>
                    ) : (
                      <p className="text-muted small">Chưa có kết quả.</p>
                    )}
                  </div>

                  <div className="col-md-8">
                    <h6>Tóm tắt</h6>
                    {summaryResult ? (
                      <p>{summaryResult.summary}</p>
                    ) : (
                      <p className="text-muted small">Chưa có tóm tắt.</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* ==== 20% PHẢI: TRỐNG ==== */}
          <div
            style={{
              width: "20%",
            }}
          ></div>
        </div>
      </div>
    </div>
  );
}
