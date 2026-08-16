"""Time Series Forecasting Skill - ARIMA / Prophet / LSTM templates.

Sinh code dự báo time-series: stationarity tests (ADF/KPSS), decomposition
(trend/seasonal/residual), ARIMA auto-selection, Prophet forecasting, và
LSTM sequence model với train/test split + backtesting.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


ARIMA_TEMPLATE = '''"""ARIMA forecasting pipeline / Pipeline dự báo ARIMA."""
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def check_stationarity(y: pd.Series, alpha: float = 0.05) -> dict:
    """ADF (H0: unit-root, non-stationary) + KPSS (H0: stationary)."""
    adf_stat, adf_p, *_ = adfuller(y.dropna())
    kpss_stat, kpss_p, *_ = kpss(y.dropna())
    return {
        "adf_statistic": float(adf_stat), "adf_p_value": float(adf_p),
        "adf_stationary": adf_p < alpha,
        "kpss_statistic": float(kpss_stat), "kpss_p_value": float(kpss_p),
        "kpss_stationary": kpss_p > alpha,
        "verdict": "stationary" if (adf_p < alpha and kpss_p > alpha) else "non-stationary",
    }


def decompose(y: pd.Series, period: int = 12):
    """Phân rã trend + seasonal + residual."""
    return seasonal_decompose(y, model="additive", period=period)


def auto_difference(y: pd.Series, max_d: int = 2) -> tuple[pd.Series, int]:
    """Difference đến khi stationary / Lấy sai phân đến khi dừng."""
    d = 0
    cur = y.copy()
    while d < max_d:
        s = check_stationarity(cur.dropna())
        if s["adf_stationary"]:
            return cur, d
        cur = cur.diff().dropna()
        d += 1
    return cur, d


def fit_sarimax(y: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def forecast(model, steps: int = 12) -> pd.DataFrame:
    fc = model.get_forecast(steps=steps)
    ci = fc.conf_int(alpha=0.05)
    return pd.DataFrame({
        "forecast": fc.predicted_mean,
        "lower_95": ci.iloc[:, 0],
        "upper_95": ci.iloc[:, 1],
    })
'''

PROPHET_TEMPLATE = '''"""Prophet forecasting / Dự báo bằng Prophet."""
from __future__ import annotations
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_components


def fit_prophet(df: pd.DataFrame, horizon: str = "30 days") -> dict:
    """df phải có cột `ds` (datetime) và `y` (numeric).

    Returns:
        Dict chứa model, forecast, và cross-val metrics.
    """
    m = Prophet(
        seasonality_mode="additive",
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    m.fit(df)

    future = m.make_future_dataframe(periods=30, freq="D")
    fc = m.predict(future)

    cv = cross_validation(m, horizon=horizon, parallel="processes")
    metrics = performance_metrics(cv).to_dict("records")
    return {"model": m, "forecast": fc, "cv_metrics": metrics}


def plot_forecast(m, fc):
    """Ve do thi du bao + components (trend, weekly, yearly)."""
    fig1 = m.plot(fc)
    fig2 = plot_components(m, fc)
    return fig1, fig2
'''

LSTM_TEMPLATE = '''"""LSTM time-series forecasting with PyTorch / LSTM cho time-series."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SeqDataset(Dataset):
    def __init__(self, series: np.ndarray, window: int = 24, horizon: int = 1):
        self.x, self.y = [], []
        for i in range(len(series) - window - horizon + 1):
            self.x.append(series[i:i + window])
            self.y.append(series[i + window:i + window + horizon])
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


class LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers,
                            dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train(series: np.ndarray, epochs: int = 50, window: int = 24):
    """Train LSTM trên chuỗi đã standardized / Huấn luyện LSTM."""
    mean, std = series.mean(), series.std()
    series = (series - mean) / std

    ds = SeqDataset(series, window=window)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = LSTMForecaster()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb.squeeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        if ep % 10 == 0:
            print(f"epoch={ep:3d} mse={total / len(ds):.4f}")
    return model, (mean, std)
'''

BACKTEST_HINTS = """
Backtesting Best Practices / Thực hành tốt cho backtest
=========================================================
- Use temporal split (no shuffling) — train / validation / test in chronological order.
- Walk-forward validation: retrain on each new observation (rolling-origin).
- Metrics: MAE / RMSE / MAPE / MASE (scales better across series).
- Compare against naive baseline: y[t+1] = y[t]. Always beat the baseline.
- Beware data leakage: differencing, scaling fit ONLY on training window.
- Check residual autocorrelation (Ljung-Box) → if significant, model is missing structure.
"""


class TimeSeriesSkill(Skill):
    """Sinh template dự báo time-series (ARIMA / Prophet / LSTM)."""

    category = SkillCategory.DATA
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "time series", "time-series", "forecast", "forecasting", "arima",
        "sarima", "prophet", "lstm", "seasonality", "trend",
        "stationarity", "decomposition", "holt-winters",
    ]
    examples = [
        "Forecast doanh số 12 tháng tới với ARIMA",
        "Setup Prophet model with seasonality",
        "Train LSTM for univariate time series",
    ]

    @property
    def name(self) -> str:
        return "time_series_forecasting"

    @property
    def description(self) -> str:
        return (
            "Sinh template dự báo time-series: ARIMA/SARIMA (statsmodels), Prophet, "
            "và LSTM (PyTorch) với stationarity tests, decomposition, backtesting."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.13
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        prompt_lower = (context.prompt or "").lower()
        if "prophet" in prompt_lower:
            recommended = "prophet"
        elif "lstm" in prompt_lower:
            recommended = "lstm"
        elif "arima" in prompt_lower or "sarima" in prompt_lower:
            recommended = "arima"
        else:
            recommended = "auto"

        artifacts: List[Dict[str, str]] = [
            {"name": "arima_pipeline.py", "language": "python", "content": ARIMA_TEMPLATE},
            {"name": "prophet_pipeline.py", "language": "python", "content": PROPHET_TEMPLATE},
            {"name": "lstm_pipeline.py", "language": "python", "content": LSTM_TEMPLATE},
            {"name": "BACKTESTING.md", "language": "markdown", "content": BACKTEST_HINTS},
        ]

        return SkillResult(
            success=True,
            output=(
                f"[time_series_forecasting] recommended={recommended}\n"
                f"Generated ARIMA + Prophet + LSTM templates + backtesting guide."
            ),
            artifacts=artifacts,
            suggestions=[
                "Check stationarity first (ADF + KPSS) — difference if non-stationary",
                "Beat the naive baseline (y[t+1] = y[t]) before reporting metrics",
                "Add exogenous regressors (holidays, promo) in SARIMAX / Prophet",
                "Use walk-forward CV (rolling-origin) instead of random k-fold",
                "Plot residual ACF / Ljung-Box test to verify no leftover structure",
            ],
            metadata={
                "skill": self.name,
                "recommended_model": recommended,
                "models_available": ["ARIMA", "SARIMA", "Prophet", "LSTM", "Holt-Winters"],
                "metrics": ["MAE", "RMSE", "MAPE", "MASE"],
                "version": self.version,
                "author": self.author,
            },
        )
