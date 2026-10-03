"""Isolation Forest detector + a rolling z-score baseline to compare against."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    def __init__(self, contamination: float = 0.02, seed: int = 0):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=300, contamination=contamination,
                                     random_state=seed, n_jobs=-1)
        self.feature_names = None

    def fit(self, X: pd.DataFrame) -> "AnomalyDetector":
        self.feature_names = list(X.columns)
        self.model.fit(self.scaler.fit_transform(X))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher = more anomalous."""
        return -self.model.score_samples(self.scaler.transform(X[self.feature_names]))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.model.predict(self.scaler.transform(X[self.feature_names])) == -1).astype(int)


def zscore_baseline(X: pd.DataFrame, threshold: float = 4.0) -> np.ndarray:
    z_cols = [c for c in X.columns if c.endswith("_z")]
    return (X[z_cols].abs() > threshold).any(axis=1).astype(int).to_numpy()


def evaluate(y_true, y_pred) -> dict:
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def incident_recall(df: pd.DataFrame, y_pred) -> float:
    """Share of incidents with at least one flagged minute (what on-call actually cares about)."""
    starts = (df["is_incident"].diff().fillna(df["is_incident"]) == 1).cumsum()
    groups = pd.Series(y_pred, index=df.index)[df["is_incident"] == 1].groupby(starts)
    return float((groups.max() == 1).mean()) if len(groups) else 0.0


MIN_HISTORY = 24 * 60 + 30  # one day for the seasonal baseline + 30 min of recent context


def score_latest(detector: AnomalyDetector, history: pd.DataFrame) -> dict:
    """Score the newest minute given recent history (used by the API)."""
    from .features import build_features
    if len(history) < MIN_HISTORY:
        raise ValueError(f"Need at least {MIN_HISTORY} minutes of history, got {len(history)}")
    X = build_features(history).tail(1)
    return {"timestamp": str(history["timestamp"].iloc[-1]),
            "anomaly_score": round(float(detector.score(X)[0]), 4),
            "is_anomaly": bool(detector.predict(X)[0])}
