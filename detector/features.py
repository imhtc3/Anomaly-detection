"""Feature engineering: compare each minute to its recent and same-time-of-day baseline."""
import pandas as pd

METRICS = ["requests", "cpu_pct", "latency_ms", "error_pct", "memory_pct"]
WINDOW = 60  # minutes


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    minute = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    for m in METRICS:
        x = df[m]
        # Seasonal baseline: typical value at this minute of day (from history only).
        seasonal = x.groupby(minute).transform(lambda s: s.shift(1).expanding().median())
        resid = (x - seasonal).fillna(0)
        roll = resid.shift(1).rolling(WINDOW, min_periods=10)
        out[f"{m}_z"] = ((resid - roll.mean()) / roll.std()).fillna(0).clip(-20, 20)
        out[f"{m}_resid"] = resid
        out[f"{m}_trend"] = x.diff(30).fillna(0)  # catches slow drifts like memory leaks
    return out
