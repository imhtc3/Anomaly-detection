"""Turn per-minute flags into incident alerts (no alert storm: one alert per incident)."""
import pandas as pd

from .features import METRICS


def group_alerts(df: pd.DataFrame, flags, scores, min_minutes: int = 8,
                 merge_gap: int = 5) -> pd.DataFrame:
    d = df.assign(flag=flags, score=scores).reset_index(drop=True)
    flagged = d.index[d["flag"] == 1].tolist()
    alerts, start, prev = [], None, None
    for i in flagged + [None]:
        if start is None:
            start = prev = i
            continue
        if i is not None and i - prev <= merge_gap:
            prev = i
            continue
        if prev - start + 1 >= min_minutes:
            window = d.loc[start:prev]
            baseline = d.loc[max(0, start - 120):start - 1, METRICS].median()
            change = ((window[METRICS].median() - baseline) / baseline.abs().clip(lower=1e-6)).abs()
            alerts.append({"start": window["timestamp"].iloc[0], "end": window["timestamp"].iloc[-1],
                           "minutes": prev - start + 1, "peak_score": round(window["score"].max(), 3),
                           "likely_cause": change.idxmax()})
        start = prev = i
    return pd.DataFrame(alerts)
