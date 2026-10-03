"""Train on the first 21 days, evaluate on the last 9 (time-based split, no leakage)."""
import pickle

import pandas as pd

from detector.alerts import group_alerts
from detector.data import generate_metrics
from detector.features import build_features
from detector.model import AnomalyDetector, evaluate, incident_recall, zscore_baseline


def main():
    df = generate_metrics()
    X = build_features(df)
    split = df["timestamp"] < "2026-09-22"
    warmup = df["timestamp"] >= "2026-09-02"  # first day has no seasonal baseline yet

    train_mask = split & warmup
    test_df, X_test = df[~split], X[~split]

    detector = AnomalyDetector(contamination=df.loc[train_mask, "is_incident"].mean()).fit(X[train_mask])
    preds = {"Rolling z-score (baseline)": zscore_baseline(X_test),
             "Isolation Forest": detector.predict(X_test)}

    print(f"Train: {train_mask.sum():,} minutes | Test: {len(test_df):,} minutes "
          f"| incident minutes in test: {test_df['is_incident'].sum():,}\n")
    print(f"{'Model':<28}{'Precision':>10}{'Recall':>8}{'F1':>7}{'Incidents caught':>18}")
    for name, p in preds.items():
        m = evaluate(test_df["is_incident"], p)
        print(f"{name:<28}{m['precision']:>10.2f}{m['recall']:>8.2f}{m['f1']:>7.2f}"
              f"{incident_recall(test_df, p):>17.0%}")

    alerts = group_alerts(test_df, preds["Isolation Forest"], detector.score(X_test))
    n_true = int((test_df["is_incident"].diff() == 1).sum())
    real = sum(test_df[test_df["timestamp"].between(a.start, a.end)]["is_incident"].max()
               for a in alerts.itertuples())
    print(f"\n{len(alerts)} alerts raised ({real} on real incidents, "
          f"{real / max(len(alerts), 1):.0%} alert precision) for {n_true} incidents. First 5:")
    print(alerts.head().to_string(index=False))

    with open("model.pkl", "wb") as f:
        pickle.dump(detector, f)
    print("\nSaved model.pkl")


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
