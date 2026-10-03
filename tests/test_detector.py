import unittest

import numpy as np
import pandas as pd

from detector.alerts import group_alerts
from detector.data import generate_metrics
from detector.features import build_features
from detector.model import AnomalyDetector, evaluate, incident_recall


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_metrics(days=6, n_incidents=12)
        cls.X = build_features(cls.df)

    def test_data_has_labeled_incidents(self):
        self.assertGreater(self.df["is_incident"].sum(), 0)
        self.assertEqual(len(self.df), 6 * 24 * 60)

    def test_features_have_no_nans(self):
        self.assertFalse(self.X.isna().any().any())

    def test_features_use_only_past_data(self):
        # Changing the future must not change today's features.
        changed = self.df.copy()
        changed.loc[changed.index[-100:], "cpu_pct"] = 99
        a = build_features(self.df).iloc[:-100]
        b = build_features(changed).iloc[:-100]
        pd.testing.assert_frame_equal(a, b)

    def test_scores_higher_on_incidents(self):
        det = AnomalyDetector(contamination=0.03).fit(self.X)
        s = det.score(self.X)
        inc = self.df["is_incident"] == 1
        self.assertGreater(s[inc.to_numpy()].mean(), s[~inc.to_numpy()].mean())

    def test_evaluate(self):
        m = evaluate([1, 0, 1, 0], [1, 0, 0, 0])
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 0.5)

    def test_incident_recall(self):
        df = pd.DataFrame({"is_incident": [0, 1, 1, 0, 1, 1, 0]})
        self.assertEqual(incident_recall(df, [0, 1, 0, 0, 0, 0, 0]), 0.5)

    def test_alerts_merge_consecutive_minutes(self):
        df = self.df.head(300).copy()
        flags = np.zeros(300, dtype=int)
        flags[150:160] = 1
        flags[163:170] = 1  # small gap -> same alert
        alerts = group_alerts(df, flags, np.zeros(300))
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts.iloc[0]["minutes"], 20)


if __name__ == "__main__":
    unittest.main()


class ScoreLatestTests(unittest.TestCase):
    def test_score_latest_flags_incident_more_than_normal(self):
        from detector.model import MIN_HISTORY, score_latest
        df = generate_metrics(days=4, n_incidents=10)
        det = AnomalyDetector(contamination=0.03).fit(build_features(df))
        inc = df.index[(df["is_incident"] == 1) & (df.index > MIN_HISTORY)][5]
        ok = df.index[(df["is_incident"] == 0) & (df.index > MIN_HISTORY + 500)][0]
        s_inc = score_latest(det, df.loc[inc - MIN_HISTORY + 1:inc])["anomaly_score"]
        s_ok = score_latest(det, df.loc[ok - MIN_HISTORY + 1:ok])["anomaly_score"]
        self.assertGreater(s_inc, s_ok)

    def test_score_latest_rejects_short_history(self):
        from detector.model import score_latest
        df = generate_metrics(days=2, n_incidents=2)
        det = AnomalyDetector().fit(build_features(df))
        with self.assertRaises(ValueError):
            score_latest(det, df.head(100))
