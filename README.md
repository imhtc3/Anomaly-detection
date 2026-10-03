# Metric Anomaly Detection

Detects incidents in service metrics (traffic, CPU, latency, errors, memory) with an Isolation Forest, groups flagged minutes into actionable alerts, and serves the model through a FastAPI endpoint.

## How it works

```
per-minute metrics -> features -> Isolation Forest -> per-minute flags -> grouped alerts -> /score API
(30 days, 5 signals)  (seasonal    (trained on        (anomaly score)    (one alert per
                      residual,     days 1-21)                            incident + likely cause)
                      z-score, trend)
```

1. **Data** (`detector/data.py`): 30 days of per-minute metrics with a daily traffic cycle and noise, plus 60 labeled incidents of 5 kinds: CPU spikes, latency spikes, error bursts, traffic drops, and slow memory leaks.
2. **Features** (`detector/features.py`): for each metric, the gap from its usual value at that time of day, a rolling z-score of that gap, and a 30-minute trend. Features use only past data, which a unit test enforces.
3. **Model** (`detector/model.py`): Isolation Forest trained on the first 21 days, tested on the last 9 days (time-based split, no leakage). Compared against a rolling z-score rule.
4. **Alerts** (`detector/alerts.py`): merges nearby flagged minutes, drops blips shorter than 8 minutes, and names the metric that moved most as the likely cause.
5. **API** (`api.py`): `POST /score` takes recent history and returns the newest minute's anomaly score.

## Results (test period: 9 days, 12,960 minutes, 17 incidents)

| Model | Precision | Recall | F1 | Incidents caught |
| --- | --- | --- | --- | --- |
| Rolling z-score (baseline) | 0.72 | 0.06 | 0.12 | 82% |
| **Isolation Forest** | 0.59 | 0.52 | 0.55 | **100%** |

After grouping: **22 alerts, 21 on real incidents (95% alert precision), all 17 incidents caught.**

Why the baseline's recall is so low: once a spike enters its rolling window, the spike inflates the window's standard deviation, so later minutes of the same incident stop looking unusual. It also misses slow memory leaks entirely. The trend and seasonal features fix both problems.

## Run it

```bash
pip install -r requirements.txt
python train.py                           # trains, evaluates, saves model.pkl
python -m unittest discover -s tests -v   # 9 tests
uvicorn api:app --reload                  # API at http://127.0.0.1:8000/docs
python demo_client.py                     # in a second terminal: scores a normal and an incident minute
```

## Design decisions

- **Isolation Forest**: incidents are rare and mostly unlabeled in real systems, so an unsupervised model fits. Labels are used only to evaluate.
- **Time-based split**: a random split would leak future patterns into training.
- **Seasonal baseline**: 3 pm traffic is normal at 3 pm but not at 3 am, so each minute is compared to the same time on previous days.
- **Incident-level metrics**: on-call engineers care whether each incident triggers one clear alert, not per-minute F1, so both are reported.

## Next steps

- Train on real metrics (Prometheus, Azure Monitor) or on the Telemetry Analytics Pipeline's warehouse
- Per-metric thresholds and severity levels
- Stream scoring with Kafka instead of request/response
