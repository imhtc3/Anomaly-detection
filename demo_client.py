"""Send one normal window and one incident window to the running API."""
import json
import urllib.request

from detector.data import generate_metrics
from detector.model import MIN_HISTORY

df = generate_metrics()
df["timestamp"] = df["timestamp"].astype(str)
cols = ["timestamp", "requests", "cpu_pct", "latency_ms", "error_pct", "memory_pct"]
incident_end = df.index[(df["is_incident"] == 1) & (df.index > MIN_HISTORY)][5]
normal_end = df.index[(df["is_incident"] == 0) & (df.index > MIN_HISTORY + 5000)][0]

for name, end in [("normal minute", normal_end), ("incident minute", incident_end)]:
    window = df.loc[end - MIN_HISTORY + 1:end, cols].to_dict("records")
    req = urllib.request.Request("http://127.0.0.1:8000/score", method="POST",
                                 data=json.dumps({"points": window}).encode(),
                                 headers={"Content-Type": "application/json"})
    print(name, "->", json.loads(urllib.request.urlopen(req).read()))
