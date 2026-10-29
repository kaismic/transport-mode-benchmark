"""Shared portable formats; source runs are never modified."""
import hashlib
import json
from pathlib import Path

FEATURES = [f"{s}#{f}" for s, stats in (
    ("accelerometer", ["mean", "standard_deviation", "range", "minimum"]),
    ("gyroscope", ["mean", "standard_deviation", "range", "minimum"]),
    ("magnetometer", ["standard_deviation", "range", "minimum"])) for f in stats]
SENSORS = {"accelerometer": ["ax", "ay", "az"], "gyroscope": ["gx", "gy", "gz"],
           "magnetometer": ["mx", "my", "mz"]}
RATES = {"accelerometer": 30, "gyroscope": 4, "magnetometer": 2}
RUNTIME = {"provider": "CPUExecutionProvider", "intra_op_threads": 1,
           "inter_op_threads": 1, "execution_mode": "sequential",
           "graph_optimization": "all", "batch_size": 1, "spinning": False,
           "android_onnxruntime": "1.24.3"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")


def extract_window(rows, start, seconds):
    """Independent reference for the mobile replay contract (rows: ts + 9 axes)."""
    import numpy as np
    stop = start + seconds * 1000
    features = []
    for index, sensor in enumerate(SENSORS):
        values, times = [], []
        for row in rows:
            if not start <= row[0] < stop:
                continue
            vector = row[1 + 3 * index:4 + 3 * index]
            if any(v is None or not np.isfinite(v) for v in vector):
                continue
            times.append(row[0])
            values.append(float(np.linalg.norm(vector)))
        if len(times) < RATES[sensor] * seconds:
            return None
        if np.diff([start, *times, stop]).max() > 500:
            return None
        # Quality counts duplicates; features use the first finite vector at each timestamp.
        values = np.asarray(values)[np.r_[True, np.diff(times) > 0]]
        stats = [np.mean(values), np.std(values, ddof=0), np.ptp(values), np.min(values)]
        features.extend(stats if index != 2 else stats[1:])
    return np.asarray(features, dtype=np.float32).tolist()
