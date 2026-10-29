import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from common import FEATURES, extract_window, read_json, sha256, write_json
from device import apk_size, schedule
from prepare import export_pipeline, verify
from report import describe, report

ROOT = Path(__file__).parents[1]


def test_export_preserves_fitted_indicators_and_empty_columns():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(90, 11)).astype(np.float32)
    values[:30, 0] = np.nan
    values[:, 1] = np.nan
    frame = pd.DataFrame(values, columns=FEATURES)
    labels = np.tile([0, 1, 2], 30)
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                         ("model", XGBClassifier(n_estimators=5, max_depth=2, n_jobs=1))])
    pipeline.fit(frame, labels)
    fixture = np.concatenate([values, np.full((1, 11), np.nan, dtype=np.float32)])
    verify(pipeline, export_pipeline(pipeline), fixture)


def test_window_quality_duplicates_and_boundaries():
    rows = [[t, 3., 4., 0., 0., 0., 2., 1., 0., 0.] for t in range(0, 1000, 20)]
    expected = [5., 0., 0., 5., 2., 0., 0., 2., 0., 0., 1.]
    np.testing.assert_allclose(extract_window(rows, 0, 1), expected)
    duplicate = [0, *[100.] * 9]
    assert extract_window([rows[0], duplicate, *rows[1:]], 0, 1) == expected
    assert extract_window([*rows, [1000, *[100.] * 9]], 0, 1) == expected
    assert extract_window(rows[:20], 0, 1) is None
    assert extract_window(rows[30:], 0, 1) is None


def test_schedule_is_reproducible_and_balanced():
    ids = [f"m{i}" for i in range(9)]
    tasks = schedule(ids, 5, 42)
    assert tasks == schedule(ids, 5, 42)
    for repeat in range(5):
        assert sorted(m for r, m in tasks if r == repeat) == ids
    assert tasks[:9] != tasks[9:18]


def sample_run(index):
    return {"schema_version": 1, "run_id": f"run{index}", "manifest_sha256": "manifest",
            "device": {"manufacturer": "test", "model": "phone", "chipset": "cpu", "android": "16",
                       "runtime_version": "1.24.3", "build_type": "release"},
            "app_commit": "abc123", "fresh_process": True, "mode": "latency", "status": "complete",
            "interruptions": [], "models": [{"id": "model", "window_seconds": 100, "seed": 42,
                                                "bytes": 123, "sha256": "model-hash"}],
            "samples": [{"kind": "warm", "model_id": "model", "native_us": i, "app_us": i + 2}
                        for i in range(1000)], "memory": []}


def test_report_recomputes_statistics_and_protocol(tmp_path):
    source = tmp_path / "raw"
    for index in range(5):
        write_json(source / f"r{index}.json", sample_run(index))
    rows = report(source, tmp_path / "report")
    assert rows[0]["native_us_median"] == 499.5
    assert rows[0]["native_us_p95"] == pytest.approx(949.05)
    assert rows[0]["protocol_complete"]
    changed = sample_run(5)
    changed["device"]["model"] = "other phone"
    write_json(source / "different.json", changed)
    with pytest.raises(ValueError, match="Do not combine"):
        report(source, tmp_path / "report")


def test_report_excludes_failures_and_rejects_duplicate_runs(tmp_path):
    run = sample_run(0)
    run["status"] = "cancelled"
    write_json(tmp_path / "input/r0.json", run)
    assert report(tmp_path / "input", tmp_path / "out") == []
    write_json(tmp_path / "input/duplicate.json", run)
    with pytest.raises(ValueError, match="Duplicate"):
        report(tmp_path / "input", tmp_path / "out")


def test_memory_report_requires_baseline_and_warmed_samples(tmp_path):
    for index in range(5):
        run = sample_run(index)
        run["mode"] = "memory"
        run["samples"] = []
        write_json(tmp_path / "input" / f"r{index}.json", run)
    assert not report(tmp_path / "input", tmp_path / "out")[0]["protocol_complete"]
    for index in range(5):
        path = tmp_path / "input" / f"r{index}.json"
        run = read_json(path)
        run["memory"] = [{"model_id": "model", "phase": phase, "pss_kib": pss}
                         for phase, pss in [("baseline", 100), ("loaded", 110), ("warmed", 120), ("inference_end", 120)]]
        write_json(path, run)
    row = report(tmp_path / "input", tmp_path / "out")[0]
    assert row["protocol_complete"]
    assert row["warmed_pss_delta_kib_median"] == 20
    import csv
    with (tmp_path / "out/memory.csv").open() as stream:
        samples = list(csv.DictReader(stream))
    assert len(samples) == 20
    assert {s["pss_delta_kib"] for s in samples} == {"0", "10", "20"}


def test_packaged_size_categories(tmp_path):
    import zipfile
    path = tmp_path / "app.apk"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("lib/arm64-v8a/libonnxruntime.so", b"runtime")
        archive.writestr("assets/flutter_assets/assets/benchmark/models/model.onnx", b"model")
    result = apk_size(path)
    assert result["onnx_runtime_libraries"]["uncompressed_bytes"] == 7
    assert result["models"]["uncompressed_bytes"] == 5


@pytest.mark.skipif(not (ROOT / "assets/benchmark/manifest.json").exists(), reason="Run prepare.py first")
def test_prepared_nine_models_and_source_replay_parity():
    bundle = ROOT / "assets/benchmark"
    manifest = read_json(bundle / "manifest.json")
    assert {(m["window_seconds"], m["seed"]) for m in manifest["models"]} == {
        (w, s) for w in [100, 120, 140] for s in [42, 43, 44]}
    for model in manifest["models"]:
        assert sha256(bundle / model["asset"]) == model["sha256"]
        assert sha256(bundle / model["fixture_asset"]) == model["fixture_sha256"]
        fixtures = read_json(bundle / model["fixture_asset"])
        values = np.array([[np.nan if x is None else x for x in r["input"]] for r in fixtures["rows"]], dtype=np.float32)
        import onnxruntime as ort
        session = ort.InferenceSession(str(bundle / model["asset"]), providers=["CPUExecutionProvider"])
        labels, probabilities = session.run(None, {"features": values})
        np.testing.assert_array_equal(labels, [r["label"] for r in fixtures["rows"]])
        np.testing.assert_allclose(probabilities, [r["probabilities"] for r in fixtures["rows"]], rtol=1e-5, atol=1e-6)
    # Compare exported replay against the ORIGINAL extractor, not the exporter helper.
    source = ROOT.parents[1] / "ml/sydney-tmd-v1"
    if not source.exists():
        return
    sys.path.insert(0, str(source))
    from sydney_tmd.data import extract
    from common import SENSORS
    for item in manifest["replay"]:
        rows = [json.loads(line) for line in (bundle / item["asset"]).read_text().splitlines()]
        columns = ["ts", *[c for g in SENSORS.values() for c in g]]
        frame = pd.DataFrame(rows, columns=columns)
        # Append an invalid boundary event so original endpoint logic includes the declared cap.
        frame.loc[len(frame)] = [item["end_ms"], *[np.nan] * 9]
        golden = read_json(bundle / item["golden_asset"])
        for window in [100, 120, 140]:
            original, _ = extract(frame, window)
            indexed = {int(r.window_start_ms): r for r in original.itertuples()}
            for expected in golden[str(window)]:
                start = expected["start_ms"]
                assert (start in indexed) == (expected["features"] is not None)
                if start in indexed:
                    found = original[original.window_start_ms == start][FEATURES].to_numpy()[0]
                    np.testing.assert_allclose(found, expected["features"], rtol=1e-5, atol=1e-6)
