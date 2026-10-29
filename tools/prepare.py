"""Prepare trusted Sydney V1 runs without importing or modifying their code."""
import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

from common import FEATURES, RUNTIME, SENSORS, extract_window, read_json, sha256, write_json


def export_pipeline(pipeline):
    update_registered_converter(XGBClassifier, "XGBoostXGBClassifier",
                                calculate_linear_classifier_output_shapes, convert_xgboost,
                                options={"nocl": [True, False], "zipmap": [True, False]})
    # skl2onnx's SimpleImputer converter omits add_indicator. Build the fitted
    # preprocessing explicitly, including any all-empty columns dropped at fit.
    imputer = pipeline[0]
    kept = np.flatnonzero(np.isfinite(imputer.statistics_)).astype(np.int64)
    indicators = imputer.indicator_.features_.astype(np.int64)
    converted = convert_sklearn(pipeline[-1], initial_types=[("prepared", FloatTensorType([None, len(kept) + len(indicators)]))],
                               options={id(pipeline[-1]): {"zipmap": False}},
                               target_opset={"": 17, "ai.onnx.ml": 3})
    helper, tensor = onnx.helper, onnx.TensorProto
    initializers = [onnx.numpy_helper.from_array(kept, "bench_kept"),
                    onnx.numpy_helper.from_array(imputer.statistics_[kept].astype(np.float32), "bench_medians")]
    nodes = [helper.make_node("IsNaN", ["features"], ["bench_missing"]),
             helper.make_node("Gather", ["features", "bench_kept"], ["bench_values"], axis=1),
             helper.make_node("Gather", ["bench_missing", "bench_kept"], ["bench_kept_missing"], axis=1),
             helper.make_node("Where", ["bench_kept_missing", "bench_medians", "bench_values"], ["bench_imputed"])]
    if len(indicators):
        initializers.append(onnx.numpy_helper.from_array(indicators, "bench_indicators"))
        nodes += [helper.make_node("Gather", ["bench_missing", "bench_indicators"], ["bench_flags"], axis=1),
                  helper.make_node("Cast", ["bench_flags"], ["bench_float_flags"], to=tensor.FLOAT),
                  helper.make_node("Concat", ["bench_imputed", "bench_float_flags"], ["prepared"], axis=1)]
    else:
        nodes.append(helper.make_node("Identity", ["bench_imputed"], ["prepared"]))
    existing = list(converted.graph.node)
    del converted.graph.node[:]
    converted.graph.node.extend([*nodes, *existing])
    del converted.graph.input[:]
    converted.graph.input.append(helper.make_tensor_value_info("features", tensor.FLOAT, [None, len(FEATURES)]))
    converted.graph.initializer.extend(initializers)
    # Keep models readable by Android ORT 1.24 as well as the desktop validator.
    converted.ir_version = min(converted.ir_version, 10)
    onnx.checker.check_model(converted)
    return converted.SerializeToString()


def parity_inputs(frame):
    values = frame[FEATURES].to_numpy(dtype=np.float32)
    missing = np.repeat(values[:1], len(FEATURES) + 1, axis=0)
    for index in range(len(FEATURES)):
        missing[index, index] = np.nan
    missing[-1] = np.nan
    return np.concatenate([values, missing])


def verify(pipeline, model_bytes, values):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])
    frame = pd.DataFrame(values, columns=FEATURES)
    expected_labels, expected_probabilities = pipeline.predict(frame), pipeline.predict_proba(frame)
    labels, probabilities = session.run(None, {"features": values})
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_allclose(probabilities, expected_probabilities, rtol=1e-5, atol=1e-6)
    return expected_labels, expected_probabilities, float(np.max(np.abs(probabilities - expected_probabilities)))


def fixture_rows(values, labels, probabilities):
    return [{"input": [float(x) if np.isfinite(x) else None for x in row],
             "label": int(label), "probabilities": probability.tolist()}
            for row, label, probability in zip(values, labels, probabilities)]


def prepare_replay(run, output, seconds, session_ids):
    cohort = read_json(run / "cohort.json")["sessions"]
    replay = []
    for label, mode in enumerate(["bus", "car", "train"]):
        candidates = sorted((x for x in cohort if x["label"] == label), key=lambda x: x["session_id"])
        selected = None
        for item in candidates:
            if session_ids and item["session_id"] not in session_ids:
                continue
            event_path = run / "events" / f"{item['key']}.parquet"
            if sha256(event_path) != item["event_sha256"]:
                raise ValueError(f"Changed source event table: {event_path}")
            frame = pd.read_parquet(event_path)
            if frame.ts.max() - frame.ts.min() < seconds * 1000:
                continue
            start, end = int(frame.ts.min()), int(frame.ts.min()) + seconds * 1000
            frame = frame[frame.ts < end]
            columns = ["ts", *[c for group in SENSORS.values() for c in group]]
            rows = [[int(row[0]), *[float(v) if np.isfinite(v) else None for v in row[1:]]]
                    for row in frame[columns].itertuples(index=False, name=None)]
            golden = {}
            for window in [100, 120, 140]:
                golden[str(window)] = [{"start_ms": t, "end_ms": t + window * 1000,
                                         "features": extract_window(rows, t, window)}
                                        for t in range(start, end - window * 1000 + 1, window * 500)]
            if not all(any(x["features"] is not None for x in group) for group in golden.values()):
                continue
            selected = (item, rows, golden, start, end)
            break
        if selected is None:
            raise ValueError(f"No eligible {mode} replay session of {seconds}s with valid windows")
        item, rows, golden, start, end = selected
        relative = f"replay/{mode}.jsonl"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        golden_path = f"replay/{mode}-golden.json"
        write_json(output / golden_path, golden)
        replay.append({"session_id": item["session_id"], "mode": mode, "asset": relative,
                       "sha256": sha256(path), "source_event_sha256": item["event_sha256"],
                       "golden_asset": golden_path, "golden_sha256": sha256(output / golden_path),
                       "start_ms": start, "end_ms": end, "rows": len(rows)})
    return replay


def prepare(run, output, pipelines, replay_seconds=300, session_ids=None):
    run, output, pipelines = Path(run).resolve(), Path(output).resolve(), Path(pipelines).resolve()
    if run == output or run in output.parents or run == pipelines or run in pipelines.parents:
        raise ValueError("Outputs must be outside the immutable source run")
    output.mkdir(parents=True, exist_ok=True)
    # Remove the completion marker first: a partial export must never be benchmarkable.
    manifest_path = output / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    prepared, metadata = read_json(run / "prepared.json"), read_json(run / "final/metadata.json")
    for name, checksum in prepared["files"].items():
        if sha256(run / name) != checksum:
            raise ValueError(f"Changed prepared artifact: {name}")
    for name, checksum in read_json(run / "final/complete.json")["files"].items():
        if sha256(run / "final" / name) != checksum:
            raise ValueError(f"Changed final artifact: {name}")
    if sha256(run / "final/pipeline.joblib") != metadata["artifact_sha256"]:
        raise ValueError("Changed saved pipeline")
    if metadata["ordered_features"] != FEATURES:
        raise ValueError("This adapter requires the Sydney V1 feature contract")
    versions = {name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib", "scipy", "pyarrow"]}
    # Training artifacts are reconstructed using the original numerical stack.
    for name in versions:
        if versions[name] != metadata["dependency_versions"][name]:
            raise ValueError(f"{name} must be {metadata['dependency_versions'][name]}, got {versions[name]}")
    frames = {w: pd.read_parquet(run / f"features-{w}.parquet") for w in [100, 120, 140]}
    saved = joblib.load(run / "final/pipeline.joblib")  # trusted local run only
    print("Validating saved pipeline conversion first", flush=True)
    verify(saved, export_pipeline(saved), parity_inputs(frames[metadata["window_seconds"]]))
    split = read_json(run / "splits.json")
    models = []
    pipelines.mkdir(parents=True, exist_ok=True)
    for window, frame in frames.items():
        training = frame[frame.session_id.isin(split["final_train_sessions"])] if "final_train_sessions" in split else frame
        for seed in [42, 43, 44]:
            model_id = f"sydney-v1-w{window}-s{seed}"
            winner = metadata["winners"][str(window)][str(seed)]
            if window == metadata["window_seconds"] and seed == metadata["seed"]:
                pipeline = saved
                shutil.copyfile(run / "final/pipeline.joblib", pipelines / f"{model_id}.joblib")
            else:
                classifier = XGBClassifier(**winner["params"], objective="multi:softprob", eval_metric="mlogloss",
                                           tree_method="hist", device="cpu", random_state=seed,
                                           n_jobs=metadata["provenance"]["threads"])
                pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", classifier)])
                pipeline.fit(training[FEATURES], training.label,
                             model__sample_weight=compute_sample_weight("balanced", training.label))
                joblib.dump(pipeline, pipelines / f"{model_id}.joblib")
            values = parity_inputs(frame)
            model_bytes = export_pipeline(pipeline)
            labels, probabilities, max_error = verify(pipeline, model_bytes, values)
            model_path, fixture_path = f"models/{model_id}.onnx", f"fixtures/{model_id}.json"
            (output / "models").mkdir(exist_ok=True)
            (output / model_path).write_bytes(model_bytes)
            write_json(output / fixture_path, {"source_rows": len(frame), "rows": fixture_rows(values, labels, probabilities)})
            imputer = pipeline[0]
            models.append({"id": model_id, "family": "xgboost", "window_seconds": window, "step_seconds": window // 2,
                           "seed": seed, "asset": model_path, "sha256": sha256(output / model_path), "bytes": len(model_bytes),
                           "joblib_bytes": (pipelines / f"{model_id}.joblib").stat().st_size,
                           "joblib_sha256": sha256(pipelines / f"{model_id}.joblib"),
                           "fixture_asset": fixture_path, "fixture_sha256": sha256(output / fixture_path),
                           "ordered_features": FEATURES, "labels": ["bus", "car", "train"],
                           "input": {"name": "features", "dtype": "float32", "shape": [1, len(FEATURES)]},
                           "outputs": {"label": "label", "probabilities": "probabilities"},
                           "preprocessing": {"median": imputer.statistics_.tolist(),
                                             "indicator_indices": imputer.indicator_.features_.tolist()},
                           "winner": winner, "parity": {"rows": len(values), "max_absolute_error": max_error,
                                                         "rtol": 1e-5, "atol": 1e-6}})
            print(f"{model_id}: {len(model_bytes):,} bytes, parity {len(values)} rows, max error {max_error:.3g}", flush=True)
    replay = prepare_replay(run, output, replay_seconds, session_ids)
    manifest = {"schema_version": 1, "source_run": run.name, "source_prepared_sha256": sha256(run / "prepared.json"),
                "source_model_sha256": metadata["artifact_sha256"], "training_versions": versions,
                "export_versions": {p: importlib.metadata.version(p) for p in ["onnx", "onnxruntime", "skl2onnx", "onnxmltools"]},
                "runtime": RUNTIME, "independent_test_cohort": False, "models": models, "replay": replay,
                "replay_contract": {"schema": "sydney-sensor-v1", "columns": ["ts", *[c for g in SENSORS.values() for c in g]],
                                    "maximum_gap_ms": 500, "duplicate_policy": "first finite vector per timestamp",
                                    "std_ddof": 0, "feature_dtype": "float32"}}
    write_json(manifest_path, manifest)
    print(f"Complete: {manifest_path}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("assets/benchmark"))
    parser.add_argument("--pipelines", type=Path, default=Path("artifacts/pipelines"))
    parser.add_argument("--replay-seconds", type=int, default=300)
    parser.add_argument("--session-id", action="append", help="Restrict replay selection to these session IDs")
    args = parser.parse_args()
    if args.replay_seconds < 140:
        parser.error("--replay-seconds must be at least 140")
    prepare(args.run, args.output, args.pipelines, args.replay_seconds, args.session_id)
