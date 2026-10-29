"""Recompute comparison tables exclusively from raw device exports."""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from common import read_json, write_json


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def describe(values):
    if not values:
        return {}
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": percentile(values, .95), "std": statistics.stdev(values) if len(values) > 1 else 0}


def csv_write(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def report(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    runs = []
    for path in source.rglob("*.json"):
        if output == path.parent or output in path.parents:
            continue
        value = read_json(path)
        if isinstance(value, dict) and "samples" in value and "models" in value:
            runs.append((path, value))
    if not runs:
        raise ValueError("No raw benchmark JSON exports found")
    identities = {(r["manifest_sha256"], r["device"]["manufacturer"], r["device"]["model"],
                   r["device"]["chipset"], r["device"]["android"], r["device"]["runtime_version"],
                   r["app_commit"], r["device"].get("device_id"),
                   r["device"].get("os_build_fingerprint")) for _, r in runs}
    if len(identities) != 1:
        raise ValueError("Do not combine different devices, builds, runtimes or manifests")
    if len({r["run_id"] for _, r in runs}) != len(runs):
        raise ValueError("Duplicate run_id; remove duplicate copies before reporting")
    output.mkdir(parents=True, exist_ok=True)
    groups, raw, memory = defaultdict(list), [], []
    excluded = []
    for path, run in runs:
        if run["status"] != "complete":
            excluded.append({"file": str(path), "status": run["status"], "error": run.get("error")})
            continue
        for sample in run["samples"]:
            raw.append({"run_id": run["run_id"], "mode": run["mode"], **sample})
        for sample in run["memory"]:
            memory.append({"run_id": run["run_id"], "mode": run["mode"], **sample})
        for model in run["models"]:
            groups[(model["id"], run["mode"])].append((run, model))
    rows = []
    for (model_id, mode), items in sorted(groups.items()):
        model = items[0][1]
        native, app, per_run_medians, processing, first, first_native, loading, load_app = [], [], [], [], [], [], [], []
        phase_memory = defaultdict(list)
        deltas = []
        for run, _ in items:
            current = [s for s in run["samples"] if s["model_id"] == model_id]
            warm = [s for s in current if s["kind"] == "warm"]
            native.extend(s["native_us"] for s in warm)
            app.extend(s["app_us"] for s in warm)
            if warm:
                per_run_medians.append(statistics.median(s["native_us"] for s in warm))
            first.extend(s["app_us"] for s in current if s["kind"] == "first")
            first_native.extend(s["native_us"] for s in current if s["kind"] == "first")
            loading.extend(s["native_us"] for s in current if s["kind"] == "load")
            load_app.extend(s["app_us"] for s in current if s["kind"] == "load")
            processing.extend(s["processing_us"] for s in current if s["kind"] == "replay" and s["accepted"])
            mem = [s for s in run["memory"] if s["model_id"] == model_id]
            for sample in mem:
                phase_memory[sample["phase"]].append(sample["pss_kib"])
            baselines = [s["pss_kib"] for s in mem if s["phase"] == "baseline"]
            if baselines:
                baseline = baselines[0]
                for sample in mem:
                    deltas.append({"run_id": run["run_id"], "model_id": model_id, "phase": sample["phase"],
                                   "pss_delta_kib": sample["pss_kib"] - baseline})
        official = (len(items) >= 5 and all(r["fresh_process"] and r["device"]["build_type"] == "release"
                    and r["app_commit"] not in ["unrecorded", "development"] and not r["interruptions"]
                    for r, _ in items))
        if mode == "latency":
            official = official and all(sum(s["kind"] == "warm" and s["model_id"] == model_id
                                           for s in r["samples"]) == 1000 for r, _ in items)
        row = {"model_id": model_id, "mode": mode, "window_seconds": model["window_seconds"], "seed": model["seed"],
               "model_bytes": model["bytes"], "model_sha256": model["sha256"],
               "process_runs": len(items), "protocol_complete": official}
        for name, values in [("native_us", native), ("app_us", app), ("first_app_us", first),
                             ("first_native_us", first_native), ("load_native_us", loading),
                             ("load_app_us", load_app), ("replay_processing_us", processing),
                             ("process_native_median_us", per_run_medians)]:
            row.update({f"{name}_{key}": value for key, value in describe(values).items()})
        for phase, values in phase_memory.items():
            row[f"{phase}_pss_kib_median"] = statistics.median(values)
            delta_values = [d["pss_delta_kib"] for d in deltas if d["phase"] == phase]
            if delta_values:
                row[f"{phase}_pss_delta_kib_median"] = statistics.median(delta_values)
        all_memory = [s["pss_kib"] for r, _ in items for s in r["memory"] if s["model_id"] == model_id]
        if all_memory:
            row["observed_peak_pss_kib"] = max(all_memory)
        rows.append(row)
        memory.extend({"mode": mode, **delta} for delta in deltas)
    csv_write(output / "comparison.csv", rows)
    # Preserve nested prediction/probability data as JSON cells in CSV.
    import json
    csv_write(output / "samples.csv", [{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in x.items()} for x in raw])
    csv_write(output / "memory.csv", memory)
    windows = []
    window_groups = defaultdict(list)
    for row in rows:
        window_groups[(row["mode"], row["window_seconds"])].append(row)
    for (mode, window), group in sorted(window_groups.items()):
        item = {"mode": mode, "window_seconds": window, "candidates": len(group), "seeds": [r["seed"] for r in group]}
        for key in ["native_us_median", "app_us_median", "replay_processing_us_median", "model_bytes", "observed_peak_pss_kib"]:
            values = [r[key] for r in group if key in r]
            if values:
                item.update({f"{key}_candidate_{k}": v for k, v in describe(values).items()})
        windows.append(item)
    csv_write(output / "windows.csv", windows)
    expected_ids = set().union(*(set(r.get("expected_model_ids", [m["id"] for m in r["models"]])) for _, r in runs))
    measured_ids = {row["model_id"] for row in rows}
    missing_ids = sorted(expected_ids - measured_ids)
    size_reports = [read_json(path) for path in source.rglob("apk-size.json") if output not in path.parents]
    write_json(output / "summary.json", {"schema_version": 1, "comparison": rows, "windows": windows,
                                         "excluded": excluded, "missing_model_ids": missing_ids, "apk_sizes": size_reports})
    lines = ["# Phone benchmark comparison", "", f"Device/build identity: `{next(iter(identities))}`", "",
             "Times are microseconds; memory is KiB. Peak memory is an observed sample maximum.", "",
             "| Model | Mode | ONNX bytes | Runs | Native median / p95 µs | App median µs | Protocol complete |",
             "|---|---|---:|---:|---:|---:|---|"]
    def number(row, key):
        return f"{row[key]:.2f}" if key in row else "—"
    for row in rows:
        lines.append(f"| {row['model_id']} | {row['mode']} | {row['model_bytes']} | {row['process_runs']} | "
                     f"{number(row, 'native_us_median')} / {number(row, 'native_us_p95')} | "
                     f"{number(row, 'app_us_median')} | {row['protocol_complete']} |")
    lines += ["", f"Model coverage: {len(measured_ids)}/{len(expected_ids)}. Missing: {', '.join(missing_ids) or 'none'}.",
              "", "Candidates differ in tuned hyperparameters as well as seeds. Window summaries give equal weight to each candidate.",
              "", "These full-cohort candidates have no independent accuracy score. Existing LOPO scores describe the training study.",
              "", f"Failed/cancelled exports excluded: {len(excluded)}. See summary.json.",
              "", "Protocol completeness requires five fresh release processes, recorded commits, no thermal interruptions, and (for latency) 1,000 warm calls per process."]
    for size in size_reports:
        lines += ["", f"APK: {size['apk_bytes']:,} bytes; runtime native libraries: "
                  f"{size['onnx_runtime_libraries']['compressed_bytes']:,} compressed bytes; "
                  f"bundled models: {size['models']['compressed_bytes']:,} compressed bytes."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/report"))
    args = parser.parse_args()
    report(args.input, args.output)
