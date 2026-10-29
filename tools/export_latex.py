"""Export a self-contained LaTeX account of every benchmark summary statistic.

The 99-run study is kept separate from diagnostics and subsequent APK builds.
Raw per-call/per-snapshot records remain available in the companion CSV archive.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
import zipfile

from common import read_json, sha256, write_json
from report import describe

MODES = {"latency": "Inference latency", "memory": "Inference memory",
         "replay_fast": "Accelerated replay", "replay_fast_memory": "Accelerated replay memory",
         "replay_realtime": "Real-time replay"}
METRICS = {"native_us": "Native inference", "app_us": "Full Flutter inference call",
           "first_app_us": "First inference: full Flutter call",
           "first_native_us": "First inference: native execution",
           "load_native_us": "Native asset read and session creation",
           "load_app_us": "Full Flutter model load",
           "replay_processing_us": "Accelerated replay processing",
           "process_native_median_us": "Between-process native medians",
           "model_bytes": "Model size (bytes)", "observed_peak_pss_kib": "Observed peak PSS (KiB)"}
STATS = ["count", "mean", "median", "p95", "std"]


def tex(value):
    translations = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_\allowbreak{}", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}"}
    escaped = "".join(translations.get(c, c) for c in str(value))
    escaped = escaped.replace("/", r"/\allowbreak{}").replace(":", r":\allowbreak{}")
    return re.sub(r"[a-f0-9]{32,}", lambda m: r"\allowbreak{}".join(m[0][i:i+16] for i in range(0, len(m[0]), 16)), escaped)


def number(value, places=3):
    if value is None:
        return "---"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if value != 0 and abs(value) < .001:
        mantissa, power = f"{value:.8e}".split("e")
        return rf"\({mantissa}\times10^{{{int(power)}}}\)"
    return f"{value:,.{places}f}"


class Document:
    def __init__(self):
        self.lines = []
        self.numeric_cells = 0
        self.table_count = 0

    def add(self, value=""):
        self.lines.append(value)

    def paragraph(self, value):
        self.add(tex(value) + "\n")

    def section(self, title, level="section"):
        self.add(rf"\{level}{{{tex(title)}}}")

    def table(self, caption, headers, rows, align=None, note=None):
        rows = list(rows)
        if not rows:
            return
        self.table_count += 1
        align = align or ("l" + "r" * (len(headers) - 1))
        self.add(r"\begingroup\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}")
        self.add(rf"\begin{{longtable}}{{@{{}}{align}@{{}}}}")
        self.add(rf"\caption{{{tex(caption)}}}\\")
        header = " & ".join(rf"\textbf{{{tex(x)}}}" for x in headers) + r" \\"
        self.add(r"\toprule" + "\n" + header + "\n" + r"\midrule\endfirsthead")
        self.add(r"\multicolumn{" + str(len(headers)) + r"}{l}{\small\itshape Continued from previous page}\\")
        self.add(r"\toprule" + "\n" + header + "\n" + r"\midrule\endhead")
        self.add(r"\midrule\multicolumn{" + str(len(headers)) + r"}{r}{\small Continued on next page}\\\endfoot")
        self.add(r"\bottomrule\endlastfoot")
        for row in rows:
            self.numeric_cells += sum(isinstance(x, (int, float)) for x in row)
            self.add(" & ".join(number(x) if x is None or isinstance(x, (int, float)) else tex(x) for x in row) + r" \\")
        self.add(r"\end{longtable}\endgroup")
        if note:
            self.add(r"\noindent\small " + tex(note) + r"\normalsize\par\medskip")

    def key_values(self, caption, pairs):
        self.table(caption, ["Item", "Recorded value"], pairs,
                   align=r">{\raggedright\arraybackslash}p{.34\textwidth}>{\raggedright\arraybackslash}p{.61\textwidth}")


def load_runs(directory):
    runs = []
    for path in sorted(Path(directory).rglob("*.json")):
        value = read_json(path)
        if isinstance(value, dict) and "samples" in value and "models" in value:
            runs.append(value)
    return runs


def definitions(doc):
    """Reader-facing glossary, kept with the exporter so regeneration preserves it."""
    doc.section("Definitions and abbreviations")
    doc.paragraph("This reference explains the terms used in the result tables. A statistic must be read together with its units, sample count and table caption: the observations may be inference calls, process medians, memory snapshots or fitted candidates.")

    def glossary(caption, entries):
        doc.table(caption, ["Term", "Definition and interpretation"], entries,
                  align=r">{\raggedright\arraybackslash}p{.24\textwidth}>{\raggedright\arraybackslash}p{.71\textwidth}")

    doc.section("Statistics and table notation", "subsection")
    glossary("Statistical definitions", [
        ("N / count", "Number of observations used for a statistic. Each per-model warm-latency table has N=5,000 calls from five processes; the between-process-median table has N=5. A larger N does not by itself mean more independent models or participants."),
        ("Mean", "Arithmetic average: sum the observations and divide by N. Unusually slow calls can pull the mean upward."),
        ("Median / p50", "Middle value of the ordered observations, averaging the two middle values when N is even. Approximately half the observations lie at or below the median."),
        ("p95", "95th percentile. Approximately 95% of observations are at or below this value and 5% are higher. For latency it describes the slower end of the measured calls. Ties, interpolation and rounding affect the exact proportion."),
        ("SD / std", "Sample standard deviation: a measure of spread around the mean, in the same units as the observations. This report uses denominator N-1. A larger SD means more dispersion; it is not the standard error or a confidence interval."),
        ("N=1, SD=0", "For a single observation, sample SD cannot be estimated. The exporter prints zero by convention; this does not demonstrate that the operation has no variability."),
        ("Per-model vs window p95", "Per-model latency p95 is calculated across 5,000 measured calls. Window-summary p95 is calculated across the three seed candidates' summary values. These percentiles describe different sets of observations."),
        ("Confidence interval (CI)", "An interval produced by a statistical method to quantify uncertainty in an estimated quantity. A p95 is a distribution percentile, not a 95% confidence interval. This benchmark report does not calculate confidence intervals."),
        ("--- in a table", "No value is available or the metric does not apply. A dash does not represent zero."),
    ])
    doc.paragraph("Example: a latency p95 of 0.224 ms means approximately 95% of measured calls finished within 0.224 ms. It does not promise that every call finishes within this time. For 5,000 calls, the slowest 5% corresponds to about 250 calls.")
    doc.paragraph("The exporter uses linear interpolation at zero-based position 0.95(N-1). Equivalently, order values from x(1) to x(N) using one-based ranks. With h=1+0.95(N-1), j=floor(h), and g=h-j, the percentile is:")
    doc.add(r"\[\operatorname{p95}=(1-g)x_{(j)}+g x_{(\min(j+1,N))}.\]")
    doc.paragraph("For N=5,000, h=4,750.05, so the result is 95% of the 4,750th ordered value plus 5% of the 4,751st. For three candidate summaries, h=2.9, so their p95 is 10% of the middle value plus 90% of the largest. Neither calculation is a maximum or a confidence bound.")

    doc.section("Memory measurements", "subsection")
    glossary("Memory definitions", [
        ("RAM", "Random-access memory: the device's working memory. The device RAM figure is distinct from memory attributed to this app."),
        ("PSS", "Proportional Set Size. Counts memory resident for the process, assigning shared pages in proportion to the number of processes sharing them. It is useful for attributing physical-memory cost without charging every process the full shared page."),
        ("RSS", "Resident Set Size. Counts the process's resident memory, including shared pages at their full size. Summing RSS across processes can count the same shared page more than once."),
        ("Native PSS", "Android's native-heap PSS category, read from Debug.MemoryInfo.nativePss. It is one part of total PSS, not a measurement of every native allocation or all native-library code pages."),
        ("Dalvik PSS", "Android's managed-runtime heap PSS category, read from Debug.MemoryInfo.dalvikPss. The historical name remains in the API even on Android's ART runtime. It does not represent all Flutter/Dart memory."),
        ("Baseline", "Snapshot after initializing the global runtime and loading the prepared input/expected fixtures, with no model session loaded. It already includes substantial benchmark-harness memory."),
        ("Loaded", "Snapshot after model asset reading and inference-session creation, before the first prediction and validation calls."),
        ("Warmed", "Snapshot after parity validation and, for inference runs, 100 explicit warmup calls. Replay warms the model through validation before playback."),
        ("Observed / end", "Observed snapshots are taken during inference or replay. End snapshots are taken after that workload; replay end is recorded for each session. These phase labels do not imply continuous memory tracing."),
        ("Delta / incremental PSS", "A phase snapshot minus the same process's baseline. Deltas are calculated within each process before aggregation. They estimate extra process cost and include runtime/validation/harness effects, not only model weights."),
        ("Observed peak PSS", "Largest PSS value among the collected snapshots. A brief higher allocation between snapshots may be missed."),
        ("High-water RSS", "Largest RSS reached during the process lifetime so far, as reported by the operating system. It includes earlier initialization and validation and is not a per-phase incremental peak."),
    ])
    doc.paragraph("Illustrative example: if a process has 4 MiB of private resident memory and shares another 10 MiB equally with one other process, its RSS is 14 MiB and its PSS is 4+10/2=9 MiB. Separately, a hypothetical baseline PSS of 120 MiB and warmed PSS of 150 MiB give a within-process delta of 30 MiB. These examples are explanatory, not additional phone measurements.")
    doc.add(r"\noindent Memory terminology and category names follow the \href{https://developer.android.com/tools/dumpsys\#meminfo}{Android memory inspection documentation} and the \href{https://developer.android.com/reference/android/os/Debug.MemoryInfo}{Debug.MemoryInfo API reference}.\par")

    doc.section("Timing and experimental procedure", "subsection")
    glossary("Timing and benchmark definitions", [
        ("Inference / prediction", "Applying a fitted model to an input feature vector to produce a class label and class probabilities."),
        ("Latency", "Elapsed time to perform an operation. The native, full Flutter, load and replay timers cover different operations; their captions define the boundary."),
        ("Native inference", "The timed Java/JNI ONNX Runtime session call using an already constructed input tensor. Tensor creation, output decoding/disposal and Flutter-channel transfer are outside this timer."),
        ("Full Flutter call", "Elapsed time from requesting inference in Dart until the native result returns, including channel transfer, tensor creation, execution, output decoding and disposal. Feature extraction happens separately."),
        ("Load / first inference", "Load covers asset reading and session creation with the global runtime already initialized. First inference is the first prediction after loading. Neither guarantees uncached storage reads because prior checksum checks and OS caches can affect them."),
        ("Warmup / warm inference", "Warmup calls exercise the loaded model before collecting the main timing samples. Warm inference reuses the existing session. This study performs 100 explicit warmups after validation before 1,000 measured calls per inference process."),
        ("Batch size 1", "One feature vector is classified per inference call. The benchmark does not measure large-batch throughput."),
        ("Process repetition", "A separate app-process launch for a candidate. Repeated calls within one process are different from independent process repetitions. The latency and memory studies use five fresh processes per candidate."),
        ("Replay", "Processing recorded sensor events. Accelerated replay runs without original waiting intervals; real-time replay follows event timestamp spacing. Neither is live sensor acquisition."),
        ("Decode / buffer / feature", "Decode covers reading/decoding recorded input. Buffer work retains and removes events for overlapping windows. Feature extraction computes the ordered statistical inputs supplied to the model."),
        ("Replay processing", "Buffer work since the preceding window, quality checks/feature extraction and the full inference call. Decode and waiting are recorded separately or excluded, as specified in the report."),
        ("Schedule lag", "How late a replay prediction finishes relative to its window's expected completion time. It includes processing and scheduling delays; it is distinct from native inference latency."),
        ("Window W / step", "W is the duration of sensor history used to compute features. Step is the interval between successive window starts. Here W=100/120/140 seconds and step=50/60/70 seconds."),
        ("Observation delay", "Time required to collect a full valid sensor window before the first result. It is separate from millisecond-scale processing and may be longer when quality checks reject a window."),
        ("Seed / candidate", "A seed controls random choices in tuning/training. A candidate is one fitted window/seed model. These candidates have different winning hyperparameters, so seed comparisons do not isolate randomness alone."),
        ("Protocol complete", "The report's completeness flag: at least five fresh release processes, recorded code versions and no recorded thermal interruptions, plus the required call counts or memory phases. It is not an accuracy guarantee or a claim of population representativeness."),
    ])

    doc.section("Units, formats and software abbreviations", "subsection")
    glossary("Units and abbreviations", [
        ("s / ms / us", "Seconds / milliseconds / microseconds. The ASCII suffix us denotes microseconds: 1 second=1,000 ms=1,000,000 us, and 1 ms=1,000 us."),
        ("Byte / KiB / MiB", "A byte is 8 bits. KiB means kibibyte (1,024 bytes); MiB means mebibyte (1,048,576 bytes). These binary units differ from decimal kB (1,000 bytes) and MB (1,000,000 bytes)."),
        ("Hz", "Hertz, meaning events or cycles per second. In the quality checks it is the sensor sampling rate, not the model prediction rate."),
        ("Float32", "A 32-bit floating-point numerical representation, used for the model's input features. It does not imply the entire app or all intermediate calculations use 32-bit arithmetic."),
        ("CPU / execution provider", "Central Processing Unit / the ONNX Runtime backend that executes model operators. This study uses CPUExecutionProvider and does not measure GPU or neural-accelerator performance."),
        ("ONNX / ORT", "Open Neural Network Exchange / ONNX Runtime. ONNX is the exported model representation; ONNX Runtime is the software that loads and executes it."),
        ("APK", "Android application package: the .apk file installed on the phone. It includes app/runtime code and bundled models and fixtures, so APK size is different from model graph size."),
        ("ABI / ARM64", "Application Binary Interface / the 64-bit ARM architecture targeted by this APK. An ABI specifies binary compatibility conventions for the device and native libraries."),
        ("SDK / API", "Software Development Kit / Application Programming Interface. The Android SDK number recorded here is the platform API level; it is distinct from the device's reported Android release string."),
        ("JNI / platform channel", "Java Native Interface / Flutter's Dart-to-Android message bridge. JNI connects Java/Kotlin-side runtime calls to native code; the Flutter platform channel adds another layer included in the full-call timer."),
        ("Joblib", "Python object-serialization format used for the saved training pipeline. Its byte size includes Python-side pipeline state and is reported separately from the deployed ONNX artifact."),
        ("XGBoost", "Extreme Gradient Boosting, the boosted-tree classifier family used for all nine candidates."),
        ("Macro F1", "Unweighted mean of per-class F1 scores. Each F1 combines precision P and recall R as 2PR/(P+R). Scores range from 0 to 1, with larger values better. The saved inner macro F1 values in this document are tuning-selection scores, not independent deployment accuracy."),
        ("Parity / rtol / atol", "Parity is agreement with the reference implementation. Labels must match exactly; probabilities pass when the absolute difference is no greater than atol + rtol times the absolute reference value. Here atol=1e-6 and rtol=1e-5."),
        ("SHA-256 / commit", "A content checksum / a recorded Git source revision. Checksums identify the exact input/model bytes; the commit identifies the source version used to build the measured app."),
    ])
    doc.add(r"\clearpage")


def complete_summary_tables(doc, summary):
    """Render every numerical leaf from comparison/windows, with a coverage check."""
    coverage = set()
    comparison = summary["comparison"]
    doc.section("Run coverage and protocol status")
    doc.table("All candidate/mode result groups", ["Window s", "Seed", "Mode", "Runs", "ONNX bytes", "Complete"],
              [[r["window_seconds"], r["seed"], {"latency": "Latency", "memory": "Memory", "replay_fast": "Replay"}[r["mode"]],
                r["process_runs"], r["model_bytes"], r["protocol_complete"]] for r in comparison],
              note="Complete is the exporter's protocol-completeness flag. Replay has one process per candidate and is functional validation; latency and inference memory each have five.")
    for i, row in enumerate(comparison):
        coverage.update(("comparison", i, k) for k in ["window_seconds", "seed", "process_runs", "model_bytes", "protocol_complete"])
    doc.paragraph(f"Missing candidate IDs: {summary['missing_model_ids'] or 'none'}. Failed/cancelled exports in the successful study report: {len(summary['excluded'])}.")
    for mode in ["latency", "memory", "replay_fast"]:
        doc.section(MODES[mode])
        selected = [(i, r) for i, r in enumerate(comparison) if r["mode"] == mode]
        prefixes = sorted({k.removesuffix("_count") for _, r in selected for k in r if k.endswith("_count")})
        for prefix in prefixes:
            group = [(i, r) for i, r in selected if prefix + "_count" in r]
            rows = []
            for i, r in group:
                rows.append([r["window_seconds"], r["seed"], *[r[prefix + "_" + s] for s in STATS]])
                coverage.update(("comparison", i, prefix + "_" + s) for s in STATS)
            caption = METRICS.get(prefix, prefix)
            doc.table(caption + " (microseconds)", ["Window s", "Seed", "N", "Mean", "Median", "p95", "SD"], rows,
                      note="N is the number of observations. SD uses the sample standard deviation (N-1); p95 uses linear interpolation. Between-process medians have N=5, rather than N=5,000 inference calls." if prefix == "process_native_median_us" else None)
        if mode == "memory":
            memory_rows = []
            for i, r in selected:
                phases = [k.removesuffix("_pss_kib_median") for k in r if k.endswith("_pss_kib_median")]
                for phase in phases:
                    absolute, delta = phase + "_pss_kib_median", phase + "_pss_delta_kib_median"
                    memory_rows.append([r["window_seconds"], r["seed"], phase.replace("_", " "), r[absolute], r[delta]])
                    coverage.update({("comparison", i, absolute), ("comparison", i, delta)})
            doc.table("Published PSS phase medians and within-process changes (KiB)",
                      ["Window s", "Seed", "Phase", "PSS median", "Delta median"], memory_rows,
                      note="Each delta is computed against that process's baseline before aggregation. The median delta need not equal the difference of separately pooled medians.")
            doc.table("Observed maximum whole-process PSS (KiB)", ["Window s", "Seed", "Observed peak"],
                      [[r["window_seconds"], r["seed"], r["observed_peak_pss_kib"]] for _, r in selected])
            coverage.update(("comparison", i, "observed_peak_pss_kib") for i, _ in selected)
    doc.section("Window summaries across three fitted candidates")
    doc.paragraph("These summaries weight each candidate equally. The three candidates at a window have seeds 42, 43 and 44 and different winning hyperparameters. A p95 in this section is across three candidate summaries, not a tail inference latency or confidence bound.")
    windows = summary["windows"]
    doc.table("Window-group membership", ["Mode", "Window s", "Candidates", "Seeds"],
              [[{"latency": "Latency", "memory": "Memory", "replay_fast": "Replay"}[r["mode"]], r["window_seconds"],
                r["candidates"], ", ".join(map(str, r["seeds"]))] for r in windows])
    for i, r in enumerate(windows):
        coverage.update(("windows", i, k) for k in ["window_seconds", "candidates"])
    prefixes = sorted({k.removesuffix("_candidate_count") for r in windows for k in r if k.endswith("_candidate_count")})
    for prefix in prefixes:
        group = [(i, r) for i, r in enumerate(windows) if prefix + "_candidate_count" in r]
        rows = []
        for i, r in group:
            rows.append([{"latency": "Latency", "memory": "Memory", "replay_fast": "Replay"}[r["mode"]], r["window_seconds"],
                         *[r[prefix + "_candidate_" + s] for s in STATS]])
            coverage.update(("windows", i, prefix + "_candidate_" + s) for s in STATS)
        doc.table("Candidate-level distribution: " + prefix.replace("_", " "),
                  ["Mode", "Window s", "N", "Mean", "Median", "p95", "SD"], rows)
    missing = [(collection, i, key) for collection in ["comparison", "windows"]
               for i, row in enumerate(summary[collection]) for key, value in row.items()
               if isinstance(value, (int, float)) and (collection, i, key) not in coverage]
    if missing:
        raise ValueError(f"Numerical summary fields not rendered: {missing}")
    return len(coverage)


def raw_memory_tables(doc, runs, title):
    doc.section(title, "subsection")
    phases = defaultdict(list)
    for run in runs:
        for sample in run["memory"]:
            model = next(m for m in run["models"] if m["id"] == sample["model_id"])
            phases[(model["window_seconds"], model["seed"], sample["phase"])].append(sample)
    for metric, title in [("pss_kib", "Total PSS"), ("rss_kib", "Resident set size"),
                          ("native_pss_kib", "Native PSS"), ("dalvik_pss_kib", "Dalvik PSS")]:
        rows = []
        for (window, seed, phase), samples in sorted(phases.items()):
            values = [s[metric] for s in samples if s.get(metric) is not None]
            if values:
                stats = describe(values)
                rows.append([window, seed, phase.replace("inference_observed", "observed").replace("_", " "), *[stats[s] for s in STATS]])
        doc.table(title + " by phase (KiB; recomputed from snapshots)",
                  ["Window s", "Seed", "Phase", "N", "Mean", "Median", "p95", "SD"], rows)
    if phases:
        doc.paragraph("Resident-set high-water values are lifetime process maxima and include earlier validation. They are not instantaneous incremental model peaks.")
        grouped = defaultdict(list)
        for (window, seed, _), samples in phases.items():
            grouped[(window, seed)].extend(s.get("process_high_water_rss_kib") for s in samples if s.get("process_high_water_rss_kib") is not None)
        doc.table("Lifetime high-water RSS extrema observed (KiB)", ["Window s", "Seed", "Snapshots", "Minimum", "Maximum"],
                  [[w, s, len(v), min(v), max(v)] for (w, s), v in sorted(grouped.items())])


def diagnostic_tables(doc, label, runs):
    if not runs:
        return
    doc.section(label, "subsection")
    doc.paragraph("Diagnostic results below are kept separate from the 99-run study. They do not satisfy a five-process benchmark protocol.")
    for run in runs:
        model = run["models"][0]
        doc.paragraph(f"Model: {model['id']}; mode: {run['mode']}; status: {run['status']}.")
        doc.paragraph(f"Source APK commit: {run['app_commit']}. Result time (UTC): {run['run_id']}.")
        distributions = defaultdict(list)
        for sample in run["samples"]:
            for key in ["native_us", "app_us", "processing_us", "decode_us", "buffer_us", "feature_us", "schedule_lag_us"]:
                if sample.get(key) is not None:
                    distributions[(sample["kind"], key)].append(sample[key])
            if sample.get("prediction"):
                for key in ["native_us", "app_us"]:
                    distributions[("replay prediction", key)].append(sample["prediction"][key])
        doc.table("Diagnostic timing distributions (microseconds)", ["Operation", "Metric", "N", "Mean", "Median", "p95", "SD"],
                  [[kind, metric.replace("_us", "").replace("_", " "), *[describe(v)[s] for s in STATS]]
                   for (kind, metric), v in sorted(distributions.items())])
        if run["mode"] == "replay_realtime":
            starts = {s["session_id"]: s["start_ms"] for s in run["replay"]}
            replay = [s for s in run["samples"] if s["kind"] == "replay"]
            doc.table("Real-time prediction boundaries and lag", ["Boundary s", "Lag ms", "Processing ms", "Native us", "App us", "Rows max"],
                      [[(s["window_end_ms"] - starts[s["session_id"]]) / 1000, s["schedule_lag_us"] / 1000,
                        s["processing_us"] / 1000, s["prediction"]["native_us"], s["prediction"]["app_us"], s["max_buffer_rows"]] for s in replay])
            doc.table("Real-time predicted probabilities (class order: bus, car, train)",
                      ["Boundary s", "Label", "Bus", "Car", "Train"],
                      [[(s["window_end_ms"] - starts[s["session_id"]]) / 1000, s["prediction"]["label"],
                        *[f"{v:.10f}" for v in s["prediction"]["probabilities"]]] for s in replay])
            doc.paragraph("The run was deliberately cancelled after two boundaries. It is a pacing/cancellation check, not a completed 15-minute trial. Decode totals in the raw record are charged to the next emitted window; early buffering measurements used integer-microsecond rounding per event.")
        if run["memory"]:
            raw_memory_tables(doc, [run], "Diagnostic process memory")


def export(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    summary = read_json(project / "artifacts/pixel4a-report/summary.json")
    manifest = read_json(project / "assets/benchmark/manifest.json")
    runs = load_runs(project / "results/pixel4a-study")
    if len(runs) != 99 or any(r["status"] != "complete" for r in runs):
        raise ValueError("Expected the complete 99-run study")
    identity = {(r["manifest_sha256"], r["app_commit"], r["device"]["device_id"]) for r in runs}
    if len(identity) != 1:
        raise ValueError("Study contains mixed benchmark identities")
    if {r["manifest_sha256"] for r in runs} != {sha256(project / "assets/benchmark/manifest.json")}:
        raise ValueError("Manifest differs from measured bundle")
    doc = Document()
    doc.add(r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm,headheight=14pt]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,booktabs,longtable,array,pdflscape,amsmath,fancyhdr,lastpage}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\LTpre}{6pt}\setlength{\LTpost}{6pt}
\setlength{\emergencystretch}{3em}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{Sydney TMD -- Smartphone Benchmark}
\fancyhead[R]{29 October 2026}
\fancyfoot[C]{\thepage\ / \pageref{LastPage}}
\hypersetup{pdftitle={Sydney TMD: complete smartphone benchmark results},pdfauthor={TCCT project}}
\begin{document}
\hypersetup{pageanchor=false}
\begin{titlepage}
\vspace*{20mm}
{\Large TCCT / TRANSPORT MODE DETECTION\par}
\vspace{9mm}
{\huge\bfseries Complete Smartphone\\Benchmark Results\par}
\vspace{6mm}
{\Large Nine Sydney V1 XGBoost deployment candidates\par}
\vspace{4mm}
{\large Pixel 4a -- 29 October 2026\par}
\vspace{12mm}
""")
    doc.paragraph("This document exports every numerical aggregate in the benchmark comparison and window-summary tables, with model-size/parity details, complete timing distributions, memory phases, replay measurements, APK accounting, model configuration and separately labelled diagnostic checks. Full-precision aggregates and all per-call/per-snapshot records accompany the document in the data archive.")
    doc.paragraph("Primary study: 99 successful fresh-process runs, comprising 45 latency runs, 45 inference-memory runs and nine accelerated replay runs. Latency and memory each have five processes per fitted candidate. The replay study has one process per candidate and is functional validation only.")
    lat = [r for r in summary["comparison"] if r["mode"] == "latency"]
    mem = [r for r in summary["comparison"] if r["mode"] == "memory"]
    doc.paragraph(f"Across candidates, median native inference ranges from {min(r['native_us_median'] for r in lat)/1000:.3f} to {max(r['native_us_median'] for r in lat)/1000:.3f} ms. Median full Flutter calls range from {min(r['app_us_median'] for r in lat)/1000:.3f} to {max(r['app_us_median'] for r in lat)/1000:.3f} ms. Warmed whole-process PSS ranges from {min(r['warmed_pss_kib_median'] for r in mem)/1024:.2f} to {max(r['warmed_pss_kib_median'] for r in mem)/1024:.2f} MiB.")
    doc.paragraph("These figures describe this handset and software configuration. Whole-process memory includes the benchmark harness. There is no independent accuracy score for the full-cohort deployment artifacts.")
    doc.add(r"\vfill\textbf{Units:} 1 KiB = 1,024 bytes; 1 MiB = 1,048,576 bytes. Timing tables use microseconds unless explicitly marked otherwise.\end{titlepage}\hypersetup{pageanchor=true}")
    doc.add(r"\tableofcontents\clearpage")
    definitions(doc)
    doc.section("Study setup and interpretation")
    device = runs[0]["device"]
    doc.key_values("Device and fixed execution settings", [
        ("Device / chipset", device["manufacturer"] + " " + device["model"] + " / " + device["chipset"]),
        ("Reported Android / SDK", f"{device['android']} / {device['sdk']}"),
        ("Reported OS fingerprint", device["os_build_fingerprint"]),
        ("RAM reported by Android (bytes)", device["ram_bytes"]),
        ("App / Flutter / ONNX Runtime", f"{device['app_version']} / {runs[0]['flutter_version']} / {device['runtime_version']}"),
        ("APK source commit", runs[0]["app_commit"]),
        ("Build / ABI", "Release / ARM64"), ("Provider", "CPUExecutionProvider"),
        ("Batch / intra-op / inter-op threads", "1 / 1 / 1"),
        ("Execution / graph optimization / spinning", "Sequential / all / disabled"),
        ("Explicit inference warmups / measured calls", "100 / 1,000 per process"),
        ("Repetitions / shuffle seed / cooldown", "5 / 42 / 2 seconds"),
        ("Replay sessions / duration", "3 sessions, 300 seconds each"),
        ("Observation windows / steps (seconds)", "100, 120, 140 / 50, 60, 70"),
        ("Input features / classes", "11 float32 features / bus=0, car=1, train=2"),
        ("Recorded thermal interruptions", sum(len(r["interruptions"]) for r in runs)),
    ])
    doc.paragraph("The Android release field reports 17 while the recorded build fingerprint identifies an Android 13 build. Both original values are retained rather than silently resolving the discrepancy.")
    doc.table("Numerical study inventory", ["Mode", "Processes", "Candidates", "Warm calls", "Replay windows", "Memory snapshots"],
              [[MODES[mode], len(group := [r for r in runs if r["mode"] == mode]), 9,
                sum(s["kind"] == "warm" for r in group for s in r["samples"]),
                sum(s["kind"] == "replay" for r in group for s in r["samples"]),
                sum(len(r["memory"]) for r in group)] for mode in ["latency", "memory", "replay_fast"]])
    conditions = []
    for field in ["battery_level", "battery_temperature_tenths_c", "screen_brightness", "thermal_status", "plugged"]:
        values = [r[phase][field] for r in runs for phase in ["device_start", "device"] if r.get(phase, {}).get(field) is not None]
        conditions.append([field.replace("_", " "), len(values), min(values), max(values)])
    doc.table("Recorded start/end operating conditions", ["Field", "Observations", "Minimum", "Maximum"], conditions,
              note="Battery temperature is in tenths of a degree Celsius. Plugged=2 means USB power. Screen brightness is the Android setting, not calibrated luminance. Power saving was disabled; the app kept the screen on.")
    doc.key_values("Metric boundaries", [
        ("Native load", "Asset read and session creation with the global runtime already initialized."),
        ("Native inference", "OrtSession.run with a prepared input tensor; excludes tensor creation and output decoding/disposal."),
        ("Full Flutter call", "Includes platform-channel transfer, tensor construction, native execution, output decoding/disposal and return transfer."),
        ("Replay processing", "Buffer work since the previous window, feature extraction and full inference call. Decode is recorded separately."),
        ("PSS / RSS", "Whole-process proportional/resident memory. Baseline already includes runtime and prepared input/expected fixtures."),
        ("Observation delay", "Sensor-window duration; distinct from model computation. First prediction requires a full valid window."),
    ])
    doc.paragraph("Before each run, checksums and Android label/probability parity are verified. The first inference occurs before validation/warmup; measured warm calls follow validation and 100 explicit warmups. Checksum reads can warm filesystem caches: load times are process-start measurements with uncontrolled OS cache, not guaranteed cold-flash reads. Sampled memory peaks can miss brief transients.")
    doc.paragraph("Mean and median are ordinary sample summaries. The p95 index is 0.95(N-1), linearly interpolated between adjacent sorted values. SD is the sample standard deviation with denominator N-1; N=1 is reported with SD=0 as a bookkeeping convention, not evidence of zero uncertainty. No confidence intervals are inferred from repeated calls.")
    doc.section("Compact per-model result summary")
    by_memory = {r["model_id"]: r for r in mem}
    doc.table("Core latency, model size and memory results", ["W s", "Seed", "ONNX bytes", "Native ms", "p95 ms", "App ms", "PSS MiB", "Delta MiB"],
              [[r["window_seconds"], r["seed"], r["model_bytes"], r["native_us_median"]/1000, r["native_us_p95"]/1000,
                r["app_us_median"]/1000, by_memory[r["model_id"]]["warmed_pss_kib_median"]/1024,
                by_memory[r["model_id"]]["warmed_pss_delta_kib_median"]/1024] for r in lat],
              note="Latency pools 5,000 measured calls per candidate. PSS and delta are medians across five process snapshots. PSS includes Flutter, native runtime, fixtures and validation overhead; delta estimates extra process cost rather than isolated weight storage.")
    doc.section("Model serialization and numerical parity")
    models = manifest["models"]
    doc.table("Artifact sizes, feature-table cases and conversion error", ["W s", "Seed", "ONNX bytes", "Joblib bytes", "Cases", "Max abs. error"],
              [[m["window_seconds"], m["seed"], m["bytes"], m["joblib_bytes"], m["parity"]["rows"], m["parity"]["max_absolute_error"]] for m in models])
    doc.paragraph(f"Total deployed model bytes: {sum(m['bytes'] for m in models):,}. Total desktop parity cases across nine candidates: {sum(m['parity']['rows'] for m in models):,}. Exact label agreement is required; probability tolerance is rtol=1e-5 and atol=1e-6. Cases include every saved feature-table row plus one missing-value case per feature and an all-missing case. The same corpus is checked on Android before a successful run is retained.")
    doc.section("APK size accounting")
    for apk in summary["apk_sizes"]:
        doc.table("Measured APK ZIP-entry contributions (bytes)", ["Component", "Uncompressed", "Stored/compressed"],
                  [[k.replace("_", " "), apk[k]["uncompressed_bytes"], apk[k]["compressed_bytes"]]
                   for k in ["onnx_runtime_libraries", "models", "fixtures", "replay"]] +
                  [["Entire APK file", None, apk["apk_bytes"]], ["Other entries and container overhead", None,
                    apk["apk_bytes"] - sum(apk[k]["compressed_bytes"] for k in ["onnx_runtime_libraries", "models", "fixtures", "replay"])]] )
        doc.paragraph("Native runtime libraries are stored uncompressed. Shared Java code, Flutter libraries, ZIP alignment/signatures and other entries are not attributed entirely to ONNX Runtime. Model graphs are much smaller than the full runtime/package.")
    covered = complete_summary_tables(doc, summary)
    doc.section("Additional snapshot distributions")
    raw_memory_tables(doc, [r for r in runs if r["mode"] == "memory"], "Inference-memory study")
    doc.section("Accelerated replay component timings")
    replay = [(r, s) for r in runs if r["mode"] == "replay_fast" for s in r["samples"] if s["kind"] == "replay"]
    doc.table("Replay window coverage and largest retained row count", ["W s", "Seed", "Windows", "Accepted", "Rejected", "Max rows"],
              [[m["window_seconds"], m["seed"], len(group := [s for r, s in replay if s["model_id"] == m["id"]]),
                sum(s["accepted"] for s in group), sum(not s["accepted"] for s in group), max(s["max_buffer_rows"] for s in group)] for m in models])
    doc.paragraph("All replay sessions are streamed with bounded time buffers. Their selected prefixes are 300 seconds long; prediction steps are half the window duration. Quality requires accelerometer/gyroscope/magnetometer rates of at least 30/4/2 Hz and gaps no larger than 500 ms, including window boundaries. Quality counts duplicate rows, but statistics use the first finite vector at each timestamp. Population SD and original m/s-squared, rad/s and microtesla units are preserved.")
    for key in ["decode_us", "buffer_us", "feature_us", "processing_us", "prediction.native_us", "prediction.app_us"]:
        rows = []
        for m in models:
            samples = [s for _, s in replay if s["model_id"] == m["id"] and s["accepted"]]
            values = [s["prediction"][key.split(".")[1]] if key.startswith("prediction.") else s[key] for s in samples]
            rows.append([m["window_seconds"], m["seed"], *[describe(values)[s] for s in STATS]])
        doc.table("Replay component: " + key.replace("_", " ") + " (microseconds)",
                  ["W s", "Seed", "N", "Mean", "Median", "p95", "SD"], rows)
    doc.paragraph("Replay processing in the measured APK used per-event integer-microsecond rounding for buffering; it can undercount sub-microsecond event work. The current source accumulates high-resolution elapsed ticks. The earlier replay metadata also stated inference-only defaults of 100 warmups/1,000 calls; those counters do not describe replay. Classification/feature parity was unaffected. Later diagnostics of the corrected APK are separate below.")
    doc.section("Separately recorded diagnostic results")
    diagnostic_paths = [("Initial corrected-release latency smoke check", "results/smoke-latency-fixed"),
                        ("Full single-model replay memory check", "artifacts/replay-memory-validation"),
                        ("Real-time pacing and deliberate cancellation", "artifacts/realtime-validation"),
                        ("Final committed APK smoke checks", "artifacts/final-smoke")]
    diagnostic_runs = []
    for label, path in diagnostic_paths:
        extra = load_runs(project / path)
        diagnostic_runs.extend(extra)
        diagnostic_tables(doc, label, extra)
    doc.section("Model configuration and provenance")
    doc.paragraph("One saved full-cohort pipeline (window 100 s, seed 42) was reused; eight candidates were refitted with their saved winning parameters and original training settings. No hyperparameter search was repeated. Seeds vary the selected hyperparameters as well as the stochastic fit, so these are nine fitted deployment candidates rather than a controlled seed-only ablation. Winning inner-validation scores below are selection scores, not independently held-out deployment accuracy.")
    params = sorted(models[0]["winner"]["params"])
    for group in [params[:5], params[5:]]:
        doc.table("Saved winning hyperparameters", ["W s", "Seed", *group],
                  [[m["window_seconds"], m["seed"], *[f"{m['winner']['params'][k]:.10g}" for k in group]] for m in models])
    doc.table("Saved inner selection and preprocessing state", ["W s", "Seed", "Winning trial", "Inner macro F1", "Indicators"],
              [[m["window_seconds"], m["seed"], m["winner"]["trial"], f"{m['winner']['score']:.10f}", len(m["preprocessing"]["indicator_indices"])] for m in models])
    medians = {}
    for w in [100, 120, 140]:
        selected = [m for m in models if m["window_seconds"] == w]
        if any(m["preprocessing"] != selected[0]["preprocessing"] for m in selected):
            raise ValueError("Preprocessing differs across seeds; expand the imputation table")
        medians[w] = selected[0]["preprocessing"]["median"]
    doc.table("Fitted training-partition medians (identical across seeds within each window)",
              ["Ordered feature", "100 s", "120 s", "140 s"],
              [[name, *[f"{medians[w][i]:.10g}" for w in [100, 120, 140]]] for i, name in enumerate(models[0]["ordered_features"])],
              align=r"p{.45\textwidth}rrr")
    doc.key_values("Training and export dependency versions", sorted({**manifest["training_versions"], **manifest["export_versions"]}.items()))
    doc.paragraph("Source training run: " + manifest["source_run"] + ". No independent test cohort, quantization, accelerator, live-sensor, energy or background-service measurements are included. The complete five-repetition real-time study was not run.")
    doc.section("Artifact integrity and reproducibility")
    doc.paragraph("All values come from locally saved raw exports and the prepared manifest. The source LaTeX is self-contained. The companion archive preserves full-precision aggregate CSV/JSON and per-call/per-snapshot CSVs; it avoids printing 45,000 individual timed calls in a result-summary document.")
    doc.paragraph("Aggregate tables display three decimal places except exact integer counts/bytes, scientific conversion errors and explicitly higher-precision model configuration. Repeated N=1 diagnostics are not pooled with the primary study or with a different app commit. Byte counts refer to the measured APK, not a later rebuild.")
    hashes = [("Manifest", sha256(project / "assets/benchmark/manifest.json")),
              ("Measured APK", summary["apk_sizes"][0]["sha256"]),
              ("Summary JSON", sha256(project / "artifacts/pixel4a-report/summary.json"))]
    hashes += [(m["id"], m["sha256"]) for m in models]
    for name, value in hashes:
        doc.add(r"\noindent\textbf{" + tex(name) + r"}\quad\texttt{" + r"\allowbreak{}".join(value[i:i+16] for i in range(0, len(value), 16)) + r"}\par")
    doc.add(r"\end{document}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(doc.lines) + "\n", encoding="utf-8")
    supplemental = project / "artifacts/latex-export"
    supplemental.mkdir(parents=True, exist_ok=True)
    write_json(supplemental / "coverage.json", {"summary_numeric_fields_rendered": covered,
               "table_count": doc.table_count, "numeric_table_cells": doc.numeric_cells,
               "study_process_runs": len(runs), "diagnostic_process_runs": len(diagnostic_runs)})
    with zipfile.ZipFile(supplemental / "phone-benchmark-results-data.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(output, "phone-benchmark-results.tex")
        for path in (project / "artifacts/pixel4a-report").iterdir():
            if path.suffix in [".csv", ".json", ".md"]:
                archive.write(path, "primary-study/" + path.name)
        for label, path in diagnostic_paths:
            for file in (project / path).rglob("*.json"):
                value = read_json(file)
                if "samples" in value and "models" in value:
                    archive.write(file, "diagnostics/" + str(file.relative_to(project)).replace("\\", "/"))
        # Manifest contains model parameters/provenance, not raw sensor recordings.
        archive.write(project / "assets/benchmark/manifest.json", "model-manifest.json")
        archive.write(supplemental / "coverage.json", "export-coverage.json")
    print(f"Wrote {output}: {doc.table_count} tables; all {covered} numerical summary fields covered")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("reports/phone-benchmark-results.tex"))
    args = parser.parse_args()
    export(args.project, args.output)
