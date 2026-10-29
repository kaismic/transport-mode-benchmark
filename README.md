# TMD Phone Benchmark

A separate Flutter/Android harness for comparing classification models on a physical phone. The first bundle contains nine full-cohort Sydney V1 XGBoost candidates: windows 100/120/140 seconds × seeds 42/43/44. Training runs and the detector/collector apps are read-only inputs.

## Current validation

- Nine pipelines prepared from `ml/sydney-tmd-v1/runs/sydney-20261007`; the saved 100-second/seed-42 pipeline is reused, and the other eight are refitted with their saved winning parameters and original class weights/threads. No tuning is repeated.
- Desktop ONNX labels match Python on **38,820** cases, including missing-value fixtures. Maximum probability difference is **1.1920929e-7**. Desktop and Android use ONNX Runtime **1.24.3**.
- Deployed models total **1,794,861 bytes**, individually **104,934–483,463 bytes**. These are complete preprocessing/classifier graphs with 11 float32 inputs and three probabilities.
- Python tests check conversion, missing indicators, original-extractor replay parity, process schedules, APK accounting and report calculations. Flutter tests check streamed feature parity, bounded buffers, cancellation cleanup and UI initialization.
- The ARM64 release APK runs on a connected Pixel 4a. All nine models passed Android numerical parity; five fresh-process latency runs and five memory runs per model completed, followed by complete accelerated replay for all nine. See [VALIDATION.md](VALIDATION.md) for measured results and remaining test limits. The app checks Android parity before retaining a successful run.

## Prepare the local model/data bundle

Use Python 3.14 (validated with 3.14.5), Flutter 3.44/Dart 3.12, and Android SDK/JDK tooling compatible with the generated Gradle project. Android 8/API 26 or later is required. The supplied build command targets ARM64 phones; other architectures require rebuilding for that target.

From this project directory in PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r tools/requirements.txt
.venv/Scripts/python tools/prepare.py --run ../../ml/sydney-tmd-v1/runs/sydney-20261007
.venv/Scripts/python -m pytest tests -q
flutter pub get
flutter test
flutter analyze
```

The exporter verifies the source prepared/final checksums and numerical dependency versions, validates the saved model conversion first, then prepares all candidates. It writes `assets/benchmark/manifest.json` **last**; missing/failed parity prevents publishing a complete bundle. `artifacts/pipelines/` contains the nine joblib files. Only load trusted joblib artifacts.

Replay uses the same three selected sessions across all candidates, one per class. The first 300 seconds of each session are converted from the run's normalized event tables; original units, timestamps and duplicate rows are retained. `--replay-seconds` changes the cap (minimum 140); repeat `--session-id` to restrict selection. Original sampling rates can exceed the newer collector's 60 Hz cap.

Generated model/data bundles, joblib files, APKs and results are local ignored artifacts. A source-only clone must run preparation before building a useful APK. The APK includes the prepared recorded-data fixtures; no backend or network service is used.

## Build and run on a phone

After committing source changes, build an APK with its actual commit embedded:

```powershell
./tools/build.ps1
adb devices -l
.venv/Scripts/python tools/device.py --apk build/app/outputs/flutter-apk/app-release.apk --output results/latency
```

Enable USB debugging and authorize the computer on the phone. The desktop runner rejects emulators and ambiguous device selections; use `--serial SERIAL` when needed. It installs the APK, shuffles model order reproducibly for each repetition, force-stops/restarts the app for each model, and collects **five fresh process runs per model**, with 100 warmups and 1,000 timed calls in each run. Calls cycle through source feature vectors in a fixed shuffled order. The first inference precedes parity validation and warmup.

Measure memory separately:

```powershell
.venv/Scripts/python tools/device.py --mode memory --output results/memory
.venv/Scripts/python tools/device.py --mode replay_fast --output results/replay-fast
.venv/Scripts/python tools/device.py --mode replay_fast_memory --output results/replay-fast-memory
```

Real-time replay follows original event spacing. The default bundle takes **15 minutes per model per repetition**; nine models × five repetitions take at least 11.25 hours for each real-time mode:

```powershell
.venv/Scripts/python tools/device.py --mode replay_realtime --output results/replay-realtime
.venv/Scripts/python tools/device.py --mode replay_realtime_memory --output results/replay-realtime-memory
```

Use `--model sydney-v1-w100-s42 --repetitions 1` for a smoke run. This is useful for validation but does not satisfy the five-repetition protocol. `--timeout-seconds` controls per-process waiting and `--cooldown-seconds` the gap between processes. The runner stops on a failed/cancelled run or bundle mismatch; diagnostic exports already collected are retained. Ctrl+C stops the active app. No handset settings are changed automatically.

Keep the screen/brightness, charging, connectivity, power-saving and background workload consistent. Keep the app in the foreground; backgrounding cancels a running measurement. The app keeps the screen awake. Thermal status MODERATE or worse pauses execution until the phone cools; pauses are recorded. A thermal pause during replay fails that replay so its original event spacing is not silently changed. Thermal monitoring is unavailable before API 29 and is reported as `-1`.

The app also offers model selection, six measurement modes, progress, cancellation, and Android sharing of JSON plus timing/memory CSVs. **Run suite in the app is exploratory:** it reuses one process. Use the desktop runner for independent repetitions. Results live under the app's external files `results/` directory and are collected with ADB without a debuggable build.

## Metric boundaries and interpretation

| Metric | Boundary |
|---|---|
| Deployed model bytes | Full ONNX graph, including fitted imputation and indicator handling. Current exporter embeds all data; external-weight formats require another exporter/loader. |
| Native load time | Asset read plus session creation, with the global runtime already initialized. Flutter load time additionally includes channel/worker overhead. |
| Native inference | `OrtSession.run` with an already constructed batch-one tensor; excludes tensor construction, output decoding/disposal and channel transfer. |
| App inference | Flutter platform call including native tensor construction, execution, output decoding/disposal and return transfer. Feature-vector construction happens before this timer. |
| Replay processing | Buffer work since the previous window, quality/feature extraction and app inference. File reading/JSON decoding is recorded separately; waiting, parity checks, memory sampling and UI/result export are excluded. |
| Memory | Whole-process PSS/RSS, with a runtime/input/expected-fixture baseline before model loading. Loaded/warmed differences estimate added process cost. Native buffers are included; Dart heap alone is insufficient. |
| Observation delay | 100/120/140 seconds; predictions step by 50/60/70 seconds. Separate from model computation time. |

Fixed runtime settings: full official Android ONNX Runtime, CPU provider only, sequential execution, one intra/inter-op thread, all graph optimizations, spinning disabled, batch one, no quantization. Native work runs on a dedicated worker; Dart buffering/features run in the Flutter isolate. Native timings include the Java/JNI session call rather than claiming isolated tree-kernel time.

Checksums are verified before loading, which can warm filesystem caches. Load/first-inference results are **process-start measurements with uncontrolled OS cache**, not guaranteed cold-flash reads. Repeated inference measures sustained use; real-time replay also includes long idle gaps.

Memory runs take baseline, loaded, warmed, periodic and final snapshots using Android `Debug.MemoryInfo` plus `/proc/self/status`. Inference memory is sampled every 100 calls; replay memory at most once a second. Samples and diagnostic validation allocate memory themselves, so whole-process values include harness costs. Input/expected fixtures are present at baseline; replay parsing, golden fixtures and output records add harness overhead afterward. Samples may miss transient peaks. RSS high-water values describe the entire process lifetime, including validation, and are never presented as per-model incremental peaks. Fresh processes avoid contamination from previously loaded models.

The three seeds also have different winning hyperparameters. A larger window does not imply more model inputs: all nine accept 11 features. Longer windows affect observation time and buffering; fitted ensembles determine much of inference cost. These full-cohort deployment candidates have no independent test score. Existing LOPO accuracy describes the training study, not a new evaluation of these artifacts.

## Export and analyze

```powershell
.venv/Scripts/python tools/report.py results --output artifacts/report
```

The report writes `REPORT.md`, `comparison.csv`, `windows.csv`, `samples.csv`, `memory.csv` and `summary.json`. It recomputes median, p95 (linear interpolation), mean, standard deviation, variability of process medians, load/first-inference times and memory phases/deltas from raw exports. Individual candidates remain visible; window summaries weight each candidate equally. APK entry accounting separates native runtime libraries, model graphs, fixtures and replay data; Java code/signature/ZIP overhead is not attributed exactly.

Failed/cancelled exports are excluded and listed. Different manifests/devices/Android versions/runtime versions/app commits cannot be combined. Duplicate run IDs are rejected. Coverage and protocol-completeness flags identify missing candidates, insufficient repetitions, same-process runs, development builds or thermal interruptions. Keep different bundles/builds in separate result trees. Reports do not turn thousands of repeated calls into independent training seeds or generalize one phone's results to other hardware.

## Interfaces and extension

`assets/benchmark/manifest.json` uses schema version 1. Each model supplies an ID, family, window/step, seed, artifact/fixture checksums, feature and label order, float32 batch-one input contract, label/probability output names, preprocessing provenance, winner and desktop parity results. Runtime settings are validated rather than silently ignored. Null fixture features represent NaN; infinities are rejected.

The inference runner accepts model-specific feature widths and output names. Sydney replay additionally requires the exact 11-feature ordering and `sydney-sensor-v1` adapter: half-open windows, population standard deviation, 30/4/2 Hz minimums, ≤500 ms gaps including boundaries, Euclidean magnitudes in original collector units, first finite vector per duplicate timestamp, float32 features. An incompatible feature contract fails explicitly. Future V2 families need an exporter producing this contract; other sensor/feature semantics need a new replay adapter.

On-device validation compares every saved feature-table row and missing-value fixture against Python labels/probabilities (`rtol=1e-5`, `atol=1e-6`). Replay checks every accepted/rejected boundary and feature vector against Python goldens. A mismatch marks the result failed. Cancellation closes streams/sessions; each prediction disposes tensors/results. Live sensors, background services, iOS, accelerators, quantization and energy measurement are deferred.

References: [ONNX Runtime Android](https://onnxruntime.ai/docs/tutorials/mobile/), [thread settings](https://onnxruntime.ai/docs/performance/tune-performance/threading.html), [Android benchmarking](https://developer.android.com/topic/performance/benchmarking/microbenchmark-overview), [Android memory](https://developer.android.com/tools/dumpsys#meminfo).

Release builds retain the ONNX Runtime Java API classes with the binding's required R8 keep rule; omitting it causes JNI output-conversion crashes. See [Android binding build notes](https://onnxruntime.ai/docs/build/android.html).

The desktop runner detects process exits before export, records cooldown/repetition settings, and fails promptly on missing installations. Replay buffer timing accumulates sub-microsecond durations; replay results do not claim inference-only warmup/call counts. Memory protocol flags require baseline, loaded, warmed and final phase samples.
