# Pixel 4a validation — 29 October 2026

Measured APK source commit: `0131eba622252fae8d1f190a5c38150139f0e46a` (release, ARM64).
Device-reported model: Google Pixel 4a; chipset: SM7150; Android release: 17. The reported OS fingerprint remains an Android 13 fingerprint, so treat this as the specific device/OS combination recorded in the raw exports.
ONNX Runtime 1.24.3; CPU; batch one; one intra/inter-op thread; sequential execution; full optimization; spinning off. USB powered, screen kept on; power saving off. Five fresh processes per candidate for latency and separately for memory. Model order shuffled per repetition with seed 42; two-second inter-process cooldown.

All nine candidates passed on-device label/probability parity. All 90 latency/memory process runs completed with no recorded thermal pauses. Each latency run contains 100 explicit warmups and 1,000 measured calls after parity validation. Native timer excludes tensor construction and output disposal; app-call timer includes these and the platform round trip.

| Window s | Seed | ONNX bytes | Native median ms | Native p95 ms | App median ms | Warmed PSS MiB | Warmed minus baseline MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 42 | 104934 | 0.162 | 0.224 | 0.550 | 155.67 | 28.91 |
| 100 | 43 | 186694 | 0.207 | 0.264 | 0.622 | 157.18 | 29.78 |
| 100 | 44 | 146737 | 0.187 | 0.244 | 0.589 | 157.11 | 29.91 |
| 120 | 42 | 483463 | 0.276 | 0.348 | 0.672 | 161.00 | 31.22 |
| 120 | 43 | 180924 | 0.202 | 0.259 | 0.620 | 153.50 | 28.17 |
| 120 | 44 | 144588 | 0.168 | 0.226 | 0.551 | 153.10 | 26.86 |
| 140 | 42 | 305178 | 0.206 | 0.269 | 0.609 | 151.90 | 25.46 |
| 140 | 43 | 129365 | 0.167 | 0.227 | 0.545 | 147.85 | 23.74 |
| 140 | 44 | 112978 | 0.156 | 0.214 | 0.533 | 150.06 | 26.02 |

Latency medians/p95 pool 5,000 calls per candidate. PSS columns are medians of five process snapshots; deltas are calculated within each process before taking their median. These include the Flutter/runtime/fixtures/validation harness and do not isolate model weights. More detailed process variability, loading, first inference, PSS/RSS phases and observed peaks are available in the generated report.

All nine candidates also completed accelerated replay of the same three 300-second sessions (one process per candidate). This is functional device validation, not the five-repetition replay study. An additional full replay memory pass succeeded for the 100-second/seed-42 candidate.

Local raw exports: `results/pixel4a-study/`. Recomputed tables and JSON/CSV report: `artifacts/pixel4a-report/`. APK size accounting: `artifacts/apk-size.json`. These generated artifacts are ignored by Git; this summary is tracked.

The measured native/model and inference-memory behavior is unchanged by later replay timer-resolution and replay metadata corrections. The earlier accelerated replay buffer microseconds were rounded at each event; use a current build for new replay processing measurements. The current result format separates inference warmup/call counts from replay runs.

No independent accuracy evaluation, live sensor test, energy measurement, or full five-repetition real-time study was performed.

Real-time replay was checked on the phone through two accepted prediction boundaries (100 and 150 seconds) and then cancelled using the app's Cancel button. Both prediction records followed the 50-second step; completion lag was approximately 58 and 65 ms respectively, including processing and scheduling. The app exported the cancelled run and closed its stream/session. This validates pacing/cancellation, not a completed 15-minute real-time trial. The cancelled diagnostic is in `artifacts/realtime-validation/` and is excluded from the successful measurement study.
