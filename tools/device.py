"""Run independent Android process repetitions and collect completed exports."""
import argparse
import json
import random
import re
import subprocess
import time
import uuid
import zipfile
from pathlib import Path

from common import read_json, sha256, write_json

PACKAGE = "au.edu.tcct.transport_mode_benchmark"
MODES = ["latency", "memory", "replay_fast", "replay_realtime", "replay_fast_memory", "replay_realtime_memory"]


def apk_size(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        def size(predicate):
            selected = [x for x in entries if predicate(x.filename)]
            return {"uncompressed_bytes": sum(x.file_size for x in selected),
                    "compressed_bytes": sum(x.compress_size for x in selected)}
        return {"apk_bytes": Path(path).stat().st_size, "sha256": sha256(path),
                "onnx_runtime_libraries": size(lambda n: n.startswith("lib/") and "onnxruntime" in n),
                "models": size(lambda n: "/benchmark/models/" in n),
                "fixtures": size(lambda n: "/benchmark/fixtures/" in n),
                "replay": size(lambda n: "/benchmark/replay/" in n),
                "note": "ZIP entry contributions; excludes ZIP/signature overhead and shared Java code."}


def schedule(ids, repetitions, seed):
    rng, result = random.Random(seed), []
    for repeat in range(repetitions):
        order = list(ids)
        rng.shuffle(order)
        result.extend((repeat, model) for model in order)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial")
    parser.add_argument("--apk", type=Path, help="Install this release APK before running")
    parser.add_argument("--manifest", type=Path, default=Path("assets/benchmark/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/device"))
    parser.add_argument("--model", action="append")
    parser.add_argument("--mode", choices=MODES, default="latency")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    args = parser.parse_args()
    if args.repetitions < 1 or args.timeout_seconds < 1 or args.cooldown_seconds < 0:
        parser.error("Invalid repetition, timeout or cooldown value")
    prefix = [args.adb, *(["-s", args.serial] if args.serial else [])]
    def adb(*parts, check=True):
        return subprocess.run([*prefix, *map(str, parts)], check=check, capture_output=True, text=True)
    devices = [line.split()[0] for line in adb("devices").stdout.splitlines()[1:]
               if len(line.split()) >= 2 and line.split()[1] == "device"]
    if not args.serial:
        if len(devices) != 1:
            parser.error("Connect one authorized physical Android phone, or specify --serial")
        prefix.extend(["-s", devices[0]])
    if adb("shell", "getprop", "ro.kernel.qemu").stdout.strip() == "1":
        parser.error("Use a physical phone for reported benchmarks")
    if args.apk:
        adb("install", "-r", args.apk)
    if not adb("shell", "pm", "path", PACKAGE).stdout.strip():
        parser.error("Benchmark app is not installed; provide --apk with the release APK")
    manifest = read_json(args.manifest)
    ids = args.model or [m["id"] for m in manifest["models"]]
    if len(set(ids)) != len(ids) or not set(ids) <= {m["id"] for m in manifest["models"]}:
        parser.error("Unknown or duplicate model IDs")
    if not all(re.fullmatch(r"[A-Za-z0-9_-]{1,45}", model) for model in ids):
        parser.error("Model IDs must contain 1–45 letters, digits, underscores or hyphens")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.apk:
        write_json(args.output / "apk-size.json", apk_size(args.apk))
    plan = schedule(ids, args.repetitions, args.seed)
    write_json(args.output / "schedule.json", {"schema_version": 1, "mode": args.mode, "seed": args.seed,
                                               "manifest_sha256": sha256(args.manifest), "tasks": plan})
    for repeat, model in plan:
        token = f"{args.mode}_r{repeat}_{model}_{uuid.uuid4().hex[:8]}"
        remote = f"/sdcard/Android/data/{PACKAGE}/files/results/{token}.json"
        print(f"[{repeat + 1}/{args.repetitions}] {model} / {args.mode}", flush=True)
        adb("shell", "am", "force-stop", PACKAGE)
        adb("shell", "am", "start", "-W", "-n", f"{PACKAGE}/.MainActivity",
            "--es", "benchmark_model", model, "--es", "benchmark_mode", args.mode,
            "--es", "benchmark_token", token)
        deadline = time.monotonic() + args.timeout_seconds
        try:
            while time.monotonic() < deadline:
                exists = adb("shell", "test", "-f", remote, check=False).returncode == 0
                if exists:
                    break
                time.sleep(2)
            else:
                raise TimeoutError(f"No completion export for {token}; inspect the phone or adb logcat")
            for suffix in ["json", "samples.csv", "memory.csv"]:
                adb("pull", remote.removesuffix("json") + suffix, args.output / f"{token}.{suffix}")
            result = read_json(args.output / f"{token}.json")
            if result["manifest_sha256"] != sha256(args.manifest):
                raise ValueError("Installed APK contains a different benchmark bundle; rebuild and reinstall")
            if result["status"] != "complete":
                raise RuntimeError(f"Device run {result['status']}: {result.get('error')}")
            if not result["fresh_process"] or result["device"]["build_type"] != "release":
                raise RuntimeError("Results require one model in a fresh release process")
        finally:
            adb("shell", "am", "force-stop", PACKAGE)
        if args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)
    print(f"Collected {len(plan)} process runs in {args.output}")


if __name__ == "__main__":
    main()
