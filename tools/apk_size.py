"""Inspect actual ZIP entry contributions in a built APK."""
import argparse
from pathlib import Path
from common import write_json
from device import apk_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('apk', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/apk-size.json'))
    args = parser.parse_args()
    result = apk_size(args.apk)
    write_json(args.output, result)
    print(f"APK: {result['apk_bytes']:,} bytes; report: {args.output}")
