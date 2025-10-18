import os
import subprocess
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process YAML configurations.")
    parser.add_argument(
        "--base-path",
        type=str,
        required=True,
        help="Root folder to search for 'val' directories",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Only process the directories including 'filter'",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path)

    val_dirs = list(base_path.rglob("val"))
    print(f"find {len(val_dirs)} val dirs")

    filter_str = args.filter
    if filter_str:
        val_dirs = [d for d in val_dirs if filter_str in str(d)]
        print(f"filtered: {len(val_dirs)}")

    for val_dir in sorted(val_dirs):
        print(f"processing: {val_dir}")

        try:
            result = subprocess.run(
                ["python", "scripts/metric.py", "-i", str(val_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"failed [{val_dir}]: {e.stderr}")
        print("-" * 50)
    print("all scene processed")


if __name__ == "__main__":
    main()
