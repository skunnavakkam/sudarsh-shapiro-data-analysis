#!/usr/bin/env python3
"""
Merge TIFF files from subdirectories using tiffcp (BigTIFF, LZW),
but if only one slice is found, do an explicit copy.
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import glob


def check_tiffcp_available():
    """Check if tiffcp is available on the system."""
    return shutil.which("tiffcp") is not None


def find_tiff_files(subdirectory_path, prefix):
    """
    Find all TIFFs matching Pos0, Pos0_1, Pos0_2, etc.
    Returns a sorted list so they merge in the right order.
    """
    pattern = os.path.join(subdirectory_path, f"{prefix}_MMStack_Pos*.ome.tif")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Warning: no files match {pattern}")
    else:
        for f in files:
            print(f"  → Found slice: {Path(f).name}")
    return files


def merge_or_copy(tiff_files, output_path):
    """
    If there's only one input, copy it; otherwise merge with tiffcp.
    """
    # Ensure parent dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Remove stale output if present
    if os.path.exists(output_path):
        print(f"Removing existing output: {output_path}")
        os.remove(output_path)

    if len(tiff_files) == 1:
        # Single file -> do a true copy
        print("Only one slice found, copying instead of merging")
        shutil.copy2(tiff_files[0], output_path)
        print(f"Copied {tiff_files[0]} → {output_path}")
        return True

    # Multiple files -> merge with tiffcp
    if not check_tiffcp_available():
        print("Error: tiffcp not found. Install libtiff-tools or equivalent.")
        return False

    cmd = ["tiffcp", "-8", "-c", "lzw"] + tiff_files + [output_path]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(f"Merged {len(tiff_files)} slices → {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print("tiffcp failed:", e.stderr or e)
        return False


def process_directory(base_directory):
    base = Path(base_directory)
    if not base.is_dir():
        print(f"Error: {base_directory} is not a directory")
        return

    subs = [d for d in base.iterdir() if d.is_dir()]
    if not subs:
        print("No subdirectories found")
        return

    for sub in subs:
        prefix = sub.name
        print(f"\nProcessing {prefix} …")
        files = find_tiff_files(str(sub), prefix)
        if not files:
            print(f"Skipping {prefix}, no TIFFs found")
            continue

        output = str(base / f"{prefix}.ome.tif")
        success = merge_or_copy(files, output)
        if not success:
            print(f"  ✗ Failed for {prefix}")


def main():
    p = argparse.ArgumentParser(
        description="Merge TIFF slices or copy if only one slice.",
    )
    p.add_argument("directory", help="Directory containing subdirectories")
    args = p.parse_args()

    print("→ Base directory:", args.directory)
    process_directory(args.directory)
    print("Done.")


if __name__ == "__main__":
    main()
