#!/usr/bin/env python
"""Inspect BrainCAP HDF5 output files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="List and preview datasets in BrainCAP HDF5 outputs."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="An HDF5 file or a BrainCAP output directory.",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset path to preview, for example cluster_means.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Maximum values shown for each dataset preview (default: 10).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Export datasets as NumPy .npy files into this directory.",
    )
    return parser


def find_hdf5_files(path: Path) -> Iterable[Path]:
    """Return HDF5 files from a file or recursively from a directory."""
    if path.is_file():
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"Expected an HDF5 file: {path}")
        return [path]
    if path.is_dir():
        return sorted(
            file
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in {".h5", ".hdf5"}
        )
    raise FileNotFoundError(f"Path does not exist: {path}")


def format_preview(dataset: h5py.Dataset, max_items: int) -> str:
    """Return a compact preview of an HDF5 dataset."""
    if dataset.shape == ():
        return repr(dataset[()])

    values = np.asarray(dataset[...]).reshape(-1)
    preview = values[:max_items]
    suffix = " ..." if values.size > max_items else ""
    return f"{preview!r}{suffix}"


def export_dataset(
    source: Path,
    dataset_name: str,
    dataset: h5py.Dataset,
    export_dir: Path,
) -> Path:
    """Export one HDF5 dataset to a uniquely named NumPy file."""
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset_name = dataset_name.replace("/", "__")
    output = export_dir / f"{source.stem}__{safe_dataset_name}.npy"
    np.save(output, np.asarray(dataset[...]))
    return output


def print_file_contents(
    path: Path,
    dataset_name: str | None,
    max_items: int,
    export_dir: Path | None,
) -> None:
    """Print metadata and selected data from one HDF5 file."""
    print(f"\nFILE: {path}")
    with h5py.File(path, "r") as handle:
        dataset_names: list[str] = []

        def collect_dataset(name: str, item: h5py.Dataset | h5py.Group) -> None:
            if isinstance(item, h5py.Dataset):
                dataset_names.append(name)

        handle.visititems(collect_dataset)
        if not dataset_names:
            print("  No datasets found.")
            return

        for name in dataset_names:
            if dataset_name is not None and name != dataset_name:
                continue
            dataset = handle[name]
            print(f"  DATASET: {name}")
            print(f"    shape: {dataset.shape}")
            print(f"    dtype: {dataset.dtype}")
            print(f"    preview: {format_preview(dataset, max_items)}")
            if export_dir is not None:
                output = export_dataset(path, name, dataset, export_dir)
                print(f"    exported: {output}")

        if dataset_name is not None and dataset_name not in dataset_names:
            available = ", ".join(dataset_names)
            raise KeyError(
                f"Dataset '{dataset_name}' was not found in {path}. "
                f"Available datasets: {available}"
            )


def main() -> int:
    """Inspect the requested HDF5 file or output directory."""
    parser = build_parser()
    args = parser.parse_args()
    if args.max_items < 1:
        parser.error("--max-items must be at least 1.")

    try:
        files = list(find_hdf5_files(args.path.resolve()))
        if not files:
            parser.error(f"No HDF5 files found under: {args.path}")
        for file in files:
            print_file_contents(file, args.dataset, args.max_items, args.export_dir)
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
