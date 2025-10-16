# Path: spec2class/src/spec2class/main.py
# This file goes in src/spec2class/

"""Command-line interface for Spec2Class"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .classifier import Spec2ClassClassifier
from .data_processor import parse_mgf_file, parse_msp_file
from .models import (
    download_models,
    get_cached_models,
    get_model_groups,
    get_models_in_group,
    print_cache_info,
    print_model_status,
    clear_model_cache,
    get_cache_directory,
)


def classify_command(args):
    """Handle classify command"""
    print(f"Spec2Class v{__version__}")
    print("=" * 60)

    classifier = Spec2ClassClassifier(device=args.device, force_download=args.force_download)

    input_path = Path(args.input)

    if args.format == "auto":
        suffix = input_path.suffix.lower()
        if suffix == ".pkl":
            format_type = "pickle"
        elif suffix == ".mgf":
            format_type = "mgf"
        elif suffix == ".msp":
            format_type = "msp"
        else:
            print(f"Error: Cannot auto-detect format for {suffix}", file=sys.stderr)
            return 1
    else:
        format_type = args.format

    if format_type in ["mgf", "msp"]:
        print(f"Parsing {format_type.upper()} file...")
        try:
            if format_type == "mgf":
                df = parse_mgf_file(str(input_path))
            else:
                df = parse_msp_file(str(input_path))
            temp_pkl = input_path.parent / f"{input_path.stem}_temp.pkl"
            df.to_pickle(temp_pkl)
            input_path = temp_pkl
        except Exception as e:
            print(f"Error parsing file: {e}", file=sys.stderr)
            return 1

    output_dir = args.output_dir if args.output_dir else input_path.parent / "results"

    try:
        results = classifier.classify_from_file(
            str(input_path), output_dir=str(output_dir), output_name=args.output_name
        )

        print("\n" + "=" * 60)
        print("Classification Summary")
        print("=" * 60)
        print(f"Total spectra classified: {len(results)}")
        print(f"\nTop 5 predicted classes:")
        class_counts = results["final_pred"].value_counts().head(5)
        for class_name, count in class_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        return 0

    except Exception as e:
        print(f"\nError during classification: {e}", file=sys.stderr)
        return 1


def download_command(args):
    """Handle download command"""
    try:
        if args.group:
            download_models(group_name=args.group, force=args.force)
        elif args.model:
            download_models(model_names=[args.model], force=args.force)
        else:
            download_models(group_name="all_models", force=args.force)
        print("\n✓ Download complete!")
        return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def list_command(args):
    """Handle list command"""
    if args.groups:
        print("\nAvailable Model Groups:")
        for group_name in get_model_groups():
            models = get_models_in_group(group_name)
            print(f"\n{group_name}: {len(models)} models")
    else:
        from .config import CHEMICAL_CLASSES

        print(f"\nBinary Classifiers ({len(CHEMICAL_CLASSES)}):")
        for i, name in enumerate(CHEMICAL_CLASSES, 1):
            print(f"  {i:2d}. {name}")
        print(f"\nSVM Model: svm_model")
    return 0


def status_command(args):
    """Handle status command"""
    try:
        print_model_status(group_name=args.group if hasattr(args, "group") else None)
        cached = get_cached_models()
        from .config import CHEMICAL_CLASSES

        total = len(CHEMICAL_CLASSES) + 1
        print(f"\nSummary: {len(cached)}/{total} models cached")
        return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def cache_command(args):
    """Handle cache management"""
    try:
        if args.cache_action == "info":
            print_cache_info(verbose=getattr(args, "verbose", False))
        elif args.cache_action == "directory":
            print(f"\n{get_cache_directory()}")
        elif args.cache_action == "clear":
            model_names = [args.model] if hasattr(args, "model") and args.model else None
            confirm = not getattr(args, "yes", False)
            if clear_model_cache(model_names=model_names, confirm=confirm):
                print("\n✓ Cache cleared")
                return 0
            else:
                return 1
        return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Spec2Class: Plant Secondary Metabolite Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"Spec2Class {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Classify
    classify_parser = subparsers.add_parser("classify", help="Classify MS/MS spectra")
    classify_parser.add_argument("-i", "--input", required=True, help="Input file path")
    classify_parser.add_argument("-o", "--output-dir", help="Output directory")
    classify_parser.add_argument("-n", "--output-name", help="Output filename")
    classify_parser.add_argument(
        "-f",
        "--format",
        choices=["auto", "pickle", "mgf", "msp"],
        default="auto",
        help="Input format",
    )
    classify_parser.add_argument(
        "-d", "--device", choices=["cpu", "cuda"], default="cpu", help="Device"
    )
    classify_parser.add_argument("--force-download", action="store_true")

    # Download
    download_parser = subparsers.add_parser("download", help="Download models")
    download_parser.add_argument("--group", choices=get_model_groups())
    download_parser.add_argument("--model", help="Specific model name")
    download_parser.add_argument("--force", action="store_true")

    # List
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--groups", action="store_true", help="List groups")

    # Status
    subparsers.add_parser("status", help="Show cache status")

    # Cache
    cache_parser = subparsers.add_parser("cache", help="Manage cache")
    cache_parser.add_argument("cache_action", choices=["info", "directory", "clear"])
    cache_parser.add_argument("--model", help="Specific model")
    cache_parser.add_argument("-v", "--verbose", action="store_true")
    cache_parser.add_argument("-y", "--yes", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "classify":
            return classify_command(args)
        elif args.command == "download":
            return download_command(args)
        elif args.command == "list":
            return list_command(args)
        elif args.command == "status":
            return status_command(args)
        elif args.command == "cache":
            return cache_command(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())