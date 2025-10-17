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

    classifier = Spec2ClassClassifier(
        mode=args.mode,
        device=args.device, 
        force_download=args.force_download
    )

    input_path = Path(args.input)

    # Auto-detect format
    if args.format == "auto":
        suffix = input_path.suffix.lower()
        if suffix == ".pkl":
            format_type = "pickle"
        elif suffix == ".mgf":
            format_type = "mgf"
        elif suffix == ".msp":
            format_type = "msp"
        elif suffix == ".csv":
            format_type = "csv"
        elif suffix == ".tsv":
            format_type = "tsv"
        else:
            print(f"Error: Cannot auto-detect format for {suffix}", file=sys.stderr)
            print(f"Supported formats: .pkl, .csv, .tsv, .mgf, .msp", file=sys.stderr)
            return 1
    else:
        format_type = args.format

    # Parse non-pickle formats
    if format_type in ["mgf", "msp", "csv", "tsv"]:
        print(f"Parsing {format_type.upper()} file...")
        try:
            if format_type == "mgf":
                df = parse_mgf_file(str(input_path))
            elif format_type == "msp":
                df = parse_msp_file(str(input_path))
            elif format_type == "csv":
                import pandas as pd
                from .core.utility_functions import read_df_and_format_mz_intensity_arrays
                df = read_df_and_format_mz_intensity_arrays(str(input_path))
            elif format_type == "tsv":
                import pandas as pd
                from .core.utility_functions import read_df_and_format_mz_intensity_arrays
                df = read_df_and_format_mz_intensity_arrays(str(input_path))
            
            # Save as temporary pickle for processing
            temp_pkl = input_path.parent / f"{input_path.stem}_temp.pkl"
            df.to_pickle(temp_pkl)
            input_path = temp_pkl
            created_temp_file = True  # Track that we created a temp file
        except Exception as e:
            print(f"Error parsing file: {e}", file=sys.stderr)
            return 1
    else:
        created_temp_file = False

    output_dir = args.output_dir if args.output_dir else input_path.parent / "results"

    try:
        results = classifier.classify_from_file(
            str(input_path), 
            output_dir=str(output_dir), 
            output_name=args.output_name,
            output_format=args.output_format,
            debug=args.debug  # Pass debug flag here
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
        
        # Clean up temp file if created
        if created_temp_file:
            temp_pkl.unlink(missing_ok=True)
        
        return 0

    except Exception as e:
        print(f"\nError during classification: {e}", file=sys.stderr)
        return 1


def download_command(args):
    """Handle download command"""
    try:
        modes_to_download = []
        
        if args.mode == "both":
            modes_to_download = ["positive", "negative"]
        else:
            modes_to_download = [args.mode]
        
        for mode in modes_to_download:
            print(f"\nDownloading {mode} mode models...")
            if args.group:
                download_models(group_name=args.group, mode=mode, force=args.force)
            elif args.model:
                download_models(model_names=[args.model], mode=mode, force=args.force)
            else:
                download_models(group_name="all_models", mode=mode, force=args.force)
            print(f"✓ Download complete for {mode} mode!")
        
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
        from .config import CHEMICAL_CLASSES_POSITIVE, CHEMICAL_CLASSES_NEGATIVE

        print(f"\n=== Positive Mode ({len(CHEMICAL_CLASSES_POSITIVE)} classes) ===")
        for i, name in enumerate(CHEMICAL_CLASSES_POSITIVE, 1):
            print(f"  {i:2d}. {name}")
        
        print(f"\n=== Negative Mode ({len(CHEMICAL_CLASSES_NEGATIVE)} classes) ===")
        for i, name in enumerate(CHEMICAL_CLASSES_NEGATIVE, 1):
            print(f"  {i:2d}. {name}")
        
        print(f"\n=== SVM Models ===")
        print("  - svm_model (positive mode)")
        print("  - svm_model (negative mode)")
        
        print(f"\nNote: Different chemical classes available for each ionization mode")
    return 0


def status_command(args):
    """Handle status command"""
    try:
        from .config import get_chemical_classes
        
        print_model_status(mode=args.mode, group_name=args.group if hasattr(args, "group") else None)
        cached = get_cached_models(mode=args.mode)
        
        num_classes = len(get_chemical_classes(args.mode))
        total = num_classes + 1  # binary models + SVM
        print(f"\nSummary: {len(cached)}/{total} models cached ({args.mode} mode)")
        return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def cache_command(args):
    """Handle cache management"""
    try:
        if args.cache_action == "info":
            print_cache_info(mode=args.mode, verbose=getattr(args, "verbose", False))
        elif args.cache_action == "directory":
            print(f"\nCache directory: {get_cache_directory(mode=args.mode)}")
            print(f"Mode: {args.mode}")
        elif args.cache_action == "clear":
            model_names = [args.model] if hasattr(args, "model") and args.model else None
            confirm = not getattr(args, "yes", False)
            if clear_model_cache(model_names=model_names, mode=args.mode, confirm=confirm):
                print(f"\n✓ Cache cleared ({args.mode} mode)")
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
    classify_parser.add_argument(
        "-m",
        "--mode",
        choices=["positive", "negative"],
        required=True,
        help="Ionization mode (REQUIRED)",
    )
    classify_parser.add_argument("-o", "--output-dir", help="Output directory")
    classify_parser.add_argument("-n", "--output-name", help="Output filename")
    classify_parser.add_argument(
        "-f",
        "--format",
        choices=["auto", "pickle", "csv", "tsv", "mgf", "msp"],
        default="auto",
        help="Input format (default: auto)",
    )
    classify_parser.add_argument(
        "--output-format",
        choices=["csv", "tsv", "pickle", "all"],
        default="csv",
        help="Output format (default: csv). Use 'all' for all formats.",
    )
    classify_parser.add_argument(
        "-d", "--device", choices=["cpu", "cuda"], default="cpu", help="Device"
    )
    classify_parser.add_argument("--force-download", action="store_true")
    classify_parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate prediction vectors (all 43 class probabilities)",
    )

    # Download
    download_parser = subparsers.add_parser("download", help="Download models")
    download_parser.add_argument(
        "-m",
        "--mode",
        choices=["positive", "negative", "both"],
        required=True,
        help="Ionization mode: positive, negative, or both (REQUIRED)",
    )
    download_parser.add_argument("--group", choices=get_model_groups())
    download_parser.add_argument("--model", help="Specific model name")
    download_parser.add_argument("--force", action="store_true")

    # List
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--groups", action="store_true", help="List groups")

    # Status
    status_parser = subparsers.add_parser("status", help="Show cache status")
    status_parser.add_argument(
        "-m",
        "--mode",
        choices=["positive", "negative"],
        required=True,
        help="Ionization mode (REQUIRED)",
    )

    # Cache
    cache_parser = subparsers.add_parser("cache", help="Manage cache")
    cache_parser.add_argument("cache_action", choices=["info", "directory", "clear"])
    cache_parser.add_argument(
        "-m",
        "--mode",
        choices=["positive", "negative"],
        required=True,
        help="Ionization mode (REQUIRED)",
    )
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