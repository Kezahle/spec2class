# Path: spec2class/src/spec2class/classifier.py
# Wrapper that uses original Spec2Class core code

import torch
import pandas as pd
from pathlib import Path
from typing import Literal, Optional

from .core import binning_df, get_pred_vecs, svm_pred
from .models import check_and_download_models, get_cache_info
from .config import get_chemical_classes


class Spec2ClassClassifier:
    """High-level wrapper for Spec2Class using original core code"""

    def __init__(
        self, 
        mode: Literal["positive", "negative"] = "positive",
        device: str = "cpu", 
        force_download: bool = False
    ):
        """
        Initialize Spec2Class classifier

        Args:
            mode: Ionization mode - 'positive' or 'negative'
            device: Device to run models on ('cpu' or 'cuda')
            force_download: Force redownload models from HuggingFace
        """
        if mode not in ["positive", "negative"]:
            raise ValueError(f"Mode must be 'positive' or 'negative', got '{mode}'")
        
        self.mode = mode
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        
        # Model parameters from original config
        self.bin_width = 0.1
        self.start_value = 50
        self.end_value = 550
        self.dropout_conv = 0.95  # MUST match training config
        self.dropout_linear = 0.95  # MUST match training config
        self.ms1_tolerance = 0.001
        self.batch_size = 128
        self.num_workers = 0
        self.params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers
        }

        print(f"Initializing Spec2Class ({mode} mode)...")

        # Download models if needed
        if force_download:
            from .models import download_models
            download_models(group_name="all_models", mode=mode, force=True)
        else:
            check_and_download_models(group_name="all_models", mode=mode)

        # Get paths to downloaded models
        cache_info = get_cache_info(mode=mode)
        
        # Find binary models directory
        binary_cache = cache_info["binary_models"]
        if binary_cache.get("files"):
            # Get directory from first file
            first_file_path = Path(binary_cache["files"][0]["path"])
            self.binary_models_dir = str(first_file_path.parent)
        else:
            raise RuntimeError(f"Binary models ({mode} mode) not found in cache")

        # Find SVM model path
        svm_cache = cache_info["svm_model"]
        if svm_cache.get("files"):
            self.svm_model_path = svm_cache["files"][0]["path"]
        else:
            raise RuntimeError(f"SVM model ({mode} mode) not found in cache")

        self.chemclass_list = get_chemical_classes(mode)

        print(f"✓ Spec2Class ready! Using {len(self.chemclass_list)} classes ({mode} mode)")
        print(f"  Binary models: {self.binary_models_dir}")
        print(f"  SVM model: {self.svm_model_path}")
        print(f"  Device: {self.device}")

    def _save_results(self, df: pd.DataFrame, output_dir: str, filename: str, output_format: str):
        """
        Save results in the specified format(s)
        
        Args:
            df: DataFrame with results
            output_dir: Directory to save results
            filename: Base filename (without extension)
            output_format: Format to save ('csv', 'tsv', 'pickle', or 'all')
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if output_format == "all":
            # Save in all formats
            df.to_csv(output_path / f"{filename}.csv", index=False)
            df.to_csv(output_path / f"{filename}.tsv", sep='\t', index=False)
            df.to_pickle(output_path / f"{filename}.pkl")
            print(f"Saved {filename} in all formats (csv, tsv, pkl)")
        elif output_format == "csv":
            csv_path = output_path / f"{filename}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path}")
        elif output_format == "tsv":
            tsv_path = output_path / f"{filename}.tsv"
            df.to_csv(tsv_path, sep='\t', index=False)
            print(f"Saved {tsv_path}")
        elif output_format == "pickle":
            pkl_path = output_path / f"{filename}.pkl"
            df.to_pickle(pkl_path)
            print(f"Saved {pkl_path}")

    def classify_from_file(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        output_name: Optional[str] = None,
        output_format: str = "csv",
        debug: bool = False,
    ) -> pd.DataFrame:
        """
        Classify spectra from input file using original Spec2Class logic

        Args:
            input_path: Path to input pickle/csv/tsv file
            output_dir: Optional output directory
            output_name: Optional output filename
            output_format: Output format - 'csv' (default), 'tsv', 'pickle', or 'all'
            debug: If True, save intermediate prediction vectors (all 43 class probabilities)

        Returns:
            DataFrame with predictions
        """
        from pathlib import Path as PathLib
        from .core.utility_functions import read_df_and_format_mz_intensity_arrays
        
        # Load input using original utility function
        input_df = read_df_and_format_mz_intensity_arrays(input_path)
        print(f"Loaded {len(input_df)} spectra from {input_path}")

        # Set output parameters
        if output_dir is None:
            output_dir = str(PathLib(input_path).parent / "results")
        if output_name is None:
            output_name = PathLib(input_path).stem

        PathLib(output_dir).mkdir(parents=True, exist_ok=True)

        # Run original Spec2Class pipeline
        print("Step 1/3: Binning spectra...")
        print(f"Debug mode: {debug}")  # Debug info
        binned_df = binning_df(
            input_df,
            self.bin_width,
            self.start_value,
            self.end_value,
            output_name,
            self.ms1_tolerance
        )

        print("Step 2/3: Getting predictions from binary models...")
        
        # Import neural_net module for get_pred_vecs
        import sys
        from pathlib import Path as PathLib
        core_path = str(PathLib(__file__).parent / "core")
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        
        net_path = str(PathLib(__file__).parent / "core" / "neural_net.py")
        net_dir = str(PathLib(__file__).parent / "core")
        
        pred_vecs_df = get_pred_vecs(
            self.chemclass_list,
            output_name,
            binned_df,
            output_dir,
            self.end_value,
            self.start_value,
            self.bin_width,
            net_path,
            net_dir,
            self.dropout_conv,
            self.dropout_linear,
            self.binary_models_dir,
            self.params
        )

        # Save prediction vectors if debug mode is enabled
        if debug:
            print("Debug mode: Saving prediction vectors...")
            self._save_results(
                pred_vecs_df,
                output_dir,
                f"{output_name}_y_pred_df",
                output_format
            )

        print("Step 3/3: Final SVM prediction...")
        results_df = svm_pred(
            self.svm_model_path,
            output_dir,
            output_name,
            pred_vecs_df,
            self.chemclass_list,
            output_format=output_format
        )

        print(f"\n✓ Classification complete! Results saved to {output_dir}")
        return results_df

    def classify_dataframe(
        self, 
        df: pd.DataFrame, 
        output_format: str = "csv",
        debug: bool = False,
    ) -> pd.DataFrame:
        """
        Classify spectra from a DataFrame

        Args:
            df: DataFrame with 'mz', 'Intensity', 'DB.' columns
            output_format: Output format - 'csv' (default), 'tsv', 'pickle', or 'all'
            debug: If True, save intermediate prediction vectors

        Returns:
            DataFrame with predictions
        """
        # Save to temp file and classify
        import tempfile
        from pathlib import Path as PathLib
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            df.to_pickle(tmp.name)
            temp_path = tmp.name

        try:
            results = self.classify_from_file(
                temp_path, 
                output_format=output_format,
                debug=debug
            )
            return results
        finally:
            PathLib(temp_path).unlink(missing_ok=True)

    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        return {
            "mode": self.mode,
            "device": self.device,
            "num_classes": len(self.chemclass_list),
            "classes": self.chemclass_list,
            "binary_models_dir": self.binary_models_dir,
            "svm_model_path": self.svm_model_path,
            "bin_width": self.bin_width,
            "mz_range": (self.start_value, self.end_value),
        }

    def print_model_info(self):
        """Print model information"""
        info = self.get_model_info()
        print("\n" + "=" * 60)
        print("Spec2Class Model Information")
        print("=" * 60)
        print(f"Mode: {info['mode']}")
        print(f"Device: {info['device']}")
        print(f"Output classes: {info['num_classes']}")
        print(f"m/z range: {info['mz_range'][0]}-{info['mz_range'][1]} Da")
        print(f"Bin width: {info['bin_width']} Da")
        print(f"Binary models: {info['binary_models_dir']}")
        print(f"SVM model: {info['svm_model_path']}")