# Path: spec2class/src/spec2class/classifier.py
# Wrapper that uses original Spec2Class core code

import torch
import pandas as pd
from pathlib import Path
from typing import Optional

from .core import binning_df, get_pred_vecs, svm_pred
from .models import check_and_download_models, get_cache_info
from .config import CHEMICAL_CLASSES


class Spec2ClassClassifier:
    """High-level wrapper for Spec2Class using original core code"""

    def __init__(self, device: str = "cpu", force_download: bool = False):
        """
        Initialize Spec2Class classifier

        Args:
            device: Device to run models on ('cpu' or 'cuda')
            force_download: Force redownload models from HuggingFace
        """
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

        print("Initializing Spec2Class...")

        # Download models if needed
        if force_download:
            from .models import download_models
            download_models(group_name="all_models", force=True)
        else:
            check_and_download_models(group_name="all_models")

        # Get paths to downloaded models
        cache_info = get_cache_info()
        
        # Find binary models directory
        binary_cache = cache_info["binary_models"]
        if binary_cache.get("files"):
            # Get directory from first file
            first_file_path = Path(binary_cache["files"][0]["path"])
            self.binary_models_dir = str(first_file_path.parent)
        else:
            raise RuntimeError("Binary models not found in cache")

        # Find SVM model path
        svm_cache = cache_info["svm_model"]
        if svm_cache.get("files"):
            self.svm_model_path = svm_cache["files"][0]["path"]
        else:
            raise RuntimeError("SVM model not found in cache")

        self.chemclass_list = CHEMICAL_CLASSES

        print(f"✓ Spec2Class ready! Using {len(self.chemclass_list)} classes")
        print(f"  Binary models: {self.binary_models_dir}")
        print(f"  SVM model: {self.svm_model_path}")
        print(f"  Device: {self.device}")

    def classify_from_file(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Classify spectra from input file using original Spec2Class logic

        Args:
            input_path: Path to input pickle/csv/tsv file
            output_dir: Optional output directory
            output_name: Optional output filename

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

        print("Step 3/3: Final SVM prediction...")
        results_df = svm_pred(
            self.svm_model_path,
            output_dir,
            output_name,
            pred_vecs_df,
            self.chemclass_list
        )

        print(f"\n✓ Classification complete! Results saved to {output_dir}")
        return results_df

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify spectra from a DataFrame

        Args:
            df: DataFrame with 'mz', 'Intensity', 'DB.' columns

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
            results = self.classify_from_file(temp_path)
            return results
        finally:
            PathLib(temp_path).unlink(missing_ok=True)

    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        return {
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
        print(f"Device: {info['device']}")
        print(f"Output classes: {info['num_classes']}")
        print(f"m/z range: {info['mz_range'][0]}-{info['mz_range'][1]} Da")
        print(f"Bin width: {info['bin_width']} Da")
        print(f"Binary models: {info['binary_models_dir']}")
        print(f"SVM model: {info['svm_model_path']}")