# Path: spec2class/src/spec2class/core/__init__.py
# This file goes in src/spec2class/core/

"""
Original Spec2Class core algorithms and functions.
This module contains the unmodified classification logic.
"""

from .binning_functions import (
    binning_df,
    binn_spectrum,
    get_ms2,
    ms2tobins,
    new_df,
    norm_ms2,
)
from .dataset_class import Dataset
from .multiclass_prediction_functions import svm_pred
from .neural_net import Net
from .prediction_vectors_functions import (
    create_dataloader,
    get_pred_vec_dfs,
    get_pred_vecs,
    test_model_1_batch,
    test_model_batches,
)
from .utility_functions import (
    get_net,
    import_nn,
    make_folder,
    read_df_and_format_mz_intensity_arrays,
    save_df,
)

__all__ = [
    # Binning
    "binning_df",
    "binn_spectrum",
    "get_ms2",
    "ms2tobins",
    "new_df",
    "norm_ms2",
    # Dataset
    "Dataset",
    # Neural Network
    "Net",
    # Predictions
    "create_dataloader",
    "get_pred_vec_dfs",
    "get_pred_vecs",
    "test_model_1_batch",
    "test_model_batches",
    # SVM
    "svm_pred",
    # Utilities
    "get_net",
    "import_nn",
    "make_folder",
    "read_df_and_format_mz_intensity_arrays",
    "save_df",
]