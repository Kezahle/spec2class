# Path: spec2class/src/spec2class/models/loader.py
# This file goes in src/spec2class/models/
# Hugging Face Model Loader for Spec2Class - Adapted from parent_ion_classifier

from __future__ import annotations

import pickle
import sys
from typing import Any, Dict, List, Literal, Optional

import torch
from huggingface_hub.utils import LocalEntryNotFoundError

from ..config import get_config_data, get_svm_config
from .manager import ModelManager

# Global model managers (initialized lazily)
_binary_model_managers: Dict[str, Optional[ModelManager]] = {
    "positive": None,
    "negative": None
}
_svm_model_managers: Dict[str, Optional[ModelManager]] = {
    "positive": None,
    "negative": None
}


def get_binary_model_manager(mode: Literal["positive", "negative"] = "positive") -> ModelManager:
    """Get the ModelManager instance for binary models"""
    global _binary_model_managers
    if _binary_model_managers[mode] is None:
        config_data = get_config_data(mode=mode)
        _binary_model_managers[mode] = ModelManager(config_data)
    return _binary_model_managers[mode]


def get_svm_model_manager(mode: Literal["positive", "negative"] = "positive") -> ModelManager:
    """Get the ModelManager instance for SVM model"""
    global _svm_model_managers
    if _svm_model_managers[mode] is None:
        svm_config = get_svm_config(mode=mode)
        _svm_model_managers[mode] = ModelManager(svm_config)
    return _svm_model_managers[mode]


def get_cache_directory(mode: Literal["positive", "negative"] = "positive") -> str:
    """Get the current cache directory being used"""
    return get_binary_model_manager(mode).get_cache_directory()


def print_cache_directory(mode: Literal["positive", "negative"] = "positive") -> None:
    """Print the current cache directory"""
    get_binary_model_manager(mode).print_cache_directory()


def get_available_models() -> List[str]:
    """Returns a sorted list of all available model names (same for both modes)"""
    return get_binary_model_manager("positive").get_available_models()


def get_model_groups() -> List[str]:
    """Returns a sorted list of all available model group names"""
    return get_binary_model_manager("positive").get_model_groups()


def get_models_in_group(group_name: str, mode: Literal["positive", "negative"] = "positive") -> List[str]:
    """Get models belonging to a specific group"""
    if group_name == "all_models":
        # Special case: return all binary models + SVM for the specified mode
        return get_binary_model_manager(mode).get_available_models() + ["svm_model"]
    elif group_name == "svm_only":
        return ["svm_model"]
    else:
        # For other groups, get from binary manager (mode-specific)
        binary_models = get_binary_model_manager(mode).get_models_in_group(group_name)
        # Add SVM for test_models group
        if group_name == "test_models":
            return binary_models + ["svm_model"]
        return binary_models


def is_model_cached(model_name: str, mode: Literal["positive", "negative"] = "positive") -> bool:
    """Check if a model is cached locally"""
    if model_name == "svm_model":
        return get_svm_model_manager(mode).is_model_cached(model_name)
    return get_binary_model_manager(mode).is_model_cached(model_name)


def download_model(
    model_name: str, 
    mode: Literal["positive", "negative"] = "positive",
    force: bool = False
) -> str:
    """Download a specific model"""
    if model_name == "svm_model":
        return get_svm_model_manager(mode).download_model(model_name, force=force)
    return get_binary_model_manager(mode).download_model(model_name, force=force)


def download_models(
    model_names: Optional[List[str]] = None, 
    group_name: Optional[str] = None, 
    mode: Literal["positive", "negative"] = "positive",
    force: bool = False
) -> Dict[str, str]:
    """Download multiple models"""
    results = {}

    if model_names is not None:
        models_to_download = model_names
    elif group_name is not None:
        models_to_download = get_models_in_group(group_name, mode=mode)
    else:
        models_to_download = get_available_models()

    binary_models = [m for m in models_to_download if m != "svm_model"]
    needs_svm = "svm_model" in models_to_download

    if binary_models:
        binary_results = get_binary_model_manager(mode).download_models(
            model_names=binary_models, force=force
        )
        results.update(binary_results)

    if needs_svm:
        svm_results = get_svm_model_manager(mode).download_models(
            model_names=["svm_model"], force=force
        )
        results.update(svm_results)

    return results


def check_and_download_models(
    model_names: Optional[List[str]] = None, 
    group_name: Optional[str] = None,
    mode: Literal["positive", "negative"] = "positive"
) -> bool:
    """Check for missing models and download them"""
    if model_names is not None:
        models_to_check = model_names
    elif group_name is not None:
        models_to_check = get_models_in_group(group_name, mode=mode)
    else:
        models_to_check = get_available_models()

    missing_models = []
    for model_name in models_to_check:
        if not is_model_cached(model_name, mode=mode):
            missing_models.append(model_name)

    if missing_models:
        print(f"Missing models for {mode} mode: {missing_models}")
        print("Downloading missing models...")
        try:
            download_models(model_names=missing_models, mode=mode)
            return True
        except Exception as e:
            print(f"Failed to download missing models: {e}", file=sys.stderr)
            return False

    return True


def load_binary_model(
    model_name: str,
    mode: Literal["positive", "negative"] = "positive",
    device: str = "cpu",
    local_files_only: bool | None = None,
    auto_download: bool = True,
) -> torch.nn.Module:
    """Load a binary classifier model from HuggingFace Hub"""
    manager = get_binary_model_manager(mode)

    if model_name not in manager.config.models:
        raise ValueError(f"Unknown model '{model_name}', available: {get_available_models()}")

    if local_files_only is True:
        auto_download = False
    elif local_files_only is False:
        auto_download = True

    def load_pytorch(path: str, device: str):
        return torch.load(path, map_location=device)

    model = manager.load_model(
        model_name, device=device, auto_download=auto_download, loader_func=load_pytorch
    )

    if model is None:
        if local_files_only is True or not auto_download:
            raise LocalEntryNotFoundError(f"Model '{model_name}' not found in local cache")
        else:
            raise RuntimeError(f"Failed to load model '{model_name}'")

    return model


def load_svm_model(
    mode: Literal["positive", "negative"] = "positive",
    device: str = "cpu",
    local_files_only: bool | None = None,
    auto_download: bool = True,
) -> Any:
    """Load the SVM model from HuggingFace Hub"""
    manager = get_svm_model_manager(mode)

    if local_files_only is True:
        auto_download = False
    elif local_files_only is False:
        auto_download = True

    def load_pickle(path: str, device: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    model = manager.load_model(
        "svm_model", device=device, auto_download=auto_download, loader_func=load_pickle
    )

    if model is None:
        if local_files_only is True or not auto_download:
            raise LocalEntryNotFoundError("SVM model not found in local cache")
        else:
            raise RuntimeError("Failed to load SVM model")

    return model


def load_model(
    model_name: str, 
    mode: Literal["positive", "negative"] = "positive",
    device: str = "cpu", 
    raise_exception: bool = False, 
    auto_download: bool = True
) -> Any:
    """Load a model (binary or SVM) with optional error handling"""
    try:
        if model_name == "svm_model":
            return load_svm_model(mode=mode, device=device, auto_download=auto_download)
        else:
            return load_binary_model(model_name, mode=mode, device=device, auto_download=auto_download)
    except Exception as e:
        if raise_exception:
            raise
        else:
            print(f"[ERROR] Could not load model '{model_name}': {e}", file=sys.stderr)
            return None


def clear_model_cache(
    model_names: Optional[List[str]] = None, 
    mode: Literal["positive", "negative"] = "positive",
    confirm: bool = True
) -> bool:
    """Clear cached models from local storage"""
    success = True

    if model_names is None or any(m != "svm_model" for m in model_names):
        binary_to_clear = None if model_names is None else [m for m in model_names if m != "svm_model"]
        success &= get_binary_model_manager(mode).clear_cache(model_names=binary_to_clear, confirm=confirm)

    if model_names is None or "svm_model" in model_names:
        success &= get_svm_model_manager(mode).clear_cache(model_names=["svm_model"], confirm=confirm)

    return success


def get_cache_info(mode: Literal["positive", "negative"] = "positive") -> Dict[str, Any]:
    """Get information about cached models and disk usage"""
    binary_info = get_binary_model_manager(mode).get_cache_info()
    svm_info = get_svm_model_manager(mode).get_cache_info()

    return {
        "mode": mode,
        "binary_models": binary_info,
        "svm_model": svm_info,
        "cache_dir": binary_info["cache_dir"],
    }


def print_cache_info(mode: Literal["positive", "negative"] = "positive", verbose: bool = False) -> None:
    """Print cache information in a user-friendly format"""
    info = get_cache_info(mode)

    print(f"Cache directory: {info['cache_dir']}")
    print(f"Mode: {info['mode']}")
    print("\n=== Binary Models ===")
    binary_info = info["binary_models"]
    print(f"Repository: {binary_info['repo_id']}")

    if binary_info.get("cached"):
        size_mb = binary_info["total_size"] / (1024 * 1024)
        print(f"Total cache size: {size_mb:.1f} MB")
        print(f"Number of files: {binary_info['file_count']}")
    else:
        print("No cached files found")

    print("\n=== SVM Model ===")
    svm_info = info["svm_model"]
    print(f"Repository: {svm_info['repo_id']}")

    if svm_info.get("cached"):
        size_mb = svm_info["total_size"] / (1024 * 1024)
        print(f"Total cache size: {size_mb:.1f} MB")
        print(f"Number of files: {svm_info['file_count']}")
    else:
        print("No cached files found")


def print_model_status(
    mode: Literal["positive", "negative"] = "positive",
    group_name: Optional[str] = None
) -> None:
    """Print status of models"""
    print(f"\n=== Binary Models ({mode} mode) ===")
    get_binary_model_manager(mode).print_status(group_name=group_name)

    if group_name is None or "svm_model" in get_models_in_group(group_name):
        print(f"\n=== SVM Model ({mode} mode) ===")
        get_svm_model_manager(mode).print_status()


def get_model_info(model_name: str, mode: Literal["positive", "negative"] = "positive") -> Dict[str, Any]:
    """Get detailed information about a specific model"""
    if model_name == "svm_model":
        return get_svm_model_manager(mode).get_model_info(model_name)
    return get_binary_model_manager(mode).get_model_info(model_name)


def get_missing_models(
    mode: Literal["positive", "negative"] = "positive",
    group_name: Optional[str] = None
) -> List[str]:
    """Get list of models that are not cached"""
    models_to_check = get_models_in_group(group_name, mode=mode) if group_name else get_available_models()

    missing = []
    for model_name in models_to_check:
        if not is_model_cached(model_name, mode=mode):
            missing.append(model_name)

    return missing


def get_cached_models(mode: Literal["positive", "negative"] = "positive") -> List[str]:
    """Get list of models that are currently cached"""
    binary_cached = get_binary_model_manager(mode).get_cached_models()
    svm_cached = get_svm_model_manager(mode).get_cached_models()

    return binary_cached + svm_cached