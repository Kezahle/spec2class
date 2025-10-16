# Path: spec2class/src/spec2class/config/__init__.py
# This file goes in src/spec2class/config/

"""Configuration for Spec2Class models - loads from JSON file"""

import json
from pathlib import Path
from typing import List
from importlib import resources

from ..models.manager import ModelConfig, ModelSpec

# Load configuration from JSON file
try:
    # Python 3.9+
    if hasattr(resources, 'files'):
        _config_path = resources.files('spec2class.config') / 'model_config.json'
        _CONFIG_DATA = json.loads(_config_path.read_text())
    else:
        # Fallback for older Python
        import pkg_resources
        _config_text = pkg_resources.resource_string('spec2class.config', 'model_config.json')
        _CONFIG_DATA = json.loads(_config_text.decode('utf-8'))
except Exception:
    # Development mode fallback
    _CONFIG_FILE = Path(__file__).parent / "model_config.json"
    with open(_CONFIG_FILE, "r") as f:
        _CONFIG_DATA = json.load(f)

# Extract chemical classes
CHEMICAL_CLASSES = _CONFIG_DATA["binary_models"]["classes"]


def get_config_data() -> ModelConfig:
    """
    Get the model configuration for Spec2Class binary models.

    Returns:
        ModelConfig with all 43 binary models (NOT including SVM)
    """
    binary_config = _CONFIG_DATA["binary_models"]
    
    # Create model specs for each chemical class
    binary_models = {}
    filename_pattern = binary_config["filename_pattern"]
    
    for class_name in binary_config["classes"]:
        filename = filename_pattern.replace("{class_name}", class_name)
        binary_models[class_name] = ModelSpec(
            filename=filename,
            description=f"Binary classifier for {class_name}"
        )

    # Define model groups (NOTE: "svm_model" is NOT in binary models)
    groups_config = _CONFIG_DATA["model_groups"]
    model_groups = {
        "all_models": binary_config["classes"],  # Only binary models here
        "test_models": groups_config["test_models"],
        "binary_only": binary_config["classes"],
    }

    return ModelConfig(
        repo_id=binary_config["repo_id"],
        revision=binary_config["revision"],
        models=binary_models,
        model_groups=model_groups,
    )


def get_svm_config() -> ModelConfig:
    """
    Get separate configuration for SVM model from its own repo.

    Returns:
        ModelConfig for SVM model
    """
    svm_config = _CONFIG_DATA["svm_model"]
    
    return ModelConfig(
        repo_id=svm_config["repo_id"],
        revision=svm_config["revision"],
        models={
            "svm_model": ModelSpec(
                filename=svm_config["filename"],
                description="SVM model for final multiclass prediction",
            )
        },
        model_groups={"svm_only": ["svm_model"]},
    )


__all__ = ["get_config_data", "get_svm_config", "CHEMICAL_CLASSES"]