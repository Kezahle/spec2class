# Path: spec2class/src/spec2class/config/__init__.py
# This file goes in src/spec2class/config/

"""Configuration for Spec2Class models - loads from JSON file"""

import json
from pathlib import Path
from typing import List, Literal
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

# Extract chemical classes - now mode-specific
CHEMICAL_CLASSES_POSITIVE = _CONFIG_DATA["binary_models"]["positive"]["classes"]
CHEMICAL_CLASSES_NEGATIVE = _CONFIG_DATA["binary_models"]["negative"]["classes"]

# For backward compatibility, default to positive
CHEMICAL_CLASSES = CHEMICAL_CLASSES_POSITIVE


def get_chemical_classes(mode: Literal["positive", "negative"] = "positive") -> List[str]:
    """
    Get the list of chemical classes for the specified mode.
    
    Args:
        mode: Ionization mode - 'positive' or 'negative'
        
    Returns:
        List of chemical class names for the specified mode
    """
    if mode == "positive":
        return CHEMICAL_CLASSES_POSITIVE
    elif mode == "negative":
        return CHEMICAL_CLASSES_NEGATIVE
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'positive' or 'negative'.")


def get_config_data(mode: Literal["positive", "negative"] = "positive") -> ModelConfig:
    """
    Get the model configuration for Spec2Class binary models.

    Args:
        mode: Ionization mode - 'positive' or 'negative'

    Returns:
        ModelConfig with all binary models for the specified mode (NOT including SVM)
    """
    binary_config = _CONFIG_DATA["binary_models"]
    mode_config = binary_config[mode]
    
    # Create model specs for each chemical class
    binary_models = {}
    filename_pattern = mode_config["filename_pattern"]
    
    # Get mode-specific classes
    classes = mode_config["classes"]
    
    for class_name in classes:
        filename = filename_pattern.replace("{class_name}", class_name)
        binary_models[class_name] = ModelSpec(
            filename=filename,
            description=f"Binary classifier for {class_name} ({mode} mode)"
        )

    # Define model groups - test_models uses Flavonoids and Steroids if available
    test_models = []
    if mode == "positive":
        test_models = ["Flavonoids", "Steroids"]
    else:  # negative mode
        test_models = ["Flavonoids", "Steroids"]  # Both exist in negative mode
    
    model_groups = {
        "all_models": classes,  # Only binary models here (mode-specific)
        "test_models": test_models,
        "binary_only": classes,
    }

    return ModelConfig(
        repo_id=mode_config["repo_id"],
        revision=mode_config["revision"],
        models=binary_models,
        model_groups=model_groups,
    )


def get_svm_config(mode: Literal["positive", "negative"] = "positive") -> ModelConfig:
    """
    Get separate configuration for SVM model from its own repo.

    Args:
        mode: Ionization mode - 'positive' or 'negative'

    Returns:
        ModelConfig for SVM model
    """
    svm_config = _CONFIG_DATA["svm_model"]
    mode_config = svm_config[mode]
    
    return ModelConfig(
        repo_id=mode_config["repo_id"],
        revision=mode_config["revision"],
        models={
            "svm_model": ModelSpec(
                filename=mode_config["filename"],
                description=f"SVM model for final multiclass prediction ({mode} mode)",
            )
        },
        model_groups={"svm_only": ["svm_model"]},
    )


__all__ = ["get_config_data", "get_svm_config", "get_chemical_classes", 
           "CHEMICAL_CLASSES", "CHEMICAL_CLASSES_POSITIVE", "CHEMICAL_CLASSES_NEGATIVE"]