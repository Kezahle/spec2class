"""
Spec2Class: Accurate Prediction of Plant Secondary Metabolite Class using Deep Learning

Spec2Class is an ensemble classification model for predicting chemical classes
of plant secondary metabolites from high resolution LC-MS/MS spectra.
"""

__version__ = "1.0.0"
__author__ = "Vicki Pol"

from .classifier import Spec2ClassClassifier
from .data_processor import parse_mgf_file, parse_msp_file
from .models import ModelManager, download_models, get_available_models, load_model

__all__ = [
    "Spec2ClassClassifier",
    "ModelManager",
    "load_model",
    "download_models",
    "get_available_models",
    "parse_mgf_file",
    "parse_msp_file",
]