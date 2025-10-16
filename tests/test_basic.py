# Path: spec2class/tests/test_basic.py
# Basic unit tests for Spec2Class

import numpy as np
import pandas as pd
import pytest


def test_imports():
    """Test that all main modules can be imported"""
    import spec2class
    from spec2class import Spec2ClassClassifier
    from spec2class.models import ModelManager
    from spec2class.config import CHEMICAL_CLASSES
    
    assert len(CHEMICAL_CLASSES) == 43
    assert spec2class.__version__ == "1.0.0"


def test_parse_mgf():
    """Test MGF file parsing"""
    from spec2class.data_processor import parse_mgf_file
    # This just tests that the function exists and can be imported
    assert callable(parse_mgf_file)


def test_parse_msp():
    """Test MSP file parsing"""
    from spec2class.data_processor import parse_msp_file
    # This just tests that the function exists and can be imported
    assert callable(parse_msp_file)


def test_model_config():
    """Test model configuration loading"""
    from spec2class.config import get_config_data, get_svm_config, CHEMICAL_CLASSES
    
    # Test binary models config
    binary_config = get_config_data()
    assert binary_config.repo_id == "VickiPol/binary_models"
    assert len(binary_config.models) == 43
    assert "Flavonoids" in binary_config.models
    
    # Test SVM config
    svm_config = get_svm_config()
    assert svm_config.repo_id == "VickiPol/SVM_model"
    assert "svm_model" in svm_config.models
    
    # Test chemical classes
    assert len(CHEMICAL_CLASSES) == 43
    assert "Flavonoids" in CHEMICAL_CLASSES


def test_model_groups():
    """Test model group functionality"""
    from spec2class.models import get_model_groups, get_models_in_group
    
    groups = get_model_groups()
    assert "all_models" in groups
    assert "test_models" in groups
    
    # Test all_models group
    all_models = get_models_in_group("all_models")
    assert len(all_models) == 44  # 43 binary + 1 SVM
    assert "svm_model" in all_models
    
    # Test test_models group
    test_models = get_models_in_group("test_models")
    assert "Flavonoids" in test_models
    assert "Steroids" in test_models
    assert "svm_model" in test_models


def test_model_manager():
    """Test ModelManager basic functionality"""
    from spec2class.models import ModelManager
    from spec2class.config import get_config_data, get_svm_config
    
    # Test binary model manager
    binary_config = get_config_data()
    binary_manager = ModelManager(binary_config)
    assert binary_manager is not None
    assert len(binary_manager.get_available_models()) == 43
    
    # Test SVM model manager
    svm_config = get_svm_config()
    svm_manager = ModelManager(svm_config)
    assert svm_manager is not None
    assert "svm_model" in svm_manager.get_available_models()


@pytest.mark.slow
def test_model_download():
    """Test downloading a single model (slow test)"""
    from spec2class.models import download_model, is_model_cached
    
    # Download Flavonoids model
    model_path = download_model("Flavonoids")
    assert model_path is not None
    assert is_model_cached("Flavonoids")


@pytest.mark.slow
def test_classifier_initialization():
    """Test classifier initialization (slow - downloads models)"""
    from spec2class import Spec2ClassClassifier
    
    # This will download models if not cached
    classifier = Spec2ClassClassifier(device='cpu')
    
    assert classifier.device == 'cpu'
    assert len(classifier.chemclass_list) == 43
    assert classifier.binary_models_dir is not None
    assert classifier.svm_model_path is not None


@pytest.mark.slow
def test_classification():
    """Test full classification pipeline (slow)"""
    from spec2class import Spec2ClassClassifier
    import pandas as pd
    import numpy as np
    import tempfile
    from pathlib import Path
    
    classifier = Spec2ClassClassifier(device='cpu')
    
    # Create test data
    data = {
        'DB.': ['test_001'],
        'mz': [np.array([100.0, 150.0, 200.0, 250.0, 300.0])],
        'Intensity': [np.array([50.0, 100.0, 75.0, 60.0, 80.0])],
        'ExactMass': [350.0]
    }
    df = pd.DataFrame(data)
    
    # Use temp directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pkl = Path(temp_dir) / "test.pkl"
        df.to_pickle(temp_pkl)
        
        # Classify
        results = classifier.classify_from_file(str(temp_pkl), output_dir=temp_dir)
        
        assert len(results) == 1
        assert 'DB.' in results.columns
        assert 'final_pred' in results.columns
        assert 'estimated_top2_pred' in results.columns
        assert 'probabilities' in results.columns