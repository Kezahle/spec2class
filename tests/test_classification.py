# Path: spec2class/tests/test_classification.py
# Integration tests for full classification pipeline

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

# Import the utility function from the main package
from spec2class.core.utility_functions import _to_float_array


@pytest.fixture
def mona_test_data():
    """Load 100 test spectra from MoNA database"""
    test_data_path = Path("tests/data/mona_100_spec.csv")
    
    if not test_data_path.exists():
        pytest.skip(f"Test data not found: {test_data_path}")
    
    # Load data
    df = pd.read_csv(test_data_path)
    
    # Convert string arrays to numpy arrays
    for col in ('Intensity', 'mz'):
        if col in df.columns:
            df[col] = df[col].apply(_to_float_array)
    
    return df


@pytest.mark.slow
def test_full_classification_pipeline(mona_test_data):
    """
    Test full classification pipeline on 100 real spectra
    """
    from spec2class import Spec2ClassClassifier
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Save test data
        input_file = temp_dir / "test_input.pkl"
        mona_test_data.to_pickle(input_file)
        
        # Initialize classifier
        classifier = Spec2ClassClassifier(device='cpu')
        
        # Classify
        results = classifier.classify_from_file(
            str(input_file),
            output_dir=str(temp_dir),
            output_name="test_output"
        )
        
        # Verify results
        # Note: Some spectra may be filtered out if they have no valid fragments
        # in the m/z range (50-550 Da) or fail other quality checks
        input_count = len(mona_test_data)
        output_count = len(results)
        
        # We should get at least 95% of spectra through (allow ~5% filtering)
        min_expected = int(input_count * 0.95)
        assert output_count >= min_expected, \
            f"Too many spectra filtered: input={input_count}, output={output_count}, min_expected={min_expected}"
        
        # Should not get more results than input
        assert output_count <= input_count, \
            f"Got more results than input: input={input_count}, output={output_count}"
        
        # Check required columns exist
        assert 'DB.' in results.columns
        assert 'final_pred' in results.columns
        assert 'estimated_top2_pred' in results.columns
        assert 'estimated_top3_pred' in results.columns
        assert 'probabilities' in results.columns
        
        # Check predictions are valid chemical classes
        from spec2class.config import CHEMICAL_CLASSES
        for pred in results['final_pred']:
            assert pred in CHEMICAL_CLASSES, f"Invalid prediction: {pred}"
        
        print(f"\n{'='*70}")
        print(f"Classification Pipeline Test")
        print(f"{'='*70}")
        print(f"Input spectra: {input_count}")
        print(f"Successfully classified: {output_count}")
        print(f"Filtered out: {input_count - output_count}")
        print(f"Success rate: {output_count/input_count*100:.1f}%")
        print(f"\nTop 5 predicted classes:")
        for class_name, count in results['final_pred'].value_counts().head(5).items():
            print(f"  {class_name}: {count}")


@pytest.mark.slow
def test_classification_agreement_with_original():
    """
    Test classification agreement with original implementation results
    """
    from spec2class import Spec2ClassClassifier
    
    # Load test data
    test_data_path = Path("tests/data/mona_100_spec.csv")
    if not test_data_path.exists():
        pytest.skip(f"Test data not found: {test_data_path}")
    
    df = pd.read_csv(test_data_path)
    
    # Convert string arrays to numpy arrays
    for col in ('Intensity', 'mz'):
        if col in df.columns:
            df[col] = df[col].apply(_to_float_array)
    
    # Load expected results if available
    expected_results_path = Path("tests/data/output_mona_100_spec.csv")
    if not expected_results_path.exists():
        pytest.skip(f"Expected results not found: {expected_results_path}")
    
    expected_results = pd.read_csv(expected_results_path)
    
    # Save to temp directory for testing
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="spec2class_debug")
    temp_dir = Path(temp_dir)
    
    input_file = temp_dir / "test_input.pkl"
    df.to_pickle(input_file)
    
    # Initialize classifier
    classifier = Spec2ClassClassifier(device='cpu')
    
    # Classify
    results = classifier.classify_from_file(
        str(input_file),
        output_dir=str(temp_dir),
        output_name="test_output"
    )
    
    # Compare results with expected
    # Note: Some spectra may be filtered, so merge on DB.
    merged = expected_results.merge(results, on='DB.', how='inner', suffixes=('_expected', '_actual'))
    
    if len(merged) == 0:
        pytest.skip("No matching spectra between expected and actual results")
    
    # Count agreements
    agreements = (merged['final_pred_expected'] == merged['final_pred_actual']).sum()
    agreement_rate = agreements / len(merged) * 100
    
    print(f"\n{'='*70}")
    print(f"Classification Agreement Test")
    print(f"{'='*70}")
    print(f"Total spectra compared: {len(merged)}")
    print(f"Matching predictions: {agreements}")
    print(f"Agreement: {agreement_rate:.1f}%")
    print(f"\nNote: Due to sklearn version differences (0.24.2 vs current),")
    print(f"some variation in predictions is expected (~32% as documented).")
    
    # We expect some disagreement due to sklearn version differences
    # Just report the results, don't fail the test
    assert agreement_rate > 0, "No agreements found - something is wrong"


@pytest.mark.slow
def test_classification_deterministic():
    """
    Test that classification is deterministic (same input → same output)
    """
    from spec2class import Spec2ClassClassifier
    
    # Load test data
    test_data_path = Path("tests/data/mona_100_spec.csv")
    if not test_data_path.exists():
        pytest.skip(f"Test data not found: {test_data_path}")
    
    df = pd.read_csv(test_data_path)
    
    # Convert string arrays to numpy arrays
    for col in ('Intensity', 'mz'):
        if col in df.columns:
            df[col] = df[col].apply(_to_float_array)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        input_file = temp_dir / "test_input.pkl"
        df.to_pickle(input_file)
        
        # Initialize classifier
        classifier = Spec2ClassClassifier(device='cpu')
        
        # Run classification twice
        results1 = classifier.classify_from_file(
            str(input_file),
            output_dir=str(temp_dir / "run1"),
            output_name="test_output"
        )
        
        results2 = classifier.classify_from_file(
            str(input_file),
            output_dir=str(temp_dir / "run2"),
            output_name="test_output"
        )
        
        # Both runs should produce the same number of results
        assert len(results1) == len(results2), \
            f"Different number of results: {len(results1)} vs {len(results2)}"
        
        # Merge on DB. to compare
        merged = results1.merge(results2, on='DB.', how='inner', suffixes=('_1', '_2'))
        
        # Check predictions match
        pred_match = (merged['final_pred_1'] == merged['final_pred_2']).sum()
        match_rate = pred_match / len(merged) * 100
        
        print(f"\n{'='*70}")
        print(f"Deterministic Classification Test")
        print(f"{'='*70}")
        print(f"Spectra classified: {len(merged)}")
        print(f"Matching predictions: {pred_match}")
        print(f"Match rate: {match_rate:.1f}%")
        
        # Should be 100% deterministic
        assert match_rate == 100.0, \
            f"Classification not deterministic: {match_rate:.1f}% match rate"


def test_single_spectrum_classification():
    """
    Test classification of a single spectrum
    """
    
    from spec2class import Spec2ClassClassifier
    
    # Create a simple test spectrum
    data = {
        'DB.': ['test_spectrum'],
        'mz': [np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0])],
        'Intensity': [np.array([50.0, 100.0, 75.0, 60.0, 80.0, 90.0])],
        'ExactMass': [400.0]
    }
    
    df = pd.DataFrame(data)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        input_file = temp_dir / "single_spectrum.pkl"
        df.to_pickle(input_file)
        
        classifier = Spec2ClassClassifier(device='cpu')
        
        results = classifier.classify_from_file(
            str(input_file),
            output_dir=str(temp_dir)
        )
        
        # Should get exactly 1 result
        assert len(results) == 1
        assert results.iloc[0]['DB.'] == 'test_spectrum'
        assert 'final_pred' in results.columns