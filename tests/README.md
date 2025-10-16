# Spec2Class Tests

This directory contains tests for the Spec2Class package.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration
├── test_basic.py           # Fast unit tests (imports, config)
├── test_classification.py  # Integration tests with real data
├── data/                   # Test data files
│   ├── mona_100_spec.csv          # Input test spectra
│   └── output_mona_100_spec.csv   # Expected output (optional)
└── README.md               # This file
```

## Running Tests

### Quick Tests (Fast, No Model Downloads)

Run only the fast unit tests that don't require model downloads:

```bash
pytest tests/test_basic.py -v
```

Or skip slow tests explicitly:

```bash
pytest -m "not slow" -v
```

### Full Tests (Slow, Downloads Models)

Run all tests including model downloads and classification:

```bash
pytest --runslow -v
```

### Integration Tests Only

Test full classification pipeline with real data:

```bash
pytest tests/test_classification.py --runslow -v
```

## Test Categories

### Fast Tests (`test_basic.py`)
- ✅ Import verification
- ✅ Configuration loading
- ✅ Model manager functionality
- ✅ Model groups and lists
- ⏱️ **Runtime**: < 1 second

### Slow Tests (marked with `@pytest.mark.slow`)
- 🔽 Model downloading
- 🔽 Classifier initialization
- 🔽 Full classification pipeline
- ⏱️ **Runtime**: 2-5 minutes (first run with downloads)

### Integration Tests (`test_classification.py`)
- 📊 Full classification on 100 test spectra
- 📊 Comparison with original results
- 📊 Deterministic behavior verification
- ⏱️ **Runtime**: 5-10 minutes

## Test Data

### Required Files

Place these files in `tests/data/`:

1. **`mona_100_spec.csv`** (Required)
   - 100 test spectra from MoNA database
   - Columns: `DB.`, `mz`, `Intensity`, `ExactMass`
   - Used for integration testing

2. **`output_mona_100_spec.csv`** (Optional)
   - Expected classification results from original implementation
   - Used for comparison/validation
   - If missing, comparison test is skipped

### Data Format

Input CSV should have:
```csv
DB.,mz,Intensity,ExactMass
spectrum_1,"[100.0 150.0 200.0]","[50.0 100.0 75.0]",300.5
spectrum_2,"[120.0 180.0 250.0]","[60.0 90.0 100.0]",350.2
```

## Expected Results

### Fast Tests
```bash
$ pytest -m "not slow" -v

test_basic.py::test_imports PASSED
test_basic.py::test_parse_mgf PASSED
test_basic.py::test_parse_msp PASSED
test_basic.py::test_model_config PASSED
test_basic.py::test_model_groups PASSED
test_basic.py::test_model_manager PASSED

====== 6 passed in 0.5s ======
```

### Full Integration Test
```bash
$ pytest tests/test_classification.py::test_classification_agreement_with_original --runslow -v

test_classification.py::test_classification_agreement_with_original PASSED

======================================================================
Classification Agreement Test
======================================================================
Total spectra: 100
Matching predictions: 100
Agreement: 100.0%

====== 1 passed in 8m 23s ======
```

## Troubleshooting

### Test Data Not Found
```
SKIPPED [1] test_classification.py: Test data not found: tests/data/mona_100_spec.csv
```

**Solution**: Copy test data to `tests/data/` directory

### Slow Tests Skipped
```
SKIPPED [6] conftest.py: need --runslow option to run
```

**Solution**: Add `--runslow` flag to pytest command

### Model Download Failures

If tests fail due to network issues:

1. Check internet connection
2. Try downloading models manually:
   ```bash
   spec2class download --group all_models
   ```
3. Verify cache:
   ```bash
   spec2class cache info
   ```

## Writing New Tests

### Adding a Fast Test
```python
def test_my_feature():
    """Test description"""
    # Your test code
    assert result == expected
```

### Adding a Slow Test
```python
@pytest.mark.slow
def test_my_slow_feature():
    """Test that requires models or is time-consuming"""
    # Your test code
    assert result == expected
```

### Using Test Data
```python
@pytest.mark.slow
def test_with_data(mona_test_data):
    """Test using the mona fixture"""
    # mona_test_data is automatically loaded
    assert len(mona_test_data) == 100
```

## CI/CD Integration

For continuous integration, use:

```yaml
# Fast tests only (no model downloads)
- name: Run fast tests
  run: pytest -m "not slow" -v

# Full tests (with model downloads)
- name: Run all tests
  run: pytest --runslow -v
```

## Test Coverage

Run with coverage:

```bash
pytest --cov=spec2class --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html
```