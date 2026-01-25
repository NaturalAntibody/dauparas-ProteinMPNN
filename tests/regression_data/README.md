# Regression Test Data

This directory contains precomputed outputs from `tied_featurize_orig()` function for various parameter combinations. These outputs serve as the "golden" reference values for regression testing.

## Purpose

During refactoring of `tied_featurize_orig()`, these saved outputs ensure that the refactored code produces **exactly** the same results as the original implementation. Any difference in output values indicates a behavioral change that needs to be investigated.

## Structure

Each test case is stored in its own subdirectory:

```
regression_data/
├── basic_single_chain_design/
│   ├── inputs.json              # Test parameters (metadata)
│   ├── 00_X.pt                  # Coordinates tensor
│   ├── 01_S.pt                  # Sequence tensor
│   ├── 02_mask.pt               # Mask tensor
│   ├── 03_lengths.npy           # Lengths array
│   ├── ...                      # All 20 outputs
│   └── 19_tied_beta.pt
├── basic_multi_chain_design/
├── ca_only_mode/
├── fixed_positions_single_chain/
├── omit_aa_single_chain/
├── tied_positions_within_chain/
├── pssm_single_chain/
└── batch_processing_3_structures/
```

## Test Cases

1. **basic_single_chain_design**: Single chain (H) designed, single chain (L) fixed
2. **basic_multi_chain_design**: Multiple chains (H, L) designed, no fixed chains
3. **ca_only_mode**: CA-only mode with single designed chain
4. **fixed_positions_single_chain**: Fixed positions (1-10) on chain H
5. **omit_aa_single_chain**: Omit C and M at positions 5, 10, 15 on chain H
6. **tied_positions_within_chain**: Positions 10, 20, 30 tied together on chain H
7. **pssm_single_chain**: PSSM with coef=0.5 and random bias/log_odds (seed=42)
8. **batch_processing_3_structures**: Batch of 3 identical structures

## Regenerating Data

⚠️ **WARNING**: Only regenerate if you're absolutely certain the current implementation is correct!

```bash
python scripts/generate_regression_data.py
```

This will overwrite all existing regression data. You should:
1. Review changes carefully using `git diff`
2. Verify all tests still pass
3. Document why the regeneration was necessary in the commit message

## File Formats

- **`.pt`**: PyTorch tensors (saved with `torch.save()`)
- **`.npy`**: NumPy arrays (saved with `np.save()`)
- **`.json`**: Python lists and metadata (saved as JSON)

## Usage in Tests

See `tests/test_tied_featurize_regression.py` for how these files are loaded and compared:

```python
# Load expected outputs
expected = load_regression_data("test_case_name")

# Compute actual outputs
actual = tied_featurize(...)

# Assert exact equality
assert_outputs_equal(actual, expected)
```

## Input Data

All tests use the same PDB structure:
- **File**: `/home/bartosz.janusz/antifold-interface/data/input_files/5GGS_standardized.pdb`
- **Chains**: H (heavy), L (light), A (antigen)
- **Note**: Structure is filtered using `select_chains()` before passing to `tied_featurize_orig()`

## Version Control

This data is tracked in Git to ensure:
1. Changes to output values are explicitly reviewed
2. Historical behavior can be recovered if needed
3. Different branches can validate against their own baseline

If a refactoring intentionally changes behavior, regenerate the data and document the change.
