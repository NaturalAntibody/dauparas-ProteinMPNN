"""Regression tests for tied_featurize function.

These tests compare current outputs against saved precomputed values to detect
any unintended changes in behavior during refactoring.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dauparas_proteinmpnn.io import parse_pdb, select_chains
from dauparas_proteinmpnn.featurize import tied_featurize_orig as tied_featurize


# Paths
PDB_PATH = Path("/home/bartosz.janusz/antifold-interface/data/input_files/5GGS_standardized.pdb")
REGRESSION_DATA_DIR = Path(__file__).parent / "regression_data"


@pytest.fixture
def device():
    """Fixture for torch device."""
    return torch.device("cpu")


@pytest.fixture
def structure():
    """Fixture to load the test PDB structure (all chains)."""
    with open(PDB_PATH, "r") as f:
        return parse_pdb(f)


def prepare_structure_for_chains(structure: dict, designed_chains: list[str], fixed_chains: list[str]) -> dict:
    """Helper to select only the chains we're working with."""
    all_chains = designed_chains + fixed_chains
    return select_chains(structure, all_chains)


def load_regression_data(test_case_name: str) -> tuple:
    """Load precomputed regression data for a test case."""
    test_dir = REGRESSION_DATA_DIR / test_case_name
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Regression data not found: {test_dir}")
    
    # Output names in order
    output_names = [
        "X", "S", "mask", "lengths", "chain_M", "chain_encoding_all", "chain_list_list",
        "visible_list_list", "masked_list_list", "masked_chain_length_list_list",
        "chain_M_pos", "omit_AA_mask", "residue_idx", "dihedral_mask",
        "tied_pos_list_of_lists_list", "pssm_coef", "pssm_bias", "pssm_log_odds_all",
        "bias_by_res_all", "tied_beta"
    ]
    
    outputs = []
    for i, name in enumerate(output_names):
        file_prefix = f"{i:02d}_{name}"
        
        # Try different file extensions
        pt_file = test_dir / f"{file_prefix}.pt"
        npy_file = test_dir / f"{file_prefix}.npy"
        json_file = test_dir / f"{file_prefix}.json"
        
        if pt_file.exists():
            outputs.append(torch.load(pt_file))
        elif npy_file.exists():
            outputs.append(np.load(npy_file))
        elif json_file.exists():
            with open(json_file, "r") as f:
                outputs.append(json.load(f))
        else:
            raise FileNotFoundError(f"Output file not found for {name} in {test_dir}")
    
    return tuple(outputs)


def assert_tensors_equal(actual: torch.Tensor, expected: torch.Tensor, name: str):
    """Assert two tensors are exactly equal."""
    assert actual.shape == expected.shape, (
        f"{name}: Shape mismatch - actual {actual.shape} vs expected {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{name}: Dtype mismatch - actual {actual.dtype} vs expected {expected.dtype}"
    )
    assert torch.equal(actual, expected), (
        f"{name}: Values don't match. Max diff: {(actual - expected).abs().max().item()}"
    )


def assert_arrays_equal(actual: np.ndarray, expected: np.ndarray, name: str):
    """Assert two numpy arrays are exactly equal."""
    assert actual.shape == expected.shape, (
        f"{name}: Shape mismatch - actual {actual.shape} vs expected {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{name}: Dtype mismatch - actual {actual.dtype} vs expected {expected.dtype}"
    )
    np.testing.assert_array_equal(actual, expected, err_msg=f"{name}: Values don't match")


def assert_lists_equal(actual: list, expected: list, name: str):
    """Assert two lists are exactly equal."""
    assert actual == expected, f"{name}: Lists don't match - actual {actual} vs expected {expected}"


def assert_outputs_equal(actual_outputs: tuple, expected_outputs: tuple):
    """Assert all outputs match exactly."""
    output_names = [
        "X", "S", "mask", "lengths", "chain_M", "chain_encoding_all", "chain_list_list",
        "visible_list_list", "masked_list_list", "masked_chain_length_list_list",
        "chain_M_pos", "omit_AA_mask", "residue_idx", "dihedral_mask",
        "tied_pos_list_of_lists_list", "pssm_coef", "pssm_bias", "pssm_log_odds_all",
        "bias_by_res_all", "tied_beta"
    ]
    
    assert len(actual_outputs) == len(expected_outputs) == 20, "Expected 20 outputs"
    
    for i, (actual, expected, name) in enumerate(zip(actual_outputs, expected_outputs, output_names)):
        if isinstance(expected, torch.Tensor):
            assert_tensors_equal(actual, expected, name)
        elif isinstance(expected, np.ndarray):
            assert_arrays_equal(actual, expected, name)
        elif isinstance(expected, list):
            assert_lists_equal(actual, expected, name)
        else:
            assert actual == expected, f"{name}: Values don't match - actual {actual} vs expected {expected}"


class TestRegressionBasic:
    """Regression tests for basic featurization scenarios."""
    
    def test_basic_single_chain_design(self, structure, device):
        """Test basic single chain design matches regression data."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        chain_dict = {structure_filtered["name"]: (designed_chains, fixed_chains)}
        
        # Compute current output
        actual = tied_featurize([structure_filtered], device, chain_dict)
        
        # Load expected output
        expected = load_regression_data("basic_single_chain_design")
        
        # Compare
        assert_outputs_equal(actual, expected)
    
    def test_basic_multi_chain_design(self, structure, device):
        """Test basic multi-chain design matches regression data."""
        designed_chains = ["H", "L"]
        fixed_chains = []
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        chain_dict = {structure_filtered["name"]: (designed_chains, fixed_chains)}
        
        # Compute current output
        actual = tied_featurize([structure_filtered], device, chain_dict)
        
        # Load expected output
        expected = load_regression_data("basic_multi_chain_design")
        
        # Compare
        assert_outputs_equal(actual, expected)
    
    def test_ca_only_mode(self, structure, device):
        """Test CA-only mode matches regression data."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        chain_dict = {structure_filtered["name"]: (designed_chains, fixed_chains)}
        
        # Compute current output
        actual = tied_featurize([structure_filtered], device, chain_dict, ca_only=True)
        
        # Load expected output
        expected = load_regression_data("ca_only_mode")
        
        # Compare
        assert_outputs_equal(actual, expected)


class TestRegressionFixedPositions:
    """Regression tests for fixed positions functionality."""
    
    def test_fixed_positions_single_chain(self, structure, device):
        """Test fixed positions on single chain matches regression data."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        protein_name = structure_filtered["name"]
        chain_dict = {protein_name: (designed_chains, fixed_chains)}
        fixed_position_dict = {protein_name: {"H": list(range(1, 11))}}
        
        # Compute current output
        actual = tied_featurize(
            [structure_filtered], 
            device, 
            chain_dict,
            fixed_position_dict=fixed_position_dict
        )
        
        # Load expected output
        expected = load_regression_data("fixed_positions_single_chain")
        
        # Compare
        assert_outputs_equal(actual, expected)


class TestRegressionOmitAA:
    """Regression tests for omit AA functionality."""
    
    def test_omit_aa_single_chain(self, structure, device):
        """Test omit AA on single chain matches regression data."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        protein_name = structure_filtered["name"]
        chain_dict = {protein_name: (designed_chains, fixed_chains)}
        omit_AA_dict = {
            protein_name: {
                "H": [
                    (np.array([5, 10, 15]), ["C", "M"]),
                ]
            }
        }
        
        # Compute current output
        actual = tied_featurize(
            [structure_filtered],
            device,
            chain_dict,
            omit_AA_dict=omit_AA_dict
        )
        
        # Load expected output
        expected = load_regression_data("omit_aa_single_chain")
        
        # Compare
        assert_outputs_equal(actual, expected)


class TestRegressionTiedPositions:
    """Regression tests for tied positions functionality."""
    
    def test_tied_positions_within_chain(self, structure, device):
        """Test tied positions within a chain matches regression data."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        protein_name = structure_filtered["name"]
        chain_dict = {protein_name: (designed_chains, fixed_chains)}
        tied_positions_dict = {
            protein_name: [
                {"H": [10, 20, 30]}
            ]
        }
        
        # Compute current output
        actual = tied_featurize(
            [structure_filtered],
            device,
            chain_dict,
            tied_positions_dict=tied_positions_dict
        )
        
        # Load expected output
        expected = load_regression_data("tied_positions_within_chain")
        
        # Compare
        assert_outputs_equal(actual, expected)


class TestRegressionPSSM:
    """Regression tests for PSSM functionality."""
    
    def test_pssm_single_chain(self, structure, device):
        """Test PSSM on single chain matches regression data."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        protein_name = structure_filtered["name"]
        chain_dict = {protein_name: (designed_chains, fixed_chains)}
        seq_len_H = len(structure_filtered["seq_chain_H"])
        pssm_dict = {
            protein_name: {
                "H": {
                    "pssm_coef": np.ones(seq_len_H) * 0.5,
                    "pssm_bias": np.random.randn(seq_len_H, 21) * 0.1,
                    "pssm_log_odds": np.random.randn(seq_len_H, 21) * 2.0,
                }
            }
        }
        
        # Note: PSSM uses random values, so we need to set seed for reproducibility
        np.random.seed(42)
        pssm_dict = {
            protein_name: {
                "H": {
                    "pssm_coef": np.ones(seq_len_H) * 0.5,
                    "pssm_bias": np.random.randn(seq_len_H, 21) * 0.1,
                    "pssm_log_odds": np.random.randn(seq_len_H, 21) * 2.0,
                }
            }
        }
        
        # Compute current output
        actual = tied_featurize(
            [structure_filtered],
            device,
            chain_dict,
            pssm_dict=pssm_dict
        )
        
        # Load expected output
        expected = load_regression_data("pssm_single_chain")
        
        # Compare
        assert_outputs_equal(actual, expected)


class TestRegressionBatch:
    """Regression tests for batch processing."""
    
    def test_batch_processing_3_structures(self, structure, device):
        """Test batch processing with 3 structures matches regression data."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        protein_name = structure_filtered["name"]
        chain_dict = {protein_name: (designed_chains, fixed_chains)}
        batch = [structure_filtered, structure_filtered, structure_filtered]
        
        # Compute current output
        actual = tied_featurize(batch, device, chain_dict)
        
        # Load expected output
        expected = load_regression_data("batch_processing_3_structures")
        
        # Compare
        assert_outputs_equal(actual, expected)
