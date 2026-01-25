"""Legacy structural validation tests for tied_featurize function.

DEPRECATED: For regression testing (comparing outputs against saved values),
use test_tied_featurize_regression.py instead.

These tests validate structural properties (shapes, dtypes, ranges) but don't
compare against saved precomputed values. They're useful for understanding the
function's behavior but not for preventing regressions during refactoring.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dauparas_proteinmpnn.io import parse_pdb, select_chains
from dauparas_proteinmpnn.featurize import tied_featurize_orig as tied_featurize


# Test data path
PDB_PATH = Path("/home/bartosz.janusz/antifold-interface/data/input_files/5GGS_standardized.pdb")


@pytest.fixture
def device():
    """Fixture for torch device."""
    return torch.device("cpu")


def prepare_structure_for_chains(structure: dict, designed_chains: list[str], fixed_chains: list[str]) -> dict:
    """Helper to select only the chains we're working with, mimicking featurize_structure behavior."""
    all_chains = designed_chains + fixed_chains
    return select_chains(structure, all_chains)


@pytest.fixture
def structure():
    """Fixture to load the test PDB structure (all chains)."""
    with open(PDB_PATH, "r") as f:
        structures = [parse_pdb(f)]
    return structures[0]


@pytest.fixture
def structure_batch(structure):
    """Fixture to create a batch with single structure."""
    return [structure]


class TestBasicFeaturization:
    """Test basic featurization without optional parameters."""

    def test_basic_single_chain_design(self, structure, device):
        """Test featurizing with single chain to design."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        chain_dict = {
            structure_filtered["name"]: (designed_chains, fixed_chains)
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
        )

        # Unpack results
        (
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list,
            visible_list_list, masked_list_list, masked_chain_length_list_list,
            chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask,
            tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all,
            bias_by_res_all, tied_beta
        ) = result

        # Basic shape checks
        B = len(structure_batch)
        L_max = max([len(s["seq"]) for s in structure_batch])
        
        assert X.shape[0] == B, "Batch size mismatch"
        assert X.shape[1] == L_max, "Sequence length mismatch"
        assert X.shape[2] == 4, "Should have 4 atoms (N, CA, C, O)"
        assert X.shape[3] == 3, "Should have 3D coordinates"
        
        assert S.shape == (B, L_max), "Sequence tensor shape mismatch"
        assert mask.shape == (B, L_max), "Mask shape mismatch"
        assert chain_M.shape == (B, L_max), "Chain mask shape mismatch"
        
        # Check that designed chain has chain_M = 1
        assert chain_M.sum() > 0, "No positions marked for design"
        
        # Check lengths
        assert len(lengths) == B
        assert all(lengths > 0), "All sequences should have positive length"

    def test_basic_multi_chain_design(self, structure, device):
        """Test featurizing with multiple chains to design."""
        designed_chains = ["H", "L"]
        fixed_chains = []
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        chain_dict = {
            structure_filtered["name"]: (designed_chains, fixed_chains)
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
        )

        X, S, mask, lengths, chain_M = result[:5]
        
        # Check that more positions are marked for design
        assert chain_M.sum() > 0, "No positions marked for design"

    def test_ca_only_mode(self, structure, device):
        """Test CA-only featurization mode."""
        # Parse PDB in CA-only mode
        with open(PDB_PATH, "r") as f:
            # Note: parse_pdb doesn't have ca_only parameter yet, so we'll test
            # with the ca_only parameter in tied_featurize
            pass
        
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        chain_dict = {
            structure_filtered["name"]: (["H"], ["L"])
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            ca_only=True,
        )

        X = result[0]
        
        # In CA-only mode, X should have shape [B, L_max, 3] (atom dimension removed)
        assert len(X.shape) == 3, "CA-only mode should have 3 dimensions [B, L_max, 3]"
        assert X.shape[2] == 3, "Last dimension should be 3 for coordinates"


class TestFixedPositions:
    """Test fixed positions functionality."""

    def test_fixed_positions_single_chain(self, structure, device):
        """Test with fixed positions on a single chain."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (["H"], ["L"])
        }
        
        # Fix positions 1-10 on chain H (1-based indexing)
        fixed_position_dict = {
            protein_name: {
                "H": list(range(1, 11))
            }
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            fixed_position_dict=fixed_position_dict,
        )

        chain_M_pos = result[10]
        
        # Fixed positions should have chain_M_pos = 0
        # Non-fixed designed positions should have chain_M_pos = 1
        assert chain_M_pos.sum() < chain_M_pos.numel(), "Some positions should be fixed"

    def test_fixed_positions_multiple_chains(self, structure, device):
        """Test with fixed positions on multiple chains."""
        designed_chains = ["H", "L"]
        fixed_chains = []
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (["H", "L"], [])
        }
        
        # Fix positions on both chains
        fixed_position_dict = {
            protein_name: {
                "H": list(range(1, 6)),
                "L": list(range(1, 4))
            }
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            fixed_position_dict=fixed_position_dict,
        )

        chain_M_pos = result[10]
        
        assert chain_M_pos.sum() < chain_M_pos.numel(), "Some positions should be fixed"


class TestOmitAA:
    """Test amino acid omission functionality."""

    def test_omit_aa_single_chain(self, structure, device):
        """Test omitting amino acids at specific positions."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (["H"], ["L"])
        }
        
        # Omit cysteine and proline at positions 10-20 on chain H
        positions = np.array(list(range(10, 21)))  # 1-based
        omit_AA_dict = {
            protein_name: {
                "H": [(positions, ["C", "P"])]
            }
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            omit_AA_dict=omit_AA_dict,
        )

        omit_AA_mask = result[11]
        
        # Check that mask has been applied (1 = omit, 0 = allow)
        assert omit_AA_mask.sum() > 0, "Some amino acids should be omitted"

    def test_omit_aa_multiple_positions(self, structure, device):
        """Test omitting different amino acids at different position ranges."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (["H"], ["L"])
        }
        
        # Multiple omission rules
        omit_AA_dict = {
            protein_name: {
                "H": [
                    (np.array([10, 11, 12]), ["C", "P"]),
                    (np.array([20, 21, 22]), ["M", "W"]),
                ]
            }
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            omit_AA_dict=omit_AA_dict,
        )

        omit_AA_mask = result[11]
        
        assert omit_AA_mask.sum() > 0, "Some amino acids should be omitted"


class TestTiedPositions:
    """Test tied positions functionality."""

    def test_tied_positions_within_chain(self, structure, device):
        """Test tying positions within the same chain."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (["H"], ["L"])
        }
        
        # Tie positions 5 and 50 on chain H to have the same amino acid
        tied_positions_dict = {
            protein_name: [
                {"H": [5, 50]}
            ]
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            tied_positions_dict=tied_positions_dict,
        )

        tied_pos_list_of_lists_list = result[14]
        tied_beta = result[19]
        
        # Check that tied positions are recorded
        assert len(tied_pos_list_of_lists_list) > 0, "Tied positions should be recorded"
        assert tied_beta.sum() > 0, "Tied beta should have non-zero values"

    def test_tied_positions_across_chains(self, structure, device):
        """Test tying positions across different chains."""
        designed_chains = ["H", "L"]
        fixed_chains = []
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (["H", "L"], [])
        }
        
        # Tie positions across chains
        tied_positions_dict = {
            protein_name: [
                {"H": [10], "L": [15]}
            ]
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            tied_positions_dict=tied_positions_dict,
        )

        tied_pos_list_of_lists_list = result[14]
        tied_beta = result[19]
        
        assert len(tied_pos_list_of_lists_list) > 0, "Tied positions should be recorded"
        assert tied_beta.sum() > 0, "Tied beta should have non-zero values"

    def test_multiple_tied_groups(self, structure, device):
        """Test multiple independent tied position groups."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (["H"], ["L"])
        }
        
        # Multiple tied groups
        tied_positions_dict = {
            protein_name: [
                {"H": [5, 50]},    # Group 1
                {"H": [10, 60]},   # Group 2
            ]
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            tied_positions_dict=tied_positions_dict,
        )

        tied_pos_list_of_lists_list = result[14]
        
        assert len(tied_pos_list_of_lists_list) > 0, "Tied positions should be recorded"


class TestPSSM:
    """Test PSSM (Position-Specific Scoring Matrix) functionality."""

    def test_pssm_single_chain(self, structure, device):
        """Test PSSM with single chain."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (["H"], ["L"])
        }
        
        # Get chain length
        seq_len_H = len(structure_batch[0]["seq_chain_H"])
        
        # Create PSSM matrices
        pssm_coef = np.ones(seq_len_H) * 0.5
        pssm_bias = np.zeros((seq_len_H, 21))
        pssm_log_odds = np.zeros((seq_len_H, 21))
        
        pssm_dict = {
            protein_name: {
                "H": {
                    "pssm_coef": pssm_coef,
                    "pssm_bias": pssm_bias,
                    "pssm_log_odds": pssm_log_odds,
                }
            }
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            pssm_dict=pssm_dict,
        )

        pssm_coef_result = result[15]
        pssm_bias_result = result[16]
        pssm_log_odds_result = result[17]
        
        # Check that PSSM data is present
        assert pssm_coef_result.sum() > 0, "PSSM coefficients should be non-zero"
        assert pssm_log_odds_result.sum() > 0, "PSSM log odds should be initialized"


class TestBiasByRes:
    """Test bias by residue functionality."""

    def test_bias_by_res_single_chain(self, structure, device):
        """Test bias by residue with single chain."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (["H"], ["L"])
        }
        
        # Get chain length
        seq_len_H = len(structure_batch[0]["seq_chain_H"])
        
        # Create bias array (bias toward certain amino acids at each position)
        bias_array = np.random.randn(seq_len_H, 21) * 0.1
        
        bias_by_res_dict = {
            protein_name: {
                "H": bias_array
            }
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            bias_by_res_dict=bias_by_res_dict,
        )

        bias_by_res_result = result[18]
        
        # Check that bias data is present
        assert bias_by_res_result.abs().sum() > 0, "Bias by residue should be non-zero"


class TestCombinedParameters:
    """Test combinations of multiple parameters."""

    def test_fixed_and_tied_positions(self, structure, device):
        """Test using both fixed and tied positions together."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (["H"], ["L"])
        }
        
        fixed_position_dict = {
            protein_name: {
                "H": list(range(1, 6))
            }
        }
        
        tied_positions_dict = {
            protein_name: [
                {"H": [10, 20]}
            ]
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            fixed_position_dict=fixed_position_dict,
            tied_positions_dict=tied_positions_dict,
        )

        chain_M_pos = result[10]
        tied_pos_list_of_lists_list = result[14]
        
        assert chain_M_pos.sum() < chain_M_pos.numel(), "Some positions should be fixed"
        assert len(tied_pos_list_of_lists_list) > 0, "Tied positions should be recorded"

    def test_all_parameters_combined(self, structure, device):
        """Test using all optional parameters together."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (["H"], ["L"])
        }
        
        # Fixed positions
        fixed_position_dict = {
            protein_name: {
                "H": list(range(1, 6))
            }
        }
        
        # Omit AAs
        omit_AA_dict = {
            protein_name: {
                "H": [(np.array([10, 11, 12]), ["C", "P"])]
            }
        }
        
        # Tied positions
        tied_positions_dict = {
            protein_name: [
                {"H": [15, 25]}
            ]
        }
        
        # PSSM
        seq_len_H = len(structure_batch[0]["seq_chain_H"])
        pssm_dict = {
            protein_name: {
                "H": {
                    "pssm_coef": np.ones(seq_len_H) * 0.5,
                    "pssm_bias": np.zeros((seq_len_H, 21)),
                    "pssm_log_odds": np.zeros((seq_len_H, 21)),
                }
            }
        }
        
        # Bias by res
        bias_by_res_dict = {
            protein_name: {
                "H": np.random.randn(seq_len_H, 21) * 0.1
            }
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            fixed_position_dict=fixed_position_dict,
            omit_AA_dict=omit_AA_dict,
            tied_positions_dict=tied_positions_dict,
            pssm_dict=pssm_dict,
            bias_by_res_dict=bias_by_res_dict,
        )

        # Verify all results are present and have expected properties
        assert len(result) == 20, "Should return 20 elements"
        
        X, S, mask, lengths, chain_M = result[:5]
        chain_M_pos, omit_AA_mask = result[10], result[11]
        tied_pos_list_of_lists_list = result[14]
        pssm_coef, pssm_bias = result[15], result[16]
        bias_by_res_result = result[18]
        
        # Basic checks that all parameters were applied
        assert chain_M_pos.sum() < chain_M_pos.numel(), "Fixed positions applied"
        assert omit_AA_mask.sum() > 0, "Omit AA mask applied"
        assert len(tied_pos_list_of_lists_list) > 0, "Tied positions applied"
        assert pssm_coef.sum() > 0, "PSSM applied"
        assert bias_by_res_result.abs().sum() > 0, "Bias by res applied"


class TestBatchProcessing:
    """Test processing multiple structures in a batch."""

    def test_multi_structure_batch(self, structure, device):
        """Test featurizing multiple structures in one batch."""
        # Filter structure to only H and L chains
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        
        # Create batch with 3 copies of the same structure
        batch = [structure_filtered, structure_filtered, structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (designed_chains, fixed_chains)
        }

        result = tied_featurize(
            batch,
            device,
            chain_dict,
        )

        X, S, mask, lengths, chain_M = result[:5]
        
        # Check batch dimension
        assert X.shape[0] == 3, "Batch size should be 3"
        assert S.shape[0] == 3, "Batch size should be 3"
        assert len(lengths) == 3, "Should have 3 sequence lengths"


class TestDeterminism:
    """Test that function is deterministic with same inputs."""

    def test_deterministic_output(self, structure, device):
        """Test that multiple calls with same input produce identical output."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        chain_dict = {
            structure_filtered["name"]: (["H"], ["L"])
        }

        # Call twice with same parameters
        result1 = tied_featurize(structure_batch, device, chain_dict)
        result2 = tied_featurize(structure_batch, device, chain_dict)

        # Compare all tensor outputs
        for i, (r1, r2) in enumerate(zip(result1, result2)):
            if isinstance(r1, torch.Tensor):
                assert torch.allclose(r1, r2, atol=1e-7), f"Tensor at index {i} differs"
            elif isinstance(r1, np.ndarray):
                assert np.allclose(r1, r2, atol=1e-7), f"Array at index {i} differs"
            elif isinstance(r1, list):
                # For nested lists, do recursive comparison
                assert r1 == r2, f"List at index {i} differs"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_visible_chains(self, structure, device):
        """Test with no visible (context) chains."""
        designed_chains = ["H", "L"]
        fixed_chains = []
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        chain_dict = {
            structure_filtered["name"]: (["H", "L"], [])  # Design all, no context
        }

        result = tied_featurize(structure_batch, device, chain_dict)
        
        # Should still work
        assert result is not None
        assert len(result) == 20

    def test_single_residue_fixed(self, structure, device):
        """Test fixing just a single residue."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        protein_name = structure_filtered["name"]
        
        chain_dict = {
            protein_name: (["H"], ["L"])
        }
        
        fixed_position_dict = {
            protein_name: {
                "H": [1]  # Fix only first position
            }
        }

        result = tied_featurize(
            structure_batch,
            device,
            chain_dict,
            fixed_position_dict=fixed_position_dict,
        )
        
        chain_M_pos = result[10]
        assert chain_M_pos.sum() > 0, "Should have some non-fixed positions"


class TestOutputShapes:
    """Test that all output tensors have correct shapes and types."""

    def test_output_tensor_shapes(self, structure, device):
        """Verify shapes of all output tensors."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        chain_dict = {
            structure_filtered["name"]: (["H"], ["L"])
        }

        result = tied_featurize(structure_batch, device, chain_dict)

        B = len(structure_batch)
        L_max = max([len(s["seq"]) for s in structure_batch])

        # Unpack and verify shapes
        X, S, mask, lengths, chain_M, chain_encoding_all = result[:6]
        chain_list_list, visible_list_list, masked_list_list = result[6:9]
        masked_chain_length_list_list, chain_M_pos, omit_AA_mask = result[9:12]
        residue_idx, dihedral_mask, tied_pos_list_of_lists_list = result[12:15]
        pssm_coef, pssm_bias, pssm_log_odds_all = result[15:18]
        bias_by_res_all, tied_beta = result[18:20]

        # Tensor shapes
        assert X.shape == (B, L_max, 4, 3), f"X shape mismatch: {X.shape}"
        assert S.shape == (B, L_max), f"S shape mismatch: {S.shape}"
        assert mask.shape == (B, L_max), f"mask shape mismatch: {mask.shape}"
        assert chain_M.shape == (B, L_max), f"chain_M shape mismatch: {chain_M.shape}"
        assert chain_encoding_all.shape == (B, L_max), f"chain_encoding_all shape mismatch"
        assert chain_M_pos.shape == (B, L_max), f"chain_M_pos shape mismatch"
        assert omit_AA_mask.shape == (B, L_max, 21), f"omit_AA_mask shape mismatch"
        assert residue_idx.shape == (B, L_max), f"residue_idx shape mismatch"
        assert dihedral_mask.shape == (B, L_max, 3), f"dihedral_mask shape mismatch: {dihedral_mask.shape}"
        assert pssm_coef.shape == (B, L_max), f"pssm_coef shape mismatch"
        assert pssm_bias.shape == (B, L_max, 21), f"pssm_bias shape mismatch"
        assert pssm_log_odds_all.shape == (B, L_max, 21), f"pssm_log_odds_all shape mismatch"
        assert bias_by_res_all.shape == (B, L_max, 21), f"bias_by_res_all shape mismatch"
        assert tied_beta.shape == (L_max,), f"tied_beta shape mismatch: {tied_beta.shape}"

        # Array shapes
        assert lengths.shape == (B,), f"lengths shape mismatch: {lengths.shape}"

        # List lengths
        assert len(chain_list_list) == B, "chain_list_list length mismatch"
        assert len(visible_list_list) == B, "visible_list_list length mismatch"
        assert len(masked_list_list) == B, "masked_list_list length mismatch"
        assert len(masked_chain_length_list_list) == B, "masked_chain_length_list_list length mismatch"
        assert len(tied_pos_list_of_lists_list) == B, "tied_pos_list_of_lists_list length mismatch"

    def test_output_types(self, structure, device):
        """Verify types of all outputs."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        chain_dict = {
            structure_filtered["name"]: (["H"], ["L"])
        }

        result = tied_featurize(structure_batch, device, chain_dict)

        # Check tensor types
        for i in [0, 1, 2, 4, 5, 10, 11, 12, 13, 15, 16, 17, 18, 19]:
            assert isinstance(result[i], torch.Tensor), f"Output {i} should be torch.Tensor"

        # Check array type
        assert isinstance(result[3], np.ndarray), "lengths should be numpy array"

        # Check list types
        for i in [6, 7, 8, 9, 14]:
            assert isinstance(result[i], list), f"Output {i} should be list"


class TestCoordinateIntegrity:
    """Test that coordinate data is properly extracted and formatted."""

    def test_coordinate_values_valid(self, structure, device):
        """Test that coordinates are valid (not NaN, not all zeros)."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        chain_dict = {
            structure_filtered["name"]: (["H"], ["L"])
        }

        result = tied_featurize(structure_batch, device, chain_dict)
        X = result[0]

        # Check for NaN
        assert not torch.isnan(X).any(), "Coordinates contain NaN"

        # Check that coordinates are not all zeros (should have actual structure)
        assert X.abs().sum() > 0, "Coordinates are all zeros"

        # Check that coordinates are in reasonable range (typical protein ~100 Å)
        assert X.abs().max() < 1000, "Coordinates seem unreasonably large"


class TestSequenceEncoding:
    """Test that sequence encoding is correct."""

    def test_sequence_encoding_valid(self, structure, device):
        """Test that sequence is properly encoded as integers."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        chain_dict = {
            structure_filtered["name"]: (["H"], ["L"])
        }

        result = tied_featurize(structure_batch, device, chain_dict)
        S = result[1]

        # Check that all values are valid amino acid indices (0-20)
        assert S.min() >= 0, "Sequence indices should be >= 0"
        assert S.max() <= 20, "Sequence indices should be <= 20"

    def test_sequence_matches_structure(self, structure, device):
        """Test that encoded sequence length matches structure."""
        designed_chains = ["H"]
        fixed_chains = ["L"]
        structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
        structure_batch = [structure_filtered]
        
        chain_dict = {
            structure_filtered["name"]: (["H"], ["L"])
        }

        result = tied_featurize(structure_batch, device, chain_dict)
        S, lengths = result[1], result[3]

        # Length should match the sequence in the structure
        expected_length = len(structure_batch[0]["seq"])
        assert lengths[0] == expected_length, f"Length mismatch: {lengths[0]} vs {expected_length}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
