"""Generate regression test data for tied_featurize_orig function."""

from pathlib import Path
import json
import numpy as np
import torch

from dauparas_proteinmpnn.io import parse_pdb, select_chains
from dauparas_proteinmpnn.featurize import tied_featurize_orig as tied_featurize

# Path to test PDB
PDB_PATH = Path("/home/bartosz.janusz/antifold-interface/data/input_files/5GGS_standardized.pdb")
REGRESSION_DATA_DIR = Path(__file__).parent.parent / "tests" / "regression_data"


def prepare_structure_for_chains(structure: dict, designed_chains: list[str], fixed_chains: list[str]) -> dict:
    """Helper to select only the chains we're working with."""
    all_chains = designed_chains + fixed_chains
    return select_chains(structure, all_chains)


def save_test_case(name: str, inputs: dict, outputs: tuple):
    """Save a test case to disk."""
    test_dir = REGRESSION_DATA_DIR / name
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save inputs (metadata)
    inputs_serializable = {}
    for key, value in inputs.items():
        if isinstance(value, torch.device):
            inputs_serializable[key] = str(value)
        elif key == "structure":
            inputs_serializable[key] = value["name"]  # Just save the name
        elif isinstance(value, np.ndarray):
            inputs_serializable[key] = value.tolist()
        else:
            inputs_serializable[key] = value
    
    with open(test_dir / "inputs.json", "w") as f:
        json.dump(inputs_serializable, f, indent=2)
    
    # Save outputs
    output_names = [
        "X", "S", "mask", "lengths", "chain_M", "chain_encoding_all", "chain_list_list",
        "visible_list_list", "masked_list_list", "masked_chain_length_list_list",
        "chain_M_pos", "omit_AA_mask", "residue_idx", "dihedral_mask",
        "tied_pos_list_of_lists_list", "pssm_coef", "pssm_bias", "pssm_log_odds_all",
        "bias_by_res_all", "tied_beta"
    ]
    
    for i, (name, value) in enumerate(zip(output_names, outputs)):
        if isinstance(value, torch.Tensor):
            torch.save(value, test_dir / f"{i:02d}_{name}.pt")
        elif isinstance(value, np.ndarray):
            np.save(test_dir / f"{i:02d}_{name}.npy", value)
        elif isinstance(value, list):
            with open(test_dir / f"{i:02d}_{name}.json", "w") as f:
                json.dump(value, f, indent=2)
        else:
            # Handle other types
            with open(test_dir / f"{i:02d}_{name}.json", "w") as f:
                json.dump(value, f, indent=2)
    
    print(f"✓ Saved test case: {name}")


def generate_all_test_cases():
    """Generate all regression test cases."""
    
    # Load structure
    with open(PDB_PATH, "r") as f:
        structure = parse_pdb(f)
    
    device = torch.device("cpu")
    
    print("Generating regression test data...")
    print("=" * 60)
    
    # Test 1: Basic single chain design
    designed_chains = ["H"]
    fixed_chains = ["L"]
    structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
    chain_dict = {structure_filtered["name"]: (designed_chains, fixed_chains)}
    
    result = tied_featurize(
        [structure_filtered],
        device,
        chain_dict,
    )
    
    save_test_case(
        "basic_single_chain_design",
        {
            "designed_chains": designed_chains,
            "fixed_chains": fixed_chains,
            "structure": structure_filtered,
            "device": device,
            "chain_dict": chain_dict,
        },
        result
    )
    
    # Test 2: Basic multi chain design
    designed_chains = ["H", "L"]
    fixed_chains = []
    structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
    chain_dict = {structure_filtered["name"]: (designed_chains, fixed_chains)}
    
    result = tied_featurize(
        [structure_filtered],
        device,
        chain_dict,
    )
    
    save_test_case(
        "basic_multi_chain_design",
        {
            "designed_chains": designed_chains,
            "fixed_chains": fixed_chains,
            "structure": structure_filtered,
            "device": device,
            "chain_dict": chain_dict,
        },
        result
    )
    
    # Test 3: CA only mode
    designed_chains = ["H"]
    fixed_chains = ["L"]
    structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
    chain_dict = {structure_filtered["name"]: (designed_chains, fixed_chains)}
    
    result = tied_featurize(
        [structure_filtered],
        device,
        chain_dict,
        ca_only=True,
    )
    
    save_test_case(
        "ca_only_mode",
        {
            "designed_chains": designed_chains,
            "fixed_chains": fixed_chains,
            "structure": structure_filtered,
            "device": device,
            "chain_dict": chain_dict,
            "ca_only": True,
        },
        result
    )
    
    # Test 4: Fixed positions single chain
    designed_chains = ["H"]
    fixed_chains = ["L"]
    structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
    protein_name = structure_filtered["name"]
    chain_dict = {protein_name: (designed_chains, fixed_chains)}
    fixed_position_dict = {protein_name: {"H": list(range(1, 11))}}
    
    result = tied_featurize(
        [structure_filtered],
        device,
        chain_dict,
        fixed_position_dict=fixed_position_dict,
    )
    
    save_test_case(
        "fixed_positions_single_chain",
        {
            "designed_chains": designed_chains,
            "fixed_chains": fixed_chains,
            "structure": structure_filtered,
            "device": device,
            "chain_dict": chain_dict,
            "fixed_position_dict": fixed_position_dict,
        },
        result
    )
    
    # Test 5: Omit AA single chain
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
    
    result = tied_featurize(
        [structure_filtered],
        device,
        chain_dict,
        omit_AA_dict=omit_AA_dict,
    )
    
    save_test_case(
        "omit_aa_single_chain",
        {
            "designed_chains": designed_chains,
            "fixed_chains": fixed_chains,
            "structure": structure_filtered,
            "device": device,
            "chain_dict": chain_dict,
            "omit_AA_dict_description": "Omit C and M at positions 5, 10, 15 on chain H",
        },
        result
    )
    
    # Test 6: Tied positions within chain
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
    
    result = tied_featurize(
        [structure_filtered],
        device,
        chain_dict,
        tied_positions_dict=tied_positions_dict,
    )
    
    save_test_case(
        "tied_positions_within_chain",
        {
            "designed_chains": designed_chains,
            "fixed_chains": fixed_chains,
            "structure": structure_filtered,
            "device": device,
            "chain_dict": chain_dict,
            "tied_positions_dict": tied_positions_dict,
        },
        result
    )
    
    # Test 7: PSSM single chain
    designed_chains = ["H"]
    fixed_chains = ["L"]
    structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
    protein_name = structure_filtered["name"]
    chain_dict = {protein_name: (designed_chains, fixed_chains)}
    seq_len_H = len(structure_filtered["seq_chain_H"])
    # Use fixed seed for reproducibility
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
    
    result = tied_featurize(
        [structure_filtered],
        device,
        chain_dict,
        pssm_dict=pssm_dict,
    )
    
    save_test_case(
        "pssm_single_chain",
        {
            "designed_chains": designed_chains,
            "fixed_chains": fixed_chains,
            "structure": structure_filtered,
            "device": device,
            "chain_dict": chain_dict,
            "pssm_dict_description": "PSSM with coef=0.5, random bias and log_odds",
        },
        result
    )
    
    # Test 8: Batch processing (3 structures)
    designed_chains = ["H"]
    fixed_chains = ["L"]
    structure_filtered = prepare_structure_for_chains(structure, designed_chains, fixed_chains)
    protein_name = structure_filtered["name"]
    chain_dict = {protein_name: (designed_chains, fixed_chains)}
    batch = [structure_filtered, structure_filtered, structure_filtered]
    
    result = tied_featurize(
        batch,
        device,
        chain_dict,
    )
    
    save_test_case(
        "batch_processing_3_structures",
        {
            "designed_chains": designed_chains,
            "fixed_chains": fixed_chains,
            "structure": structure_filtered,
            "device": device,
            "chain_dict": chain_dict,
            "batch_size": 3,
        },
        result
    )
    
    print("=" * 60)
    print(f"✓ All regression test data generated in: {REGRESSION_DATA_DIR}")
    print(f"  Total test cases: 8")


if __name__ == "__main__":
    generate_all_test_cases()
