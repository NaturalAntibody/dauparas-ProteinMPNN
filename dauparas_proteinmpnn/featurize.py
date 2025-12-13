from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
import torch

from dauparas_proteinmpnn.io import Structure, parse_pdb, select_chains
from dauparas_proteinmpnn.protein_mpnn_utils import (
    tied_featurize as tied_featurize_orig,
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
ALPHABET_DICT = dict(zip(ALPHABET, range(21)))


@dataclass
class TiedFeaturizeResult:
    X: torch.Tensor
    S: torch.Tensor
    mask: torch.Tensor
    lengths: np.ndarray
    chain_M: torch.Tensor
    chain_encoding_all: torch.Tensor
    chain_list_list: list
    visible_list_list: list
    masked_list_list: list
    masked_chain_length_list_list: list
    chain_M_pos: torch.Tensor
    omit_AA_mask: torch.Tensor
    residue_idx: torch.Tensor
    dihedral_mask: torch.Tensor
    tied_pos_list_of_lists_list: list
    pssm_coef: torch.Tensor
    pssm_bias: torch.Tensor
    pssm_log_odds_all: torch.Tensor
    bias_by_res_all: torch.Tensor
    tied_beta: torch.Tensor

    def clone(self) -> "TiedFeaturizeResult":
        """Create a deep copy of the TiedFeaturizeResult object."""
        return TiedFeaturizeResult(
            X=self.X.clone(),
            S=self.S.clone(),
            mask=self.mask.clone(),
            lengths=self.lengths.copy(),
            chain_M=self.chain_M.clone(),
            chain_encoding_all=self.chain_encoding_all.clone(),
            chain_list_list=deepcopy(self.chain_list_list),
            visible_list_list=deepcopy(self.visible_list_list),
            masked_list_list=deepcopy(self.masked_list_list),
            masked_chain_length_list_list=deepcopy(self.masked_chain_length_list_list),
            chain_M_pos=self.chain_M_pos.clone(),
            omit_AA_mask=self.omit_AA_mask.clone(),
            residue_idx=self.residue_idx.clone(),
            dihedral_mask=self.dihedral_mask.clone(),
            tied_pos_list_of_lists_list=deepcopy(self.tied_pos_list_of_lists_list),
            pssm_coef=self.pssm_coef.clone(),
            pssm_bias=self.pssm_bias.clone(),
            pssm_log_odds_all=self.pssm_log_odds_all.clone(),
            bias_by_res_all=self.bias_by_res_all.clone(),
            tied_beta=self.tied_beta.clone(),
        )


def tied_featurize(
    batch,
    device,
    chain_dict,
    fixed_position_dict=None,
    omit_AA_dict=None,
    tied_positions_dict=None,
    pssm_dict=None,
    bias_by_res_dict=None,
    ca_only=False,
) -> TiedFeaturizeResult:
    result_tuple = tied_featurize_orig(
        batch,
        device,
        chain_dict,
        fixed_position_dict=fixed_position_dict,
        omit_AA_dict=omit_AA_dict,
        tied_positions_dict=tied_positions_dict,
        pssm_dict=pssm_dict,
        bias_by_res_dict=bias_by_res_dict,
        ca_only=ca_only,
    )
    return TiedFeaturizeResult(*result_tuple)


def featurize_pdb(
    pdb: Path | TextIO,
    designed_chains: list[str],
    fixed_chains: list[str],
    device,
    chain_designed_positions: dict | None = None,
) -> TiedFeaturizeResult:
    all_chains = designed_chains + fixed_chains
    structure = parse_pdb(pdb, chain_ids=all_chains)
    return featurize_structure(
        structure, designed_chains, fixed_chains, device, chain_designed_positions
    )


def featurize_structure(
    structure: Structure,
    designed_chains: list[str],
    fixed_chains: list[str],
    device,
    chain_designed_positions: dict | None = None,
) -> TiedFeaturizeResult:
    all_chains = designed_chains + fixed_chains
    structure = select_chains(structure, all_chains)
    chain_id_dict = {structure["name"]: (designed_chains, fixed_chains)}
    fixed_positions_dict = None
    if chain_designed_positions is not None:
        fixed_positions_dict = get_fixed_positions_dict(
            structure, chain_designed_positions
        )
    return tied_featurize(
        batch=[structure],
        device=device,
        chain_dict=chain_id_dict,
        fixed_position_dict=fixed_positions_dict,
    )


def encode_sequence(features: TiedFeaturizeResult, seq: str) -> TiedFeaturizeResult:
    input_seq_length = len(seq)
    S_input = torch.tensor([ALPHABET_DICT[AA] for AA in seq], device=features.S.device)[
        None, :
    ].repeat(features.X.shape[0], 1)
    # assumes that S and S_input are alphabetically sorted for masked_chains
    features.S[:, :input_seq_length] = S_input
    return features


def get_fixed_positions_dict(
    protein: dict, chain_designed_positions: dict[str, list[int]]
) -> dict:
    seq_chains = {
        key.replace("seq_chain_", ""): seq
        for key, seq in protein.items()
        if key.startswith("seq_chain_")
    }
    res = {}
    for chain_id, seq in seq_chains.items():
        all_positions = set(range(1, len(seq) + 1))
        if chain_id in chain_designed_positions:
            if chain_designed_positions[chain_id] is None:
                designed_positions = all_positions
            else:
                designed_positions = set(chain_designed_positions[chain_id])
            fixed_positions = list(all_positions - designed_positions)
            res[chain_id] = fixed_positions
        else:
            res[chain_id] = list(all_positions)
    return {protein["name"]: res}
