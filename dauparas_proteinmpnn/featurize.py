from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
import itertools

import numpy as np
import torch

from dauparas_proteinmpnn.io import Structure, parse_pdb, select_chains

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
ALPHABET_DICT = dict(zip(ALPHABET, range(21)))


@dataclass
class Protein:
    """Unified structure containing single protein data and all configuration parameters.

    Attributes:
        structure: Dictionary containing protein structure data with keys:
            - 'name': Unique identifier for the protein
            - 'seq': Full sequence string (concatenated chains)
            - 'seq_chain_{X}': Sequence for chain X (e.g., 'seq_chain_A')
            - 'coords_chain_{X}': Dictionary of coordinates for chain X
        masked_chains: List of chain IDs to design/predict (e.g., ['A', 'B'])
        visible_chains: List of chain IDs to keep as context (e.g., ['C'])
        fixed_positions: Optional dict mapping chain IDs to lists of 1-based position indices
            Example: {'A': [1, 2, 3, 10], 'B': [5, 6]}
        omit_aa: Optional dict mapping chain IDs to lists of (positions, amino_acids) tuples
            Example: {'A': [(np.array([1, 2, 3]), ['C', 'P']), ...]}
        tied_positions: Optional list of dicts mapping chain IDs to position lists.
            All positions in each dict will be constrained to the same amino acid.
            Example: [{'A': [5, 50]}, {'A': [10], 'B': [15]}]
        pssm: Optional dict mapping chain IDs to PSSM parameter dicts with keys:
            - 'pssm_coef': Coefficient array for PSSM strength
            - 'pssm_bias': Bias values array, shape [seq_length, 21]
            - 'pssm_log_odds': Log odds array, shape [seq_length, 21]
            Example: {'A': {'pssm_coef': array, 'pssm_bias': array, 'pssm_log_odds': array}}
        bias_by_res: Optional dict mapping chain IDs to bias arrays of shape [seq_length, 21]
            Example: {'A': bias_array_A, 'B': bias_array_B}
    """

    structure: Structure
    masked_chains: list[str]
    visible_chains: list[str]
    fixed_positions: dict[str, list[int]] | None = None
    omit_aa: dict[str, list[tuple[np.ndarray, list[str]]]] | None = None
    tied_positions: list[dict[str, list[int] | list[list[int]]]] | None = None
    pssm: dict[str, dict[str, np.ndarray] | None] | None = None
    bias_by_res: dict[str, np.ndarray] | None = None

    @property
    def protein_name(self) -> str:
        """Get the protein name from the structure."""
        return self.structure["name"]


def _convert_batch_to_orig_format(
    batch: list[Protein],
) -> tuple[
    list[dict],
    dict[str, tuple[list[str], list[str]]],
    dict[str, dict[str, list[int]]] | None,
    dict[str, dict[str, list[tuple[np.ndarray, list[str]]]]] | None,
    dict[str, list[dict[str, list[int] | list[list[int]]]]] | None,
    dict[str, dict[str, dict[str, np.ndarray] | None]] | None,
    dict[str, dict[str, np.ndarray]] | None,
]:
    """Convert list[Protein] to original tied_featurize format.

    Returns:
        Tuple of (structures, chain_dict, fixed_position_dict, omit_AA_dict,
                  tied_positions_dict, pssm_dict, bias_by_res_dict)
    """
    structures = [item.structure for item in batch]

    chain_dict = {
        item.protein_name: (item.masked_chains, item.visible_chains) for item in batch
    }

    fixed_position_dict = None
    if any(item.fixed_positions is not None for item in batch):
        fixed_position_dict = {
            item.protein_name: item.fixed_positions
            for item in batch
            if item.fixed_positions is not None
        }

    omit_AA_dict = None
    if any(item.omit_aa is not None for item in batch):
        omit_AA_dict = {
            item.protein_name: item.omit_aa
            for item in batch
            if item.omit_aa is not None
        }

    tied_positions_dict = None
    if any(item.tied_positions is not None for item in batch):
        tied_positions_dict = {
            item.protein_name: item.tied_positions
            for item in batch
            if item.tied_positions is not None
        }

    pssm_dict = None
    if any(item.pssm is not None for item in batch):
        pssm_dict = {
            item.protein_name: item.pssm for item in batch if item.pssm is not None
        }

    bias_by_res_dict = None
    if any(item.bias_by_res is not None for item in batch):
        bias_by_res_dict = {
            item.protein_name: item.bias_by_res
            for item in batch
            if item.bias_by_res is not None
        }

    return (
        structures,
        chain_dict,
        fixed_position_dict,
        omit_AA_dict,
        tied_positions_dict,
        pssm_dict,
        bias_by_res_dict,
    )


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


def tied_featurize_orig(
    batch: list[dict],
    device: torch.device,
    chain_dict: dict[str, tuple[list[str], list[str]]] | None,
    fixed_position_dict: dict[str, dict[str, list[int]]] | None = None,
    omit_AA_dict: dict[str, dict[str, list[tuple[np.ndarray, list[str]]]]]
    | None = None,
    tied_positions_dict: dict[str, list[dict[str, list[int] | list[list[int]]]]]
    | None = None,
    pssm_dict: dict[str, dict[str, dict[str, np.ndarray] | None]] | None = None,
    bias_by_res_dict: dict[str, dict[str, np.ndarray]] | None = None,
    ca_only: bool = False,
):
    """Pack and pad batch into torch tensors."""

    # ============================================================================
    # SECTION 1: Initialize batch dimensions and allocate output tensors
    # ============================================================================
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    B = len(batch)
    lengths = np.array(
        [len(b["seq"]) for b in batch], dtype=np.int32
    )  # sum of chain seq lengths
    L_max = max([len(b["seq"]) for b in batch])
    if ca_only:
        X = np.zeros([B, L_max, 1, 3])
    else:
        X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted
    pssm_coef_all = np.zeros(
        [B, L_max], dtype=np.float32
    )  # 1.0 for the bits that need to be predicted
    pssm_bias_all = np.zeros(
        [B, L_max, 21], dtype=np.float32
    )  # 1.0 for the bits that need to be predicted
    pssm_log_odds_all = 10000.0 * np.ones(
        [B, L_max, 21], dtype=np.float32
    )  # 1.0 for the bits that need to be predicted
    chain_M_pos = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted
    S = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)

    # ============================================================================
    # SECTION 2: Determine which chains to mask (design) vs keep visible (context)
    # ============================================================================
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []
    for i, b in enumerate(batch):
        if chain_dict != None:
            masked_chains, visible_chains = chain_dict[
                b["name"]
            ]  # masked_chains a list of chain letters to predict [A, D, F]
        else:
            masked_chains = [item[-1:] for item in list(b) if item[:10] == "seq_chain_"]
            visible_chains = []
        masked_chains.sort()  # sort masked_chains
        visible_chains.sort()  # sort visible_chains
        all_chains = masked_chains + visible_chains

    # ============================================================================
    # SECTION 3: Process each sample in batch - extract chains and build features
    # ============================================================================
    for i, b in enumerate(batch):
        mask_dict = {}
        a = 0
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0

        # ------------------------------------------------------------------------
        # SECTION 3A: Process visible chains (context - not designed)
        # ------------------------------------------------------------------------
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f"seq_chain_{letter}"]
                chain_seq = "".join([a if a != "-" else "X" for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                if ca_only:
                    x_chain = np.array(
                        chain_coords[f"CA_chain_{letter}"]
                    )  # [chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack(
                        [
                            chain_coords[c]
                            for c in [
                                f"N_chain_{letter}",
                                f"CA_chain_{letter}",
                                f"C_chain_{letter}",
                                f"O_chain_{letter}",
                            ]
                        ],
                        1,
                    )  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))

            # ------------------------------------------------------------------------
            # SECTION 3B: Process masked chains (to be designed/predicted)
            # ------------------------------------------------------------------------
            if letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                chain_seq = b[f"seq_chain_{letter}"]
                chain_seq = "".join([a if a != "-" else "X" for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                masked_chain_length_list.append(chain_length)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 1.0 for masked
                if ca_only:
                    x_chain = np.array(
                        chain_coords[f"CA_chain_{letter}"]
                    )  # [chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack(
                        [
                            chain_coords[c]
                            for c in [
                                f"N_chain_{letter}",
                                f"CA_chain_{letter}",
                                f"C_chain_{letter}",
                                f"O_chain_{letter}",
                            ]
                        ],
                        1,
                    )  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict != None:
                    fixed_pos_list = fixed_position_dict[b["name"]][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list) - 1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_AA_dict != None:
                    for item in omit_AA_dict[b["name"]][letter]:
                        idx_AA = np.array(item[0]) - 1
                        AA_idx = np.array(
                            [
                                np.argwhere(np.array(list(alphabet)) == AA)[0][0]
                                for AA in item[1]
                            ]
                        ).repeat(idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                        omit_AA_mask_temp[idx_[:, 0], idx_[:, 1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                if pssm_dict:
                    if pssm_dict[b["name"]][letter]:
                        pssm_coef = pssm_dict[b["name"]][letter]["pssm_coef"]
                        pssm_bias = pssm_dict[b["name"]][letter]["pssm_bias"]
                        pssm_log_odds = pssm_dict[b["name"]][letter]["pssm_log_odds"]
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[b["name"]][letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

        # ------------------------------------------------------------------------
        # SECTION 3C: Handle tied positions (positions that should have same AA)
        # ------------------------------------------------------------------------
        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict != None:
            tied_pos_list = tied_positions_dict[b["name"]]
            if tied_pos_list:
                set_chains_tied = set(
                    list(itertools.chain(*[list(item) for item in tied_pos_list]))
                )
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[
                            np.argwhere(letter_list_np == k)[0][0]
                        ]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(
                                    start_idx + v[0][v_count] - 1
                                )  # make 0 to be the first
                                tied_beta[start_idx + v[0][v_count] - 1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(
                                    start_idx + v_ - 1
                                )  # make 0 to be the first
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)

        # ------------------------------------------------------------------------
        # SECTION 3D: Concatenate all chain data into single arrays for this sample
        # ------------------------------------------------------------------------
        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(
            chain_mask_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)
        m_pos = np.concatenate(
            fixed_position_mask_list, 0
        )  # [L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(
            pssm_coef_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(
            pssm_bias_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(
            pssm_log_odds_list, 0
        )  # [L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(
            bias_by_res_list, 0
        )  # [L,21], 0.0 for places where AA frequencies don't need to be tweaked

        # ------------------------------------------------------------------------
        # SECTION 3E: Pad sequences to max length and store in batch arrays
        # ------------------------------------------------------------------------
        l = len(all_sequence)
        x_pad = np.pad(
            x, [[0, L_max - l], [0, 0], [0, 0]], "constant", constant_values=(np.nan,)
        )
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]], "constant", constant_values=(0.0,))
        m_pos_pad = np.pad(m_pos, [[0, L_max - l]], "constant", constant_values=(0.0,))
        omit_AA_mask_pad = np.pad(
            np.concatenate(omit_AA_mask_list, 0),
            [[0, L_max - l]],
            "constant",
            constant_values=(0.0,),
        )
        chain_M[i, :] = m_pad
        chain_M_pos[i, :] = m_pos_pad
        omit_AA_mask[i,] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(
            chain_encoding, [[0, L_max - l]], "constant", constant_values=(0.0,)
        )
        chain_encoding_all[i, :] = chain_encoding_pad

        pssm_coef_pad = np.pad(
            pssm_coef_, [[0, L_max - l]], "constant", constant_values=(0.0,)
        )
        pssm_bias_pad = np.pad(
            pssm_bias_, [[0, L_max - l], [0, 0]], "constant", constant_values=(0.0,)
        )
        pssm_log_odds_pad = np.pad(
            pssm_log_odds_, [[0, L_max - l], [0, 0]], "constant", constant_values=(0.0,)
        )

        pssm_coef_all[i, :] = pssm_coef_pad
        pssm_bias_all[i, :] = pssm_bias_pad
        pssm_log_odds_all[i, :] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(
            bias_by_res_, [[0, L_max - l], [0, 0]], "constant", constant_values=(0.0,)
        )
        bias_by_res_all[i, :] = bias_by_res_pad

        # ------------------------------------------------------------------------
        # SECTION 3F: Convert amino acid sequences to integer labels
        # ------------------------------------------------------------------------
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)

    # ============================================================================
    # SECTION 4: Handle missing coordinates and convert to PyTorch tensors
    # ============================================================================
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    # ------------------------------------------------------------------------
    # SECTION 4A: Convert all numpy arrays to PyTorch tensors on device
    # ------------------------------------------------------------------------
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(
        dtype=torch.float32, device=device
    )
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(
        dtype=torch.float32, device=device
    )
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(
        dtype=torch.float32, device=device
    )
    
    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device)

    # ------------------------------------------------------------------------
    # SECTION 4B: Create masks for dihedral angles (phi, psi, omega)
    # ------------------------------------------------------------------------
    jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(
        dtype=torch.float32, device=device
    )
    phi_mask = np.pad(jumps, [[0, 0], [1, 0]])
    psi_mask = np.pad(jumps, [[0, 0], [0, 1]])
    omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
    dihedral_mask = np.concatenate(
        [phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]], -1
    )  # [B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(
        dtype=torch.float32, device=device
    )
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(
        dtype=torch.long, device=device
    )
    if ca_only:
        X_out = X[:, :, 0]
    else:
        X_out = X
    return (
        X_out,
        S,
        mask,
        lengths,
        chain_M,
        chain_encoding_all,
        letter_list_list,
        visible_list_list,
        masked_list_list,
        masked_chain_length_list_list,
        chain_M_pos,
        omit_AA_mask,
        residue_idx,
        dihedral_mask,
        tied_pos_list_of_lists_list,
        pssm_coef_all,
        pssm_bias_all,
        pssm_log_odds_all,
        bias_by_res_all,
        tied_beta,
    )


def tied_featurize(
    batch: list[Protein],
    device: torch.device,
    ca_only: bool = False,
) -> TiedFeaturizeResult:
    """Pack and pad batch of protein structures into PyTorch tensors for ProteinMPNN.

    This function converts protein structure data from a batch of ProteinBatch objects into
    padded tensors suitable for input to the ProteinMPNN model. It handles chain
    encoding, fixed positions, tied positions, PSSM matrices, and various masking
    operations.

    Args:
        batch: List of ProteinBatch objects, each containing a protein structure and
            all associated configuration parameters (chain selection, fixed positions, etc.)
        device: PyTorch device for tensor allocation (e.g., torch.device('cuda:0') or torch.device('cpu'))
        ca_only: If True, use only CA (carbon alpha) atoms instead of full backbone (N, CA, C, O).

    Returns:
        TiedFeaturizeResult object containing all featurized data including coordinates,
        sequences, masks, chain encodings, and configuration parameters.

    Examples:
        Basic usage with chain selection:

        >>> import torch
        >>> from dauparas_proteinmpnn.io import parse_pdb
        >>> from dauparas_proteinmpnn.featurize import tied_featurize, ProteinBatch
        >>>
        >>> # Parse PDB file
        >>> structures = parse_pdb('protein.pdb')
        >>> protein = structures[0]
        >>>
        >>> # Create batch with chain configuration
        >>> batch = [
        ...     ProteinBatch(
        ...         structure=protein,
        ...         masked_chains=['A', 'B'],
        ...         visible_chains=['C']
        ...     )
        ... ]
        >>>
        >>> # Featurize
        >>> device = torch.device('cuda:0')
        >>> result = tied_featurize(batch=batch, device=device)
        >>> X, S, mask = result.X, result.S, result.mask

        With fixed positions (prevent certain residues from being redesigned):

        >>> batch = [
        ...     ProteinBatch(
        ...         structure=protein,
        ...         masked_chains=['A', 'B'],
        ...         visible_chains=['C'],
        ...         fixed_positions={'A': list(range(10, 21))}
        ...     )
        ... ]
        >>>
        >>> result = tied_featurize(batch=batch, device=device)

        With tied positions (force same amino acid at multiple positions):

        >>> batch = [
        ...     ProteinBatch(
        ...         structure=protein,
        ...         masked_chains=['A', 'B'],
        ...         visible_chains=['C'],
        ...         tied_positions=[{'A': [5, 50]}]
        ...     )
        ... ]
        >>>
        >>> result = tied_featurize(batch=batch, device=device)

        With amino acid omissions (exclude certain amino acids at positions):

        >>> import numpy as np
        >>>
        >>> batch = [
        ...     ProteinBatch(
        ...         structure=protein,
        ...         masked_chains=['A', 'B'],
        ...         visible_chains=['C'],
        ...         omit_aa={
        ...             'A': [(np.array([15, 16, 17, 18, 19, 20]), ['C', 'P'])]
        ...         }
        ...     )
        ... ]
        >>>
        >>> result = tied_featurize(batch=batch, device=device)

        CA-only mode (useful for CA-ProteinMPNN):

        >>> structures = parse_pdb('protein.pdb', ca_only=True)
        >>> protein = structures[0]
        >>>
        >>> batch = [
        ...     ProteinBatch(
        ...         structure=protein,
        ...         masked_chains=['A', 'B'],
        ...         visible_chains=['C']
        ...     )
        ... ]
        >>>
        >>> result = tied_featurize(batch=batch, device=device, ca_only=True)
        >>> # result.X shape is now [B, L_max, 3] instead of [B, L_max, 4, 3]

        Batch processing multiple proteins:

        >>> protein1 = parse_pdb('protein1.pdb')[0]
        >>> protein2 = parse_pdb('protein2.pdb')[0]
        >>>
        >>> batch = [
        ...     ProteinBatch(
        ...         structure=protein1,
        ...         masked_chains=['A'],
        ...         visible_chains=['B']
        ...     ),
        ...     ProteinBatch(
        ...         structure=protein2,
        ...         masked_chains=['C'],
        ...         visible_chains=[]
        ...     )
        ... ]
        >>>
        >>> result = tied_featurize(batch=batch, device=device)
    """
    # Convert unified batch format to original format
    (
        structures,
        chain_dict,
        fixed_position_dict,
        omit_AA_dict,
        tied_positions_dict,
        pssm_dict,
        bias_by_res_dict,
    ) = _convert_batch_to_orig_format(batch)

    result_tuple = tied_featurize_orig(
        structures,
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
