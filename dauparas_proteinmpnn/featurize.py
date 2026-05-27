from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from Bio.Align import chain
import numpy as np
import torch

from dauparas_proteinmpnn.io import Structure, parse_pdb, select_chains

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
ALPHABET_DICT = dict(zip(ALPHABET, range(21)))


@dataclass
class Protein:
    """Unified structure containing single protein data and all configuration parameters.

    Attributes:
        structure: Structure object containing protein data
        masked_chains: List of chain IDs to design/predict (e.g., ['A', 'B'])
        visible_chains: List of chain IDs to keep as context (e.g., ['C'])
        fixed_positions: Optional dict mapping chain IDs to lists of 1-based position indices
            Example: {'A': [1, 2, 3, 10], 'B': [5, 6]}
        chain_designed_positions: Optional dict mapping chain IDs to lists of designed positions.
            If provided (and fixed_positions is None), fixed_positions will be computed automatically.
            Example: {'A': [1, 2, 3], 'B': None}  # None means design all positions
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
    chain_designed_positions: dict[str, list[int]] | None = None
    omit_aa: dict[str, list[tuple[np.ndarray, list[str]]]] | None = None
    tied_positions: list[dict[str, list[int] | list[list[int]]]] | None = None
    pssm: dict[str, dict[str, np.ndarray] | None] | None = None
    bias_by_res: dict[str, np.ndarray] | None = None

    def __post_init__(self):
        """Compute fixed_positions from chain_designed_positions if needed."""
        if self.chain_designed_positions is not None and self.fixed_positions is None:
            self.fixed_positions = self.compute_fixed_positions()

    @property
    def protein_name(self) -> str:
        """Get the protein name from the structure."""
        return self.structure.name

    def compute_fixed_positions(self) -> dict[str, list[int]]:
        """Convert designed positions to fixed positions dictionary.

        Args:
            structure: Structure object
            chain_designed_positions: Dict mapping chain IDs to lists of designed positions or None

        Returns:
            Dictionary mapping chain IDs to fixed positions
        """
        assert self.chain_designed_positions is not None
        res = {}
        for chain_id, chain_data in self.structure.chains.items():
            seq_length = len(chain_data.seq)
            all_positions = set(range(1, seq_length + 1))
            if chain_id in self.chain_designed_positions:
                if chain_id not in self.chain_designed_positions or not self.chain_designed_positions[chain_id]:
                    designed_positions = all_positions
                else:
                    designed_positions = set(self.chain_designed_positions[chain_id])
                fixed_positions = list(all_positions - designed_positions)
                res[chain_id] = fixed_positions
            else:
                res[chain_id] = list(all_positions)
        return res


@dataclass
class ItemFeatures:
    """Features extracted from a single protein structure.

    This represents the processed data from one protein before batching/padding.
    All arrays are numpy arrays with actual sequence length (no padding).
    """

    x: np.ndarray  # [L, 4, 3] or [L, 1, 3] for ca_only - backbone coordinates
    s: np.ndarray  # [L] - amino acid sequence indices
    chain_m: np.ndarray  # [L] - mask for positions to design (1.0 for masked chains)
    chain_m_pos: np.ndarray  # [L] - mask for non-fixed positions (1.0 for designable)
    chain_encoding: np.ndarray  # [L] - chain identity encoding
    residue_idx: np.ndarray  # [L] - residue indices with chain offset
    omit_aa_mask: np.ndarray  # [L, 21] - mask for amino acids to exclude
    pssm_coef: np.ndarray  # [L] - PSSM coefficient
    pssm_bias: np.ndarray  # [L, 21] - PSSM bias
    pssm_log_odds: np.ndarray  # [L, 21] - PSSM log odds
    bias_by_res: np.ndarray  # [L, 21] - per-residue amino acid bias
    tied_beta: np.ndarray  # [L] - tied position beta values
    letter_list: list[str]  # Chain letters in order
    visible_list: list[str]  # Visible (context) chain letters
    masked_list: list[str]  # Masked (designed) chain letters
    masked_chain_length_list: list[int]  # Lengths of masked chains
    tied_pos_list_of_lists: list[list[int]]  # Groups of tied position indices


@dataclass
class BatchFeatures:
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

    def clone(self) -> "BatchFeatures":
        """Create a deep copy of the BatchFeatures object."""
        return BatchFeatures(
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


def _process_item(
    protein: Protein,
    ca_only: bool = False,
) -> ItemFeatures:
    """Process a single Protein object into features.

    This function extracts and processes features from one protein,
    handling chain selection, fixed positions, tied positions, and various biases.

    Args:
        protein: Protein object containing structure and all configuration parameters
        ca_only: If True, use only CA atoms instead of full backbone

    Returns:
        ItemFeatures object containing all processed features
    """

    structure = protein.structure
    masked_chains = (
        protein.masked_chains
        if protein.masked_chains
        else list(structure.chains.keys())
    )
    visible_chains = protein.visible_chains if protein.visible_chains else []
    fixed_positions = protein.fixed_positions
    omit_aa = protein.omit_aa
    tied_positions = protein.tied_positions
    pssm = protein.pssm
    bias_by_res = protein.bias_by_res

    masked_chains = sorted(masked_chains)
    visible_chains = sorted(visible_chains)
    all_chains = masked_chains + visible_chains

    # Initialize accumulators
    x_chain_list = []
    chain_mask_list = []
    chain_seq_list = []
    chain_encoding_list = []
    fixed_position_mask_list = []
    omit_aa_mask_list = []
    pssm_coef_list = []
    pssm_bias_list = []
    pssm_log_odds_list = []
    bias_by_res_list = []
    letter_list = []
    visible_list = []
    masked_list = []
    masked_chain_length_list = []
    global_idx_start_list = [0]

    c = 1  # chain encoding counter

    # Process each chain
    for chain_id in all_chains:
        is_visible = chain_id in visible_chains
        is_masked = chain_id in masked_chains

        letter_list.append(chain_id)
        if is_visible:
            visible_list.append(chain_id)
        if is_masked:
            masked_list.append(chain_id)

        # Extract chain sequence and coordinates
        chain_data = structure.chains[chain_id]
        chain_seq = chain_data.seq
        chain_seq = "".join([a if a != "-" else "X" for a in chain_seq])
        chain_length = len(chain_seq)
        global_idx_start_list.append(global_idx_start_list[-1] + chain_length)

        if is_masked:
            masked_chain_length_list.append(chain_length)

        # Extract coordinates
        if ca_only:
            x_chain = np.array(chain_data.coords["CA"])
            if len(x_chain.shape) == 2:
                x_chain = x_chain[:, None, :]
        else:
            x_chain = np.stack(
                [chain_data.coords[atom] for atom in ["N", "CA", "C", "O"]],
                1,
            )

        x_chain_list.append(x_chain)
        chain_seq_list.append(chain_seq)
        chain_encoding_list.append(c * np.ones(chain_length))
        c += 1

        # Set up masks
        if is_visible:
            chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
            fixed_position_mask = np.ones(chain_length)  # all fixed
        else:  # is_masked
            chain_mask = np.ones(chain_length)  # 1.0 for masked chains
            fixed_position_mask = np.ones(chain_length)
            if fixed_positions is not None and chain_id in fixed_positions:
                fixed_pos_list = fixed_positions[chain_id]
                if fixed_pos_list:
                    fixed_position_mask[np.array(fixed_pos_list) - 1] = 0.0

        chain_mask_list.append(chain_mask)
        fixed_position_mask_list.append(fixed_position_mask)

        # Handle omit_aa
        omit_aa_mask_temp = np.zeros([chain_length, len(ALPHABET)], np.int32)
        if omit_aa is not None and chain_id in omit_aa:
            for item in omit_aa[chain_id]:
                idx_aa = np.array(item[0]) - 1
                aa_idx = np.array(
                    [
                        np.argwhere(np.array(list(ALPHABET)) == AA)[0][0]
                        for AA in item[1]
                    ]
                ).repeat(idx_aa.shape[0])
                idx_ = np.array([[a, b] for a in idx_aa for b in aa_idx])
                omit_aa_mask_temp[idx_[:, 0], idx_[:, 1]] = 1
        omit_aa_mask_list.append(omit_aa_mask_temp)

        # Handle PSSM
        pssm_coef_chain = np.zeros(chain_length)
        pssm_bias_chain = np.zeros([chain_length, 21])
        pssm_log_odds_chain = 10000.0 * np.ones([chain_length, 21])
        if pssm is not None and chain_id in pssm and pssm[chain_id] is not None:
            pssm_coef_chain = pssm[chain_id]["pssm_coef"]
            pssm_bias_chain = pssm[chain_id]["pssm_bias"]
            pssm_log_odds_chain = pssm[chain_id]["pssm_log_odds"]
        pssm_coef_list.append(pssm_coef_chain)
        pssm_bias_list.append(pssm_bias_chain)
        pssm_log_odds_list.append(pssm_log_odds_chain)

        # Handle bias_by_res
        if bias_by_res is not None and chain_id in bias_by_res:
            bias_by_res_list.append(bias_by_res[chain_id])
        else:
            bias_by_res_list.append(np.zeros([chain_length, 21]))

    # Concatenate all chains
    x = np.concatenate(x_chain_list, 0)
    all_sequence = "".join(chain_seq_list)
    chain_m = np.concatenate(chain_mask_list, 0)
    chain_encoding = np.concatenate(chain_encoding_list, 0)
    chain_m_pos = np.concatenate(fixed_position_mask_list, 0)
    omit_aa_mask_concat = np.concatenate(omit_aa_mask_list, 0)
    pssm_coef_concat = np.concatenate(pssm_coef_list, 0)
    pssm_bias_concat = np.concatenate(pssm_bias_list, 0)
    pssm_log_odds_concat = np.concatenate(pssm_log_odds_list, 0)
    bias_by_res_concat = np.concatenate(bias_by_res_list, 0)

    # Create residue indices
    L = len(all_sequence)
    residue_idx = np.zeros(L, dtype=np.int32)
    l0 = 0
    l1 = 0
    for chain_idx, chain_len in enumerate([len(seq) for seq in chain_seq_list]):
        l1 += chain_len
        residue_idx[l0:l1] = 100 * chain_idx + np.arange(l0, l1)
        l0 += chain_len

    # Handle tied positions
    letter_list_np = np.array(letter_list)
    tied_pos_list_of_lists = []
    tied_beta = np.ones(L)

    if tied_positions is not None:
        for tied_item in tied_positions:
            one_list = []
            for k, v in tied_item.items():
                start_idx = global_idx_start_list[
                    np.argwhere(letter_list_np == k)[0][0]
                ]
                if isinstance(v[0], list):
                    for v_count in range(len(v[0])):
                        one_list.append(start_idx + v[0][v_count] - 1)
                        tied_beta[start_idx + v[0][v_count] - 1] = v[1][v_count]
                else:
                    for v_ in v:
                        one_list.append(start_idx + v_ - 1)
            tied_pos_list_of_lists.append(one_list)

    # Convert sequence to indices
    s = np.asarray([ALPHABET.index(a) for a in all_sequence], dtype=np.int32)

    return ItemFeatures(
        x=x,
        s=s,
        chain_m=chain_m,
        chain_m_pos=chain_m_pos,
        chain_encoding=chain_encoding,
        residue_idx=residue_idx,
        omit_aa_mask=omit_aa_mask_concat,
        pssm_coef=pssm_coef_concat,
        pssm_bias=pssm_bias_concat,
        pssm_log_odds=pssm_log_odds_concat,
        bias_by_res=bias_by_res_concat,
        tied_beta=tied_beta,
        letter_list=letter_list,
        visible_list=visible_list,
        masked_list=masked_list,
        masked_chain_length_list=masked_chain_length_list,
        tied_pos_list_of_lists=tied_pos_list_of_lists,
    )


def _collate_features(
    features_list: list[ItemFeatures],
    device: torch.device,
    ca_only: bool = False,
) -> BatchFeatures:
    """Collate a list of ItemFeatures into padded batch tensors.

    Args:
        features_list: List of ItemFeatures objects
        device: PyTorch device for tensor allocation
        ca_only: If True, coordinates are CA-only [B, L, 3] instead of [B, L, 4, 3]

    Returns:
        BatchFeatures dataclass containing batch tensors and metadata
    """
    B = len(features_list)
    lengths = np.array([len(f.s) for f in features_list], dtype=np.int32)
    L_max = max(lengths)

    # Allocate batch arrays
    if ca_only:
        X = np.zeros([B, L_max, 1, 3])
    else:
        X = np.zeros([B, L_max, 4, 3])

    S = np.zeros([B, L_max], dtype=np.int32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32)
    chain_M_pos = np.zeros([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask = np.zeros([B, L_max, len(ALPHABET)], dtype=np.int32)
    pssm_coef_all = np.zeros([B, L_max], dtype=np.float32)
    pssm_bias_all = np.zeros([B, L_max, 21], dtype=np.float32)
    pssm_log_odds_all = 10000.0 * np.ones([B, L_max, 21], dtype=np.float32)
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)

    # Accumulate metadata lists
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []

    # Pack each item into batch with padding
    for i, features in enumerate(features_list):
        L = len(features.s)

        # Pad and store coordinates
        x_pad = np.pad(
            features.x,
            [[0, L_max - L], [0, 0], [0, 0]],
            "constant",
            constant_values=(np.nan,),
        )
        X[i, :, :, :] = x_pad

        # Store sequences
        S[i, :L] = features.s

        # Pad and store masks
        residue_idx[i, :L] = features.residue_idx
        chain_M[i, :L] = features.chain_m
        chain_M_pos[i, :L] = features.chain_m_pos
        chain_encoding_all[i, :L] = features.chain_encoding
        omit_AA_mask[i, :L] = features.omit_aa_mask

        # Pad and store PSSM
        pssm_coef_all[i, :L] = features.pssm_coef
        pssm_bias_all[i, :L] = features.pssm_bias
        pssm_log_odds_all[i, :L] = features.pssm_log_odds

        # Pad and store bias
        bias_by_res_all[i, :L] = features.bias_by_res

        # Store metadata
        letter_list_list.append(features.letter_list)
        visible_list_list.append(features.visible_list)
        masked_list_list.append(features.masked_list)
        masked_chain_length_list_list.append(features.masked_chain_length_list)
        tied_pos_list_of_lists_list.append(features.tied_pos_list_of_lists)

    # Handle missing coordinates
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    # Create dihedral masks
    jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
    phi_mask = np.pad(jumps, [[0, 0], [1, 0]])
    psi_mask = np.pad(jumps, [[0, 0], [0, 1]])
    omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
    dihedral_mask = np.concatenate(
        [phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]], -1
    )

    # For tied_beta, we need a single array of size L_max
    # We'll collect all tied_beta values and merge them (using max to handle overlaps)
    tied_beta = np.ones(L_max)
    for features in features_list:
        L = len(features.tied_beta)
        # Element-wise max to handle any overlapping tied positions
        tied_beta[:L] = np.maximum(tied_beta[:L], features.tied_beta)

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    S_tensor = torch.from_numpy(S).to(dtype=torch.long, device=device)
    mask_tensor = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_M_tensor = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_M_pos_tensor = torch.from_numpy(chain_M_pos).to(
        dtype=torch.float32, device=device
    )
    chain_encoding_tensor = torch.from_numpy(chain_encoding_all).to(
        dtype=torch.long, device=device
    )
    omit_AA_mask_tensor = torch.from_numpy(omit_AA_mask).to(
        dtype=torch.float32, device=device
    )
    residue_idx_tensor = torch.from_numpy(residue_idx).to(
        dtype=torch.long, device=device
    )
    dihedral_mask_tensor = torch.from_numpy(dihedral_mask).to(
        dtype=torch.float32, device=device
    )
    pssm_coef_tensor = torch.from_numpy(pssm_coef_all).to(
        dtype=torch.float32, device=device
    )
    pssm_bias_tensor = torch.from_numpy(pssm_bias_all).to(
        dtype=torch.float32, device=device
    )
    pssm_log_odds_tensor = torch.from_numpy(pssm_log_odds_all).to(
        dtype=torch.float32, device=device
    )
    bias_by_res_tensor = torch.from_numpy(bias_by_res_all).to(
        dtype=torch.float32, device=device
    )
    tied_beta_tensor = torch.from_numpy(tied_beta).to(
        dtype=torch.float32, device=device
    )

    if ca_only:
        X_out = X_tensor[:, :, 0]
    else:
        X_out = X_tensor

    return BatchFeatures(
        X_out,
        S_tensor,
        mask_tensor,
        lengths,
        chain_M_tensor,
        chain_encoding_tensor,
        letter_list_list,
        visible_list_list,
        masked_list_list,
        masked_chain_length_list_list,
        chain_M_pos_tensor,
        omit_AA_mask_tensor,
        residue_idx_tensor,
        dihedral_mask_tensor,
        tied_pos_list_of_lists_list,
        pssm_coef_tensor,
        pssm_bias_tensor,
        pssm_log_odds_tensor,
        bias_by_res_tensor,
        tied_beta_tensor,
    )


def tied_featurize(
    batch: list[Protein],
    device: torch.device,
    ca_only: bool = False,
) -> BatchFeatures:
    """Pack and pad batch of protein structures into PyTorch tensors for ProteinMPNN.

    This is the modern API that works with Protein objects directly.

    Args:
        batch: List of Protein objects, each containing a protein structure and
            all associated configuration parameters (chain selection, fixed positions, etc.)
        device: PyTorch device for tensor allocation (e.g., torch.device('cuda:0') or torch.device('cpu'))
        ca_only: If True, use only CA (carbon alpha) atoms instead of full backbone (N, CA, C, O).

    Returns:
        BatchFeatures object containing all featurized data including coordinates,
        sequences, masks, chain encodings, and configuration parameters.

    Examples:
        Basic usage with chain selection:

        >>> import torch
        >>> from dauparas_proteinmpnn.io import parse_pdb
        >>> from dauparas_proteinmpnn.featurize import tied_featurize, Protein
        >>>
        >>> # Parse PDB file
        >>> structures = parse_pdb('protein.pdb')
        >>> protein = structures[0]
        >>>
        >>> # Create batch with chain configuration
        >>> batch = [
        ...     Protein(
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
        ...     Protein(
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
        ...     Protein(
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
        ...     Protein(
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
        ...     Protein(
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
        ...     Protein(
        ...         structure=protein1,
        ...         masked_chains=['A'],
        ...         visible_chains=['B']
        ...     ),
        ...     Protein(
        ...         structure=protein2,
        ...         masked_chains=['C'],
        ...         visible_chains=[]
        ...     )
        ... ]
        >>>
        >>> result = tied_featurize(batch=batch, device=device)
    """
    # Process each protein individually
    features_list = [_process_item(protein, ca_only=ca_only) for protein in batch]

    # Collate into batch tensors
    return _collate_features(features_list, device, ca_only)


def featurize_pdb(
    pdb: Path | TextIO,
    designed_chains: list[str],
    fixed_chains: list[str],
    device,
    chain_designed_positions: dict | None = None,
) -> BatchFeatures:
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
) -> BatchFeatures:
    """Featurize a Structure object for ProteinMPNN.

    Args:
        structure: Structure object containing protein data
        designed_chains: List of chain IDs to design
        fixed_chains: List of chain IDs to keep fixed as context
        device: PyTorch device for tensors
        chain_designed_positions: Optional dict of designed positions per chain

    Returns:
        BatchFeatures with all featurized data
    """
    all_chains = designed_chains + fixed_chains
    structure = select_chains(structure, all_chains)

    protein = Protein(
        structure=structure,
        masked_chains=designed_chains,
        visible_chains=fixed_chains,
        chain_designed_positions=chain_designed_positions,
    )

    return tied_featurize(
        batch=[protein],
        device=device,
    )


def encode_sequence(features: BatchFeatures, seq: str) -> BatchFeatures:
    input_seq_length = len(seq)
    S_input = torch.tensor([ALPHABET_DICT[AA] for AA in seq], device=features.S.device)[
        None, :
    ].repeat(features.X.shape[0], 1)
    # assumes that S and S_input are ALPHABETically sorted for masked_chains
    features.S[:, :input_seq_length] = S_input
    return features
