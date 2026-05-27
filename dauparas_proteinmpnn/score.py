from dataclasses import dataclass
from pathlib import Path
from turtle import pos

import torch

from dauparas_proteinmpnn.featurize import (
    BatchFeatures,
    featurize_pdb,
    encode_sequence,
)
from dauparas_proteinmpnn.models import load_proteinmpnn
import torch.nn.functional as F

from dauparas_proteinmpnn.protein_mpnn_utils import ProteinMPNN


@dataclass
class ScoringResult:
    designed_scores: torch.Tensor
    global_scores: torch.Tensor
    logits: torch.Tensor
    designed_log_probs: torch.Tensor
    global_log_probs: torch.Tensor


def _scores_from_averaged_logits(S, averaged_logits, mask, positions_to_score):
    """Calculate scores from averaged logits"""
    log_probs = F.log_softmax(averaged_logits, dim=-1)
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    if positions_to_score is not None:
        loss = loss[:, positions_to_score]
        mask = mask[:, positions_to_score]
        log_probs = log_probs[:, positions_to_score, :]
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores, log_probs


def score_deterministic(
    model: ProteinMPNN,
    features: BatchFeatures,
    positions_to_score: list[int] | None = None,
) -> ScoringResult:
    """Score with fixed sequential decoding order for reproducibility."""

    logits, _ = model(
        features.X,
        features.S,
        features.mask,
        features.chain_M * features.chain_M_pos,
        features.residue_idx,
        features.chain_encoding_all,
        randn=None,
        use_constant_decoding_order=True,
    )

    mask_for_loss = features.mask * features.chain_M * features.chain_M_pos

    designed_scores, designed_log_probs = _scores_from_averaged_logits(
        features.S, logits, mask_for_loss, positions_to_score
    )
    global_scores, global_log_probs = _scores_from_averaged_logits(
        features.S, logits, features.mask, positions_to_score
    )

    return ScoringResult(
        designed_scores, global_scores, logits, designed_log_probs, global_log_probs
    )


def score_deterministic_batch(
    model,
    features_list: list[BatchFeatures],
    positions_to_score: list[int] | None = None,
) -> list[ScoringResult]:
    """Score multiple feature sets in a single batched forward pass with fixed decoding order.

    Args:
        model: The ProteinMPNN model
        features_list: List of BatchFeatures objects to score
        positions_to_score: Optional list of position indices to score

    Returns:
        List of ScoringResult objects, one for each input feature set
    """
    if not features_list:
        return []

    # Get max length for padding
    max_len = max(f.chain_M.shape[1] for f in features_list)
    device = features_list[0].X.device
    batch_size = len(features_list)

    # Determine number of atoms (CA-only vs full atom)
    num_atoms = features_list[0].X.shape[2]

    # Pre-allocate batched tensors
    X_batch = torch.zeros((batch_size, max_len, num_atoms, 3), device=device)
    S_batch = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    mask_batch = torch.zeros((batch_size, max_len), device=device)
    chain_M_batch = torch.zeros((batch_size, max_len), device=device)
    residue_idx_batch = torch.zeros(
        (batch_size, max_len), dtype=torch.long, device=device
    )
    chain_encoding_batch = torch.zeros(
        (batch_size, max_len), dtype=torch.long, device=device
    )
    decoding_order_batch = torch.zeros(
        (batch_size, max_len), dtype=torch.long, device=device
    )

    # Track original lengths for unpacking results
    lengths = []

    # Pack features into batch
    for i, features in enumerate(features_list):
        seq_len = features.chain_M.shape[1]
        lengths.append(seq_len)

        X_batch[i, :seq_len] = features.X[0]
        S_batch[i, :seq_len] = features.S[0]
        mask_batch[i, :seq_len] = features.mask[0]
        chain_M_batch[i, :seq_len] = features.chain_M[0] * features.chain_M_pos[0]
        residue_idx_batch[i, :seq_len] = features.residue_idx[0]
        chain_encoding_batch[i, :seq_len] = features.chain_encoding_all[0]
        decoding_order_batch[i, :seq_len] = torch.arange(seq_len, device=device)

    # Single batched forward pass
    logits_batch, _ = model(
        X_batch,
        S_batch,
        mask_batch,
        chain_M_batch,
        residue_idx_batch,
        chain_encoding_batch,
        randn=None,
        use_constant_decoding_order=True,
        decoding_order=decoding_order_batch,
    )

    # Unpack results for each sample
    results = []
    for i, seq_len in enumerate(lengths):
        logits = logits_batch[i : i + 1, :seq_len]
        S = S_batch[i : i + 1, :seq_len]
        mask = mask_batch[i : i + 1, :seq_len]
        chain_M = chain_M_batch[i : i + 1, :seq_len]

        mask_for_loss = mask * chain_M

        designed_scores = _scores_from_averaged_logits(
            S, logits, mask_for_loss, positions_to_score
        )
        global_scores = _scores_from_averaged_logits(
            S, logits, mask, positions_to_score
        )

        results.append(ScoringResult(designed_scores, global_scores, logits))

    return results


def score(
    model: ProteinMPNN,
    features: BatchFeatures,
    sample_count: int = 10,
    positions_to_score: list[int] | None = None,
) -> ScoringResult:
    """Score by averaging logits across multiple samples with random decoding orders.

    This provides a more robust estimate by averaging over different decoding orders,
    but is non-deterministic.
    """
    noise = torch.randn(
        (sample_count, features.chain_M.shape[1]), device=features.X.device
    )
    noise = torch.ones_like(noise)
    X = features.X.expand(sample_count, -1, -1, -1)
    S = features.S.expand(sample_count, -1)
    mask = features.mask.expand(sample_count, -1)
    chain_M = (features.chain_M * features.chain_M_pos).expand(sample_count, -1)
    residue_idx = features.residue_idx.expand(sample_count, -1)
    chain_encoding_all = features.chain_encoding_all.expand(sample_count, -1)

    logits, _ = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, noise)
    # Average logits across samples
    averaged_logits = logits.mean(dim=0, keepdim=True)


    mask_for_loss = mask[0:1] * chain_M[0:1]  # Take first sample's mask
    S_single = S[0:1]  # Take first sample's sequence

    designed_scores, designed_log_probs = _scores_from_averaged_logits(
        S_single, averaged_logits, mask_for_loss, positions_to_score
    )
    global_scores, global_log_probs = _scores_from_averaged_logits(
        S_single, averaged_logits, mask[0:1], positions_to_score
    )

    return ScoringResult(
        designed_scores.detach().cpu(),
        global_scores.detach().cpu(),
        averaged_logits.detach().cpu(),
        designed_log_probs.detach().cpu(),
        global_log_probs.detach().cpu(),
    )


if __name__ == "__main__":
    import numpy as np

    device = torch.device("cpu")
    model = load_proteinmpnn(device=device)
    score_chain_id = "H"
    condition_chain_ids = ["H"]
    sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFTHLTYWGQGTLVTVSS"
    fixed_chains = list(set(condition_chain_ids) - set(score_chain_id))

    features = featurize_pdb(
        Path("./data/1N8Z_normalized.pdb"),
        designed_chains=[score_chain_id],
        fixed_chains=fixed_chains,
        device=device,
    )

    features = encode_sequence(features, sequence)

    print("Testing deterministic scoring (single forward pass):")
    for i in range(5):
        scoring_result = score_deterministic(model, features, positions_to_score=None)

        perplexity = np.exp(scoring_result.designed_scores.numpy(force=True).item())
        print(f"Perplexity: {perplexity}")

    print("\nTesting multi-sample averaging (non-deterministic):")
    for i in range(5):
        scoring_result = score(
            model, features, sample_count=10, positions_to_score=None
        )

        perplexity = np.exp(scoring_result.designed_scores.numpy(force=True).item())
        print(f"Perplexity: {perplexity}")

    print("\nTesting batch scoring equivalence:")
    sequences = [
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFTHLTYWGQGTLVTVSS",
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRYTAPSFYTFDYWGQGTLVTVSS",
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWAAGRSYTFDYWGQGTLVTVSS",
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWAAGRFYAFDYWGQGTLVTVSS",
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWNAPSFYALNYWGQGTLVTVSS",
    ]

    # Score individually
    print("Individual scoring:")
    individual_features = []
    individual_results = []
    features = featurize_pdb(
        Path("./data/1N8Z_normalized.pdb"),
        designed_chains=[score_chain_id],
        fixed_chains=fixed_chains,
        device=device,
    )
    for i, seq in enumerate(sequences):
        feat = features.clone()
        feat = encode_sequence(feat, seq)
        individual_features.append(feat)
        result = score_deterministic(model, feat, positions_to_score=None)
        individual_results.append(result)
        perplexity = np.exp(result.designed_scores.numpy(force=True).item())
        print(f"  Sequence {i + 1}: Perplexity = {perplexity:.6f}")

    # Score in batch
    print("\nBatch scoring:")
    batch_results = score_deterministic_batch(
        model, individual_features, positions_to_score=None
    )
    for i, result in enumerate(batch_results):
        perplexity = np.exp(result.designed_scores.numpy(force=True).item())
        print(f"  Sequence {i + 1}: Perplexity = {perplexity:.6f}")

    # Compare results
    print("\nComparison:")
    all_match = True
    for i in range(len(sequences)):
        ind_score = individual_results[i].designed_scores.item()
        batch_score = batch_results[i].designed_scores.item()
        diff = abs(ind_score - batch_score)
        match = diff < 1e-6
        all_match = all_match and match
        print(
            f"  Sequence {i + 1}: Individual = {ind_score:.6f}, Batch = {batch_score:.6f}, Diff = {diff:.2e}, Match = {match}"
        )

    print(f"\nAll scores match: {all_match}")
