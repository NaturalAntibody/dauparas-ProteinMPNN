from dataclasses import dataclass
from pathlib import Path

import torch

from dauparas_proteinmpnn.featurize import (
    TiedFeaturizeResult,
    featurize_pdb,
    encode_sequence,
)
from dauparas_proteinmpnn.models import load_proteinmpnn
import torch.nn.functional as F


@dataclass
class ScoringResult:
    designed_scores: torch.Tensor
    global_scores: torch.Tensor
    logits: torch.Tensor


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
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores


def score_deterministic(
    model,
    features: TiedFeaturizeResult,
    positions_to_score: list[int] | None = None,
) -> ScoringResult:
    """Score with fixed sequential decoding order for reproducibility.
    """
    # Use fixed sequential decoding order
    decoding_order = torch.arange(
        features.chain_M.shape[1], device=features.X.device
    ).unsqueeze(0)
    
    logits, _ = model(
        features.X,
        features.S,
        features.mask,
        features.chain_M * features.chain_M_pos,
        features.residue_idx,
        features.chain_encoding_all,
        randn=None,
        use_input_decoding_order=True,
        decoding_order=decoding_order,
    )

    mask_for_loss = features.mask * features.chain_M * features.chain_M_pos

    designed_scores = _scores_from_averaged_logits(
        features.S, logits, mask_for_loss, positions_to_score
    )
    global_scores = _scores_from_averaged_logits(
        features.S, logits, features.mask, positions_to_score
    )

    return ScoringResult(designed_scores, global_scores, logits)


def score(
    model,
    features: TiedFeaturizeResult,
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

    designed_scores = _scores_from_averaged_logits(
        S_single, averaged_logits, mask_for_loss, positions_to_score
    )
    global_scores = _scores_from_averaged_logits(
        S_single, averaged_logits, mask[0:1], positions_to_score
    )

    return ScoringResult(designed_scores, global_scores, averaged_logits)


if __name__ == "__main__":
    import numpy as np

    device = torch.device("cuda:3")
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
        scoring_result = score(model, features, sample_count=10, positions_to_score=None)

        perplexity = np.exp(scoring_result.designed_scores.numpy(force=True).item())
        print(f"Perplexity: {perplexity}")
