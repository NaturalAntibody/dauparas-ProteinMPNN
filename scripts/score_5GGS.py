import functools
import json
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import torch
from tqdm import tqdm
from dauparas_proteinmpnn import (
    encode_sequence,
    load_abmpnn,
    score,
    featurize_pdb,
    ALPHABET,
)
from dauparas_proteinmpnn.score import score_deterministic
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

if __name__ == "__main__":
    device = torch.device("cpu")
    model = load_abmpnn(device=device)

    mut_df = pd.read_csv("data/external/5GGS_mutation_positions.csv", index_col=0)
    mut_df = mut_df.drop(columns=["new_residue"]).drop_duplicates().reset_index()
    seq = "VQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP"
    score_chain_id = "H"
    condition_chain_ids = ["A", "L"]

    res = {}
    score_fn = functools.partial(score, model=model, sample_count=10)
    score_deterministic_fn = functools.partial(score_deterministic, model=model)
    for score_fn in [score_fn, score_deterministic_fn]:
        for row in tqdm(
            mut_df.itertuples(index=True),
            total=len(mut_df),
            desc=f"Scoring with {score_fn.func.__name__.rstrip('_fn')}",
        ):
            chain_designed_positions = {
                score_chain_id: [row.positional_residue_index + 1],
            }
            features = featurize_pdb(
                pdb=Path("data/external/5GGS_normalized.pdb"),
                device=device,
                designed_chains=[score_chain_id],
                fixed_chains=condition_chain_ids,
                chain_designed_positions=chain_designed_positions,
            )
            features = encode_sequence(features, seq)
            scoring_result = score_fn(
                features=features,
                positions_to_score=[row.positional_residue_index],
            )
            assert torch.allclose(
                scoring_result.designed_log_probs, scoring_result.global_log_probs
            )
            proba = dict(
                zip(
                    ALPHABET, scoring_result.designed_log_probs.squeeze().exp().tolist()
                )
            )
            res[row.pdb_residue_number] = {
                "pdb_residue_number": row.pdb_residue_number,
                "positional_residue_index": row.positional_residue_index,
                "orig_residue": row.orig_residue,
                "proba": proba,
            }
        json.dump(
            res,
            open(
                f"abmpnn_5GGS_mutation_{score_fn.func.__name__.rstrip('_fn')}.json", "w"
            ),
        )
