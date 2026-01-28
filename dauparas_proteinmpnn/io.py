import json
from dataclasses import dataclass
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import IUPACData
from pathlib import Path
from typing import Optional, TextIO

import torch


@dataclass
class ChainData:
    """Data for a single protein chain.
    
    Attributes:
        seq: Amino acid sequence as single-letter codes
        coords: Dictionary mapping atom names to coordinate lists
            Keys: 'N', 'CA', 'C', 'O' (for full backbone) or just 'CA' (for CA-only)
            Values: List of [x, y, z] coordinates for each residue
    """
    seq: str
    coords: dict[str, list[list[float]]]


@dataclass
class Structure:
    """Protein structure data.
    
    Attributes:
        name: Unique identifier (typically filename)
        num_of_chains: Number of chains in structure
        seq: Concatenated sequence of all chains
        chains: Dictionary mapping chain IDs to ChainData objects
    """
    name: str
    num_of_chains: int
    seq: str
    chains: dict[str, ChainData]

def parse_pdb(pdb: Path | TextIO, chain_ids: Optional[list[str]] = None) -> Structure:
    """Parse a PDB file into a Structure object.
    
    Args:
        pdb: Path to PDB file or file-like object
        chain_ids: Optional list of chain IDs to parse (None = all chains)
        
    Returns:
        Structure object containing parsed protein data
    """
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure("-", pdb)
    chains_data = {}

    assert structure is not None, "Failed to parse PDB file."
    for chain in structure.get_chains():
        if chain_ids and chain.id not in chain_ids:
            continue
        
        coords_N = []
        coords_O = []
        coords_CA = []
        coords_C = []
        seq = []
        
        for residue in chain:
            seq.append(
                IUPACData.protein_letters_3to1[residue.get_resname().capitalize()]
            )
            for atom in residue.get_atoms():
                coords = atom.get_coord().tolist()
                match atom.id:
                    case "N":
                        coords_N.append(coords)
                    case "O":
                        coords_O.append(coords)
                    case "CA":
                        coords_CA.append(coords)
                    case "C":
                        coords_C.append(coords)
        
        chains_data[chain.id] = ChainData(
            seq="".join(seq),
            coords={
                "N": coords_N,
                "CA": coords_CA,
                "C": coords_C,
                "O": coords_O,
            }
        )
    
    return Structure(
        name=pdb.name if hasattr(pdb, 'name') else str(pdb),
        num_of_chains=len(chains_data),
        seq="".join(chain.seq for chain in chains_data.values()),
        chains=chains_data,
    )


def select_chains(protein: Structure, chains: list[str]) -> Structure:
    """Select specific chains from a Structure.
    
    Args:
        protein: Source Structure object
        chains: List of chain IDs to select
        
    Returns:
        New Structure containing only the specified chains
    """
    selected_chains = {chain_id: protein.chains[chain_id] for chain_id in chains}
    return Structure(
        name=protein.name,
        num_of_chains=len(chains),
        seq="".join(selected_chains[chain_id].seq for chain_id in chains),
        chains=selected_chains,
    )


def write_scores(id: str, designed_score: torch.Tensor, global_score: torch.Tensor, out_jsonl: TextIO):
    out_json = {
        "id": "pdb",
        "scores": designed_score.tolist(),
        "global_scores": global_score.tolist(),
    }
    out_jsonl.write(f"{json.dumps(out_json)}\n")


if __name__ == "__main__":
    from dataclasses import asdict

    pdb_path = Path("1ahw_modelled.pdb")
    chain_ids = ["H", "L"]  # Example chain IDs to select

    with open(pdb_path, "r") as pdb_file:
        print(f"{pdb_file.name=}")
        print(f"{type(pdb_file)=}")
        protein = parse_pdb(pdb_file, chain_ids=chain_ids)
        print(json.dumps(asdict(protein), indent=2))