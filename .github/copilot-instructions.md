# ProteinMPNN AI Coding Agent Instructions

## Project Overview

ProteinMPNN is a deep learning tool for protein sequence design given a backbone structure. The model takes PDB files as input and generates amino acid sequences optimized for the given backbone, with extensive control over which chains/positions to design, amino acid biases, and symmetry constraints.

**Key Paper**: [Robust deep learning-based protein sequence design using ProteinMPNN](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1)

## Architecture & Core Components

### 1. Data Flow Pipeline
```
PDB File → parse_pdb() → Structure dataclass → Protein dataclass → tied_featurize() → BatchFeatures → model.sample() → sequences
```

**Key Files**:
- [dauparas_proteinmpnn/io.py](dauparas_proteinmpnn/io.py): PDB parsing using BioPython
- [dauparas_proteinmpnn/featurize.py](dauparas_proteinmpnn/featurize.py): Feature engineering and constraint handling
- [dauparas_proteinmpnn/models.py](dauparas_proteinmpnn/models.py): Model loading utilities
- [dauparas_proteinmpnn/sample.py](dauparas_proteinmpnn/sample.py): Sequence generation
- [dauparas_proteinmpnn/score.py](dauparas_proteinmpnn/score.py): Scoring existing sequences

### 2. Structure Dataclass (Modern Format)

**Critical Convention**: PDB data is stored as `Structure` dataclass with nested `ChainData`:
```python
@dataclass
class ChainData:
    seq: str  # amino acid sequence
    coords: dict[str, list[list[float]]]  # {'N': [[x,y,z], ...], 'CA': ..., 'C': ..., 'O': ...}

@dataclass
class Structure:
    name: str
    num_of_chains: int
    seq: str  # concatenated sequence of all chains
    chains: dict[str, ChainData]  # {'A': ChainData(...), 'B': ChainData(...)}
```

**Legacy format**: Old dict format with chain-specific keys (e.g., `seq_chain_A`, `coords_chain_A`) still exists in legacy code but should not be used for new development.

### 3. Protein Dataclass (Unified API)

The `Protein` dataclass in [featurize.py](dauparas_proteinmpnn/featurize.py) is the **unified interface** for specifying all design constraints:
- `structure`: Structure dataclass with backbone coordinates
- `masked_chains`: chains to design
- `visible_chains`: chains kept as fixed context
- `fixed_positions`: dict mapping chains to 1-based position lists to keep fixed
- `omit_aa`: dict of (positions, amino_acids) tuples to exclude from design
- `tied_positions`: list of dicts enforcing same AA at different positions (for symmetry)
- `pssm`: Position-Specific Scoring Matrix bias
- `bias_by_res`: per-residue amino acid bias arrays [seq_length, 21]

**Design Pattern**: All new code should use `Protein` dataclass. Pass list of `Protein` objects directly to `tied_featurize()`.

## Critical Developer Workflows

### Running ProteinMPNN

**Standard workflow** (see [examples/submit_example_4.sh](examples/submit_example_4.sh)):
```bash
# 1. Parse PDB files to JSONL
python helper_scripts/parse_multiple_chains.py --input_path=inputs/pdbs/ --output_path=parsed.jsonl

# 2. Specify which chains to design
python helper_scripts/assign_fixed_chains.py --input_path=parsed.jsonl --output_path=assigned.jsonl --chain_list "A C"

# 3. (Optional) Fix specific positions
python helper_scripts/make_fixed_positions_dict.py --input_path=parsed.jsonl --output_path=fixed.jsonl --chain_list "A C" --position_list "1 2 3, 10 11 12"

# 4. Run design
python protein_mpnn_run.py \
    --jsonl_path parsed.jsonl \
    --chain_id_jsonl assigned.jsonl \
    --fixed_positions_jsonl fixed.jsonl \
    --out_folder outputs/ \
    --num_seq_per_target 10 \
    --sampling_temp "0.1"
```

### Testing Strategy

**Comprehensive regression test suite** ([tests/test_tied_featurize_regression.py](tests/test_tied_featurize_regression.py)):
- Compares outputs against saved golden values to catch behavioral changes
- Run: `pytest tests/ -v` or `poetry run pytest tests/ -v`
- Regenerate data: `python scripts/generate_regression_data.py` (only after verifying changes are intentional)
- **8 test cases** covering: basic design, CA-only mode, fixed positions, omit_aa, tied positions, PSSM, batch processing
- **Critical**: Always run regression tests when refactoring featurization logic to ensure numerical accuracy is preserved

**PDB warnings**: BioPython PDB construction warnings are filtered in [pyproject.toml](pyproject.toml) to keep test output clean.

### Environment Setup

**Package management**: Poetry
```bash
poetry install  # Install dependencies from pyproject.toml
poetry run pytest  # Run tests in virtual environment
```

**Dependencies**:
- PyTorch ≥2.2.0
- numpy <2.0 (compatibility constraint)
- biopython ==1.84
- Black line length: 120 (see [pyproject.toml](pyproject.toml))

## Project-Specific Conventions

### 1. Alphabet & Encoding

**Standard amino acid alphabet** (21 chars including X):
```python
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"  # defined in featurize.py
```
All encoding/decoding must use this order. X represents unknown/masked positions.

### 2. Position Indexing

**Critical**: Position indices in user-facing APIs (JSON files, command args) are **1-based**. Internal tensors use 0-based indexing.
- Example: `fixed_positions={'A': [1, 2, 3]}` means first 3 residues
- Conversion happens in `tied_featurize()`

### 3. Model Weights Organization

**Location**: [dauparas_proteinmpnn/model_weights/](dauparas_proteinmpnn/model_weights/)
- `vanilla_model_weights/v_48_020.pt`: Default full-atom model (48 edges, 0.20Å noise)
- `soluble_model_weights/`: Trained on soluble proteins only
- `ca_model_weights/`: CA-only models (use with `--ca_only` flag)
- `abmpnn.pt`: Antibody-specific model

**Model naming convention**: `v_{num_edges}_{noise_level}` (e.g., v_48_020 = 48 edges, 0.20Å noise)

### 4. Helper Script Design Pattern

All helper scripts ([helper_scripts/](helper_scripts/)) follow this pattern:
- Read parsed JSONL structures
- Transform into constraint dictionaries
- Write single-line JSONL output (one JSON object per line)
- Used as intermediate preprocessing steps before main `protein_mpnn_run.py`

**Example outputs**:
- `assign_fixed_chains.py`: `{"5TTA": [["A", "C"], ["B"]]}`  # [designed_chains, fixed_chains]
- `make_fixed_positions_dict.py`: `{"5TTA": {"A": [1,2,3], "B": [10,11]}}`
- `make_tied_positions_dict.py`: `{"5TTA": [{"A": [1], "B": [1]}, {"A": [2], "B": [2]}]}`  # each dict ties positions

### 5. Tied Positions for Symmetry

**Purpose**: Enforce same amino acid at multiple positions (e.g., in homooligomers)

**Format**: List of dicts, each dict groups positions to tie together:
```python
[
    {"A": [5], "B": [5]},  # Position 5 in chains A and B will be the same AA
    {"A": [10], "C": [15]}  # Position 10 in A and 15 in C will be the same AA
]
```

**Homooligomer shortcut**: Use `--homooligomer 1` in `make_tied_positions_dict.py` to auto-tie all corresponding positions across chains.

## Integration Points & External Dependencies

### BioPython PDB Parsing

**Used exclusively in** [io.py](dauparas_proteinmpnn/io.py):
- `Bio.PDB.PDBParser`: Parse PDB files
- `Bio.SeqUtils.IUPACData.protein_letters_3to1`: Convert 3-letter to 1-letter codes
- Supports file paths or file-like objects (TextIO)

### PyTorch Model Interface

**ProteinMPNN class** ([protein_mpnn_utils.py](dauparas_proteinmpnn/protein_mpnn_utils.py)):
- `model.sample()`: Generate sequences with temperature sampling
- `model.forward()`: Get logits for scoring
- Requires specific input tensors from `BatchFeatures`:
  - `X`: [batch, length, 4, 3] backbone coordinates (N, CA, C, O)
  - `S`: [batch, length] amino acid indices
  - `chain_M`: [batch, length] mask for designed positions
  - `chain_encoding_all`: [batch, length] chain identity encoding
  - `residue_idx`: [batch, length] residue indices for RBF encoding

### Output Format

**FASTA-like with metadata**:
```
>3HTN, score=1.1705, global_score=1.2045, fixed_chains=['B'], designed_chains=['A', 'C'], model_name=v_48_020, seed=37
NMYSYKK.../NMYSYKK...  # sequences separated by / for different chains
>T=0.1, sample=1, score=0.7291, seq_recovery=0.5736
NMYKYKKIGN...
```

- `score`: avg negative log prob over designed positions
- `global_score`: avg negative log prob over all positions
- `seq_recovery`: fraction matching input sequence (for scoring mode)

## Common Pitfalls & Edge Cases

1. **Modular code organization** (refactoring complete): [protein_mpnn_utils.py](dauparas_proteinmpnn/protein_mpnn_utils.py) has been fully refactored into modular components:
   - Featurization → [featurize.py](dauparas_proteinmpnn/featurize.py) with `tied_featurize()` as main entry point
   - PDB parsing → [io.py](dauparas_proteinmpnn/io.py) with `parse_pdb()` returning `Structure` dataclass
   - Sequence generation → [sample.py](dauparas_proteinmpnn/sample.py)
   - Scoring → [score.py](dauparas_proteinmpnn/score.py)
   - **API**: `tied_featurize(batch: list[Protein], device, ca_only=False) -> BatchFeatures`
   - **When writing new code**: Always use the modular API. Legacy code in protein_mpnn_utils.py remains only for backward compatibility.

2. **CA-only mode**: Requires special model weights and `--ca_only` flag. Parsing logic differs (see [parse_multiple_chains.py](helper_scripts/parse_multiple_chains.py)).

3. **PSSM format**: Requires `.npz` files with arrays: `{chain}_coef`, `{chain}_bias`, `{chain}_odds`. Use [make_pssm_input_dict.py](helper_scripts/make_pssm_input_dict.py) to create proper format.

4. **Chain ordering**: Always sorted alphabetically in `tied_featurize()`. Chain assignment must respect this ordering.

5. **Batch processing**: Structures in batch can have different lengths. Padding/masking handled by `tied_featurize()`.

## Key File Cross-References

- Main entry point: [protein_mpnn_run.py](protein_mpnn_run.py)
- Modern package API: [dauparas_proteinmpnn/](dauparas_proteinmpnn/) (prefer this for new code)
- Legacy code: [protein_mpnn_utils.py](dauparas_proteinmpnn/protein_mpnn_utils.py) (being phased out)
- Example workflows: [examples/submit_example_*.sh](examples/)
- Colab notebooks: [colab_notebooks/](colab_notebooks/) (quickstart demos)

**Note**: This repository focuses on **inference** (sequence design from structures). Training code in [training/](training/) is preserved for reference but not actively maintained.
