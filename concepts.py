"""Shared concept definitions, constants, and loader utilities."""

from pathlib import Path

import torch

DATA_DIR = Path("data")

# Maps concept name -> (parquet file, positive column, negative column)
CONCEPTS = {
    "disulfide": ("disulfide_features.parquet", "disulfide_cys_indices", "free_cys_indices"),
    "ss_helix": ("ss_features.parquet", "helix_indices", "coil_indices"),
    "ss_sheet": ("ss_features.parquet", "sheet_indices", "coil_indices"),
    "ss_helix_sheet": ("ss_features.parquet", "helix_indices", "sheet_indices"),
    "sasa": ("sasa_features.parquet", "buried_indices", "exposed_indices"),
    "ppi": ("ppi_features.parquet", "interface_indices", "non_interface_indices"),
    "binding": ("binding_features.parquet", "binding_indices", "non_binding_indices"),
    "ptm": ("ptm_features.parquet", "ptm_indices", "non_ptm_indices"),
    "disorder": ("disorder_features.parquet", "disordered_indices", "ordered_indices"),
}


def load_concept_vectors(name: str) -> dict:
    """Load pre-computed concept vectors from disk."""
    path = DATA_DIR / f"{name}_concept_vectors.pt"
    return torch.load(path, weights_only=False)


def list_computed_concepts() -> list[str]:
    """Return names of concepts that have computed vector files."""
    computed = []
    for name in CONCEPTS:
        path = DATA_DIR / f"{name}_concept_vectors.pt"
        if path.exists():
            computed.append(name)
    return computed
