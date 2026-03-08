"""ESM3 inference engine with forward hooks and concept vector projection."""

import torch
from torch import Tensor
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein

from concepts import CONCEPTS, DATA_DIR, load_concept_vectors, list_computed_concepts

MAX_SEQUENCE_LENGTH = 1000


class ESM3Engine:
    """Singleton-style engine: loads model once, registers hooks, projects residues."""

    def __init__(self):
        self.model: ESM3 | None = None
        self.device: torch.device = torch.device("cpu")
        self.n_layers: int = 0
        self.concept_vectors: dict[str, dict[int, Tensor]] = {}
        self._captured: dict[int, Tensor] = {}
        self._handles: list = []

    def load(self, device: torch.device | None = None) -> None:
        """Load ESM3 model and register forward hooks on all transformer blocks."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
        self.model.eval()
        self.n_layers = len(self.model.transformer.blocks)

        # Register hooks on all layers
        def make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                self._captured[layer_idx] = output.detach()
            return hook_fn

        for layer_idx in range(self.n_layers):
            h = self.model.transformer.blocks[layer_idx].register_forward_hook(
                make_hook(layer_idx)
            )
            self._handles.append(h)

    def load_concept_vectors(self) -> None:
        """Scan data/ for computed concept vector files and load normalized vectors."""
        self.concept_vectors.clear()
        for name in list_computed_concepts():
            saved = load_concept_vectors(name)
            layer_data = saved["layers"]
            n_layers = saved["n_layers"]
            vecs: dict[int, Tensor] = {}
            for layer_idx in range(n_layers):
                vec = layer_data[layer_idx]["concept_vector_normalized"]
                vecs[layer_idx] = vec.to(self.device)
            self.concept_vectors[name] = vecs

    def run_inference(
        self, sequence: str, concepts: list[str] | None = None
    ) -> dict:
        """Run ESM3 forward pass and project all residues onto concept directions.

        Returns dict with sequence, n_residues, n_layers, concepts list,
        and projections: {concept: [[float] * n_layers] * n_residues}.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if len(sequence) > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds maximum {MAX_SEQUENCE_LENGTH}"
            )

        # Determine which concepts to project
        if concepts is None:
            concepts = list(self.concept_vectors.keys())
        else:
            missing = [c for c in concepts if c not in self.concept_vectors]
            if missing:
                raise ValueError(f"Concept vectors not loaded: {missing}")

        # Tokenize and run forward pass
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)

        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type != "cpu",
        ):
            self.model.forward(
                sequence_tokens=protein_tensor.sequence.unsqueeze(0),
            )

        n_residues = len(sequence)

        # Project residues onto concept directions: vectorized per layer per concept
        projections: dict[str, list[list[float]]] = {}
        for concept_name in concepts:
            concept_dirs = self.concept_vectors[concept_name]
            # Build matrix [n_residues x n_layers]
            residue_projections: list[list[float]] = [
                [] for _ in range(n_residues)
            ]
            for layer_idx in range(self.n_layers):
                hidden = self._captured[layer_idx]  # (1, L, D)
                direction = concept_dirs[layer_idx]
                # Vectorized: extract all residue vectors at once (skip BOS at index 0)
                residue_vecs = hidden[0, 1 : n_residues + 1, :].float()  # (n_residues, D)
                projs = (residue_vecs @ direction).tolist()  # list of n_residues floats
                for r_idx in range(n_residues):
                    residue_projections[r_idx].append(projs[r_idx])
            projections[concept_name] = residue_projections

        # Clear captured hiddens to free memory
        self._captured.clear()

        return {
            "sequence": sequence,
            "n_residues": n_residues,
            "n_layers": self.n_layers,
            "concepts": concepts,
            "projections": projections,
        }
