"""
Extract concept vectors from ESM3's residual stream at every transformer layer.

Usage:
  python extract_concept_vector.py <concept>

Available concepts:
  disulfide    - disulfide-bonded vs free cysteines
  ss_helix     - helix vs coil residues
  ss_sheet     - sheet vs coil residues
  ss_helix_sheet - helix vs sheet residues
  sasa         - buried vs exposed residues
  ppi          - interface vs non-interface residues
  binding      - ligand/metal-binding vs non-binding residues
  ptm          - post-translationally modified vs unmodified residues
  disorder     - disordered vs ordered residues

For each concept, registers a forward hook on EVERY transformer layer,
runs inference on all chains in the dataset, and saves per-layer concept
vectors to data/<concept>_concept_vectors.pt.

ESM3 architecture notes:
  - model.encoder (EncodeInputs) sums all modality embeddings into a
    unified representation per token position
  - model.transformer.blocks is an nn.ModuleList of UnifiedTransformerBlock
  - ESM3 tokenization prepends BOS: residue index i -> token index i+1
"""

import argparse
from pathlib import Path

import torch
import pyarrow.parquet as pq
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein


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


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-layer concept vectors from ESM3."
    )
    parser.add_argument(
        "concept",
        choices=sorted(CONCEPTS.keys()),
        help="Which concept to extract.",
    )
    args = parser.parse_args()

    concept = args.concept
    parquet_file, pos_col, neg_col = CONCEPTS[concept]
    parquet_path = DATA_DIR / parquet_file
    output_path = DATA_DIR / f"{concept}_concept_vectors.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Concept: {concept}")
    print(f"  Positive column: {pos_col}")
    print(f"  Negative column: {neg_col}")
    print(f"  Device: {device}")

    # Load model
    print("Loading ESM3...")
    model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
    model.eval()

    n_layers = len(model.transformer.blocks)
    print(f"Model has {n_layers} transformer layers")

    # Load dataset
    table = pq.read_table(parquet_path)
    n_rows = table.num_rows
    print(f"Loaded {n_rows} chains from {parquet_path}")

    # Register hooks on ALL layers
    captured_hiddens: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            captured_hiddens[layer_idx] = output.detach()
        return hook_fn

    handles = []
    for layer_idx in range(n_layers):
        h = model.transformer.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(h)

    # Per-layer accumulators: list of vectors per layer
    pos_accum = [[] for _ in range(n_layers)]  # pos_accum[layer] = list of (D,) tensors
    neg_accum = [[] for _ in range(n_layers)]
    skipped = 0

    try:
        for i in range(n_rows):
            sequence = table.column("sequence")[i].as_py()
            pos_indices = table.column(pos_col)[i].as_py()
            neg_indices = table.column(neg_col)[i].as_py()
            pdb_id = table.column("pdb_id")[i].as_py()
            chain_id = table.column("chain_id")[i].as_py()

            if not pos_indices and not neg_indices:
                continue

            if (i + 1) % 50 == 0 or i == 0:
                n_pos_so_far = len(pos_accum[0])
                n_neg_so_far = len(neg_accum[0])
                print(
                    f"  [{i + 1}/{n_rows}] {pdb_id}:{chain_id} "
                    f"(len={len(sequence)}, +{len(pos_indices)}, -{len(neg_indices)}) | "
                    f"pos={n_pos_so_far}, neg={n_neg_so_far}"
                )

            try:
                protein = ESMProtein(sequence=sequence)
                protein_tensor = model.encode(protein)

                with torch.no_grad(), torch.amp.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=device.type != "cpu",
                ):
                    model.forward(
                        sequence_tokens=protein_tensor.sequence.unsqueeze(0),
                    )

                # Extract vectors from every layer
                for layer_idx in range(n_layers):
                    hidden = captured_hiddens[layer_idx]  # (1, L, D)

                    for idx in pos_indices:
                        vec = hidden[0, idx + 1, :].float().cpu()  # +1 for BOS
                        pos_accum[layer_idx].append(vec)

                    for idx in neg_indices:
                        vec = hidden[0, idx + 1, :].float().cpu()  # +1 for BOS
                        neg_accum[layer_idx].append(vec)

                # Free GPU memory from captured hiddens
                captured_hiddens.clear()

            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"    Skipped {pdb_id}:{chain_id}: {e}")

    finally:
        for h in handles:
            h.remove()

    n_pos = len(pos_accum[0])
    n_neg = len(neg_accum[0])
    print(f"\nCollection complete:")
    print(f"  Positive vectors: {n_pos}")
    print(f"  Negative vectors: {n_neg}")
    print(f"  Skipped:          {skipped}")

    if not n_pos or not n_neg:
        print("ERROR: Not enough vectors to compute concept direction.")
        return

    # Compute per-layer concept vectors
    print("Computing per-layer concept vectors...")
    layer_results = {}
    for layer_idx in range(n_layers):
        pos_mean = torch.stack(pos_accum[layer_idx]).mean(dim=0)
        neg_mean = torch.stack(neg_accum[layer_idx]).mean(dim=0)
        concept_vec = pos_mean - neg_mean
        concept_vec_norm = concept_vec / concept_vec.norm()

        layer_results[layer_idx] = {
            "concept_vector": concept_vec,
            "concept_vector_normalized": concept_vec_norm,
            "positive_mean": pos_mean,
            "negative_mean": neg_mean,
            "norm": concept_vec.norm().item(),
        }

    # Free accumulators
    del pos_accum, neg_accum

    save_dict = {
        "concept": concept,
        "pos_col": pos_col,
        "neg_col": neg_col,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_layers": n_layers,
        "d_model": layer_results[0]["concept_vector"].shape[0],
        "layers": layer_results,
    }

    torch.save(save_dict, output_path)

    # Print summary
    print(f"\nSaved to {output_path}")
    print(f"  Concept: {concept}")
    print(f"  Dimension: {save_dict['d_model']}")
    print(f"  Samples: +{n_pos} / -{n_neg}")
    print(f"\n  Per-layer norms:")
    for layer_idx in range(n_layers):
        norm = layer_results[layer_idx]["norm"]
        bar = "█" * int(norm / 2)
        print(f"    Layer {layer_idx:2d}: {norm:7.2f}  {bar}")


if __name__ == "__main__":
    main()
