"""
Validate concept vectors by re-running ESM3 inference on a held-out subset
and measuring how well the concept vector separates positive from negative
residues at each layer.

Usage:
  python validate_concept_vector.py <concept> [--n-samples 100] [--seed 42]

Reports per-layer:
  - Mean projection for positive and negative classes
  - Separation (difference of means along concept direction)
  - Linear probe accuracy (threshold at midpoint of class means)
  - ROC AUC
"""

import argparse
import random

import torch
import pyarrow.parquet as pq
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein

from concepts import CONCEPTS, DATA_DIR


def compute_roc_auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Compute ROC AUC from positive and negative projection scores."""
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = pos_scores + neg_scores

    # Sort by score descending
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])

    tp = 0
    fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5

    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        # Trapezoidal rule
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr

    return auc


def main():
    parser = argparse.ArgumentParser(
        description="Validate concept vectors against held-out data."
    )
    parser.add_argument("concept", choices=sorted(CONCEPTS.keys()))
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of chains to use for validation (default: 100).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    concept = args.concept
    parquet_file, pos_col, neg_col = CONCEPTS[concept]
    parquet_path = DATA_DIR / parquet_file
    vectors_path = DATA_DIR / f"{concept}_concept_vectors.pt"

    # Load concept vectors
    print(f"Loading concept vectors from {vectors_path}...")
    saved = torch.load(vectors_path, weights_only=False)
    n_layers = saved["n_layers"]
    d_model = saved["d_model"]
    layer_data = saved["layers"]

    # Get normalized concept vectors per layer
    concept_dirs = {}
    for layer_idx in range(n_layers):
        concept_dirs[layer_idx] = layer_data[layer_idx]["concept_vector_normalized"]

    # Load dataset and sample a held-out subset
    table = pq.read_table(parquet_path)
    n_rows = table.num_rows

    # Use chains that have BOTH positive and negative indices for cleaner eval
    valid_rows = []
    for i in range(n_rows):
        pos_indices = table.column(pos_col)[i].as_py()
        neg_indices = table.column(neg_col)[i].as_py()
        if pos_indices and neg_indices:
            valid_rows.append(i)

    random.seed(args.seed)
    random.shuffle(valid_rows)
    sample_rows = valid_rows[:args.n_samples]

    print(f"Dataset: {n_rows} total chains, {len(valid_rows)} with both classes")
    print(f"Validation subset: {len(sample_rows)} chains")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ESM3 on {device}...")
    model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
    model.eval()

    # Hook all layers
    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook_fn(module, input, output):
            captured[idx] = output.detach()
        return hook_fn

    handles = [
        model.transformer.blocks[i].register_forward_hook(make_hook(i))
        for i in range(n_layers)
    ]

    # Per-layer projection scores
    pos_projections: dict[int, list[float]] = {i: [] for i in range(n_layers)}
    neg_projections: dict[int, list[float]] = {i: [] for i in range(n_layers)}
    skipped = 0

    try:
        for count, row_idx in enumerate(sample_rows):
            sequence = table.column("sequence")[row_idx].as_py()
            pos_indices = table.column(pos_col)[row_idx].as_py()
            neg_indices = table.column(neg_col)[row_idx].as_py()
            pdb_id = table.column("pdb_id")[row_idx].as_py()
            chain_id = table.column("chain_id")[row_idx].as_py()

            if (count + 1) % 20 == 0 or count == 0:
                print(f"  [{count + 1}/{len(sample_rows)}] {pdb_id}:{chain_id}")

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

                for layer_idx in range(n_layers):
                    hidden = captured[layer_idx]  # (1, L, D)
                    direction = concept_dirs[layer_idx].to(hidden.device)

                    for idx in pos_indices:
                        vec = hidden[0, idx + 1, :].float()
                        vec = vec / vec.norm()
                        proj = (vec @ direction).item()
                        pos_projections[layer_idx].append(proj)

                    for idx in neg_indices:
                        vec = hidden[0, idx + 1, :].float()
                        vec = vec / vec.norm()
                        proj = (vec @ direction).item()
                        neg_projections[layer_idx].append(proj)

                captured.clear()

            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"    Skipped {pdb_id}:{chain_id}: {e}")

    finally:
        for h in handles:
            h.remove()

    n_pos = len(pos_projections[0])
    n_neg = len(neg_projections[0])
    print(f"\nValidation complete: {n_pos} positive, {n_neg} negative vectors, {skipped} skipped")

    if not n_pos or not n_neg:
        print("ERROR: Not enough samples.")
        return

    # Compute metrics per layer
    print(f"\n{'Layer':>5}  {'Pos Mean':>9}  {'Neg Mean':>9}  {'Sep':>8}  {'Acc':>6}  {'AUC':>6}")
    print("-" * 55)

    best_auc = 0.0
    best_layer = 0

    for layer_idx in range(n_layers):
        pos_scores = pos_projections[layer_idx]
        neg_scores = neg_projections[layer_idx]

        pos_mean = sum(pos_scores) / len(pos_scores)
        neg_mean = sum(neg_scores) / len(neg_scores)
        separation = pos_mean - neg_mean

        # Accuracy with threshold at midpoint
        threshold = (pos_mean + neg_mean) / 2
        correct = sum(1 for s in pos_scores if s > threshold) + \
                  sum(1 for s in neg_scores if s <= threshold)
        accuracy = correct / (len(pos_scores) + len(neg_scores))

        auc = compute_roc_auc(pos_scores, neg_scores)

        if auc > best_auc:
            best_auc = auc
            best_layer = layer_idx

        print(f"{layer_idx:5d}  {pos_mean:9.2f}  {neg_mean:9.2f}  {separation:8.2f}  {accuracy:5.1%}  {auc:5.3f}")

    print(f"\nBest layer: {best_layer} (AUC={best_auc:.3f})")


if __name__ == "__main__":
    main()
