"""
Extract the disulfide bond concept vector from ESM3's residual stream.

1. Load ESM3 and register a forward hook on a middle transformer layer
2. Run each protein sequence through the model, capturing hidden states
3. Collect hidden states at disulfide-bonded cysteine positions (positive)
   and free cysteine positions (negative)
4. Compute the concept vector as mean(positive) - mean(negative)
5. Save as a .pt file

ESM3 architecture (from source inspection):
  - model.encoder: EncodeInputs — fuses sequence, structure, function, SS8,
    SASA, and residue annotation embeddings by summing them into a single
    vector per token position
  - model.transformer: TransformerStack containing .blocks (nn.ModuleList
    of UnifiedTransformerBlock). The fused representation flows through
    these blocks as a unified residual stream.
  - model.output_heads: projects final hidden states to per-track logits

The hook is placed on a middle transformer block (layer n_layers//2).
After the encoder sums all modality embeddings, each transformer block
operates on the unified residual stream, so the hidden state at any
token index i corresponds exactly to residue i (offset by +1 for the
BOS token prepended during tokenization).
"""

from pathlib import Path

import torch
import pyarrow.parquet as pq
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein


DATA_DIR = Path("data")
PARQUET_PATH = DATA_DIR / "disulfide_features.parquet"
OUTPUT_PATH = DATA_DIR / "disulfide_concept_vector.pt"

# Which transformer layer to hook (will be set relative to model depth)
LAYER_FRACTION = 0.5  # middle layer


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load model
    print("Loading ESM3...")
    model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
    model.eval()

    # Determine hook layer
    n_layers = len(model.transformer.blocks)
    hook_layer_idx = n_layers // 2
    print(f"Model has {n_layers} transformer layers, hooking layer {hook_layer_idx}")

    # Load dataset
    table = pq.read_table(PARQUET_PATH)
    n_rows = table.num_rows
    print(f"Loaded {n_rows} chains from {PARQUET_PATH}")

    # Storage for hook captures
    captured_hidden = {}

    def hook_fn(module, input, output):
        # UnifiedTransformerBlock returns a tensor of shape (B, L, D)
        captured_hidden["states"] = output.detach()

    handle = model.transformer.blocks[hook_layer_idx].register_forward_hook(hook_fn)

    positive_vectors = []
    negative_vectors = []
    skipped = 0

    try:
        for i in range(n_rows):
            sequence = table.column("sequence")[i].as_py()
            disulfide_indices = table.column("disulfide_cys_indices")[i].as_py()
            free_indices = table.column("free_cys_indices")[i].as_py()
            pdb_id = table.column("pdb_id")[i].as_py()
            chain_id = table.column("chain_id")[i].as_py()

            if not disulfide_indices and not free_indices:
                continue

            if (i + 1) % 50 == 0 or i == 0:
                print(
                    f"  [{i + 1}/{n_rows}] {pdb_id}:{chain_id} "
                    f"(len={len(sequence)}, +{len(disulfide_indices)}, -{len(free_indices)}) | "
                    f"pos={len(positive_vectors)}, neg={len(negative_vectors)}"
                )

            try:
                # Encode sequence -> tokenize with BOS/EOS
                protein = ESMProtein(sequence=sequence)
                protein_tensor = model.encode(protein)

                # Forward pass (triggers hook)
                with torch.no_grad(), torch.amp.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type != "cpu"
                ):
                    model.forward(
                        sequence_tokens=protein_tensor.sequence.unsqueeze(0),
                    )

                hidden = captured_hidden["states"]  # (1, L_with_special, D)

                # ESM3 tokenization prepends BOS, so residue index i in the
                # original sequence maps to token index i+1 in the model output.
                for idx in disulfide_indices:
                    token_idx = idx + 1  # offset for BOS
                    vec = hidden[0, token_idx, :].float().cpu()
                    positive_vectors.append(vec)

                for idx in free_indices:
                    token_idx = idx + 1  # offset for BOS
                    vec = hidden[0, token_idx, :].float().cpu()
                    negative_vectors.append(vec)

            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"    Skipped {pdb_id}:{chain_id}: {e}")

    finally:
        handle.remove()

    print(f"\nCollection complete:")
    print(f"  Positive vectors (disulfide): {len(positive_vectors)}")
    print(f"  Negative vectors (free):      {len(negative_vectors)}")
    print(f"  Skipped:                       {skipped}")

    if not positive_vectors or not negative_vectors:
        print("ERROR: Not enough vectors to compute concept direction.")
        return

    # Stack and compute means
    pos_tensor = torch.stack(positive_vectors)  # (N_pos, D)
    neg_tensor = torch.stack(negative_vectors)  # (N_neg, D)

    pos_mean = pos_tensor.mean(dim=0)  # (D,)
    neg_mean = neg_tensor.mean(dim=0)  # (D,)

    concept_vector = pos_mean - neg_mean  # (D,)

    # Normalize for convenience (save both raw and normalized)
    concept_vector_normalized = concept_vector / concept_vector.norm()

    save_dict = {
        "concept_vector": concept_vector,
        "concept_vector_normalized": concept_vector_normalized,
        "positive_mean": pos_mean,
        "negative_mean": neg_mean,
        "n_positive": len(positive_vectors),
        "n_negative": len(negative_vectors),
        "hook_layer": hook_layer_idx,
        "n_layers": n_layers,
        "d_model": concept_vector.shape[0],
    }

    torch.save(save_dict, OUTPUT_PATH)

    print(f"\nConcept vector saved to {OUTPUT_PATH}")
    print(f"  Dimension: {concept_vector.shape[0]}")
    print(f"  Norm (raw): {concept_vector.norm().item():.4f}")
    print(f"  Hook layer: {hook_layer_idx}/{n_layers}")


if __name__ == "__main__":
    main()
