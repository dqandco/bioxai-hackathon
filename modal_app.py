"""
Modal cloud compute integration for the BioXAI hackathon pipeline.

Run with:
  modal run modal_app.py::download_data
  modal run modal_app.py::extract_features
  modal run modal_app.py::extract_concept_vector

Or run the full pipeline:
  modal run modal_app.py::run_full_pipeline
"""

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App and image setup
# ---------------------------------------------------------------------------

DATA_PATH = "/data"
CIF_PATH = f"{DATA_PATH}/cif_files"

# Volume for persistent data (CIF files, parquets, concept vector)
volume = modal.Volume.from_name("bioxai-data", create_if_missing=True)

# Shared image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "biopython>=1.86",
        "esm>=3.2.1.post1",
        "pyarrow>=23.0.1",
        "torch>=2.2.0",
        "torchvision",
    )
    .add_local_dir(
        ".",
        remote_path="/root",
        ignore=[".venv", "data", "__pycache__", ".git", "*.pyc", ".python-version"],
    )
)

app = modal.App("bioxai-hackathon", image=image)


# ---------------------------------------------------------------------------
# Download data
# ---------------------------------------------------------------------------


@app.function(
    volumes={DATA_PATH: volume},
    timeout=3600,
    network_file_systems={},
)
def download_data() -> dict:
    """Search RCSB PDB and download mmCIF files to the Modal volume."""
    import json
    import sys

    sys.path.insert(0, "/root")
    from download_disulfide_data import (
        TARGETS,
        download_cif,
        search_disulfide_proteins,
        search_ligand_proteins,
        search_ppi_proteins,
        search_ptm_proteins,
    )

    Path(CIF_PATH).mkdir(parents=True, exist_ok=True)

    print("Searching RCSB PDB for structures across all categories...")
    search_fns = {
        "disulfide": search_disulfide_proteins,
        "ppi": search_ppi_proteins,
        "ligand_binding": search_ligand_proteins,
        "ptm": search_ptm_proteins,
    }

    category_ids: dict = {}
    all_ids: set = set()
    for name, fn in search_fns.items():
        ids = fn(TARGETS[name])
        category_ids[name] = ids
        all_ids.update(ids)

    print(f"\nTotal unique PDB IDs across all categories: {len(all_ids)}")

    manifest_path = Path(DATA_PATH) / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({k: v for k, v in category_ids.items()}, f, indent=2)
    print(f"Saved category manifest to {manifest_path}")

    all_ids_sorted = sorted(all_ids)
    downloaded = 0
    failed = 0
    for i, pdb_id in enumerate(all_ids_sorted):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Downloading {i + 1}/{len(all_ids_sorted)}...")
        result = download_cif(pdb_id, Path(CIF_PATH))
        if result is not None:
            downloaded += 1
        else:
            failed += 1

    volume.commit()
    print(f"\nDone! Downloaded: {downloaded}, Failed: {failed}")
    return {"downloaded": downloaded, "failed": failed, "total": len(all_ids_sorted)}


# ---------------------------------------------------------------------------
# Extract features (parallel per-file)
# ---------------------------------------------------------------------------


@app.function(
    volumes={DATA_PATH: volume},
    timeout=600,
    cpu=2,
)
def process_one_cif(cif_path_str: str) -> dict[str, list]:
    """Process a single mmCIF file. Volume must be mounted to read the file."""
    """Process a single mmCIF file and return records for all traits."""
    import sys

    sys.path.insert(0, "/root")
    from extract_features import process_single_cif

    return process_single_cif(Path(cif_path_str))


@app.function(
    volumes={DATA_PATH: volume},
    timeout=7200,
    cpu=4,
)
def extract_features() -> dict:
    """Extract contrastive features from all mmCIF files in parallel."""
    import sys

    sys.path.insert(0, "/root")
    from extract_features import _save_parquet

    cif_files = sorted(Path(CIF_PATH).glob("*.cif"))
    print(f"Found {len(cif_files)} mmCIF files to process.\n")

    if not cif_files:
        print("No CIF files found. Run download_data first.")
        return {"processed": 0}

    # Process files in parallel
    cif_paths = [str(p) for p in cif_files]
    results = list(process_one_cif.starmap([(p,) for p in cif_paths]))

    # Merge results
    disulfide_records = []
    ss_records = []
    sasa_records = []
    ppi_records = []
    binding_records = []
    ptm_records = []
    disorder_records = []

    for r in results:
        disulfide_records.extend(r["disulfide"])
        ss_records.extend(r["ss"])
        sasa_records.extend(r["sasa"])
        ppi_records.extend(r["ppi"])
        binding_records.extend(r["binding"])
        ptm_records.extend(r["ptm"])
        disorder_records.extend(r["disorder"])

    data_dir = Path(DATA_PATH)
    _save_parquet(
        disulfide_records,
        [
            "pdb_id",
            "chain_id",
            "sequence",
            "seq_length",
            "disulfide_cys_indices",
            "free_cys_indices",
            "n_disulfide_cys",
            "n_free_cys",
        ],
        data_dir / "disulfide_features.parquet",
    )
    _save_parquet(
        ss_records,
        [
            "pdb_id",
            "chain_id",
            "sequence",
            "seq_length",
            "helix_indices",
            "sheet_indices",
            "coil_indices",
            "n_helix",
            "n_sheet",
            "n_coil",
        ],
        data_dir / "ss_features.parquet",
    )
    _save_parquet(
        sasa_records,
        [
            "pdb_id",
            "chain_id",
            "sequence",
            "seq_length",
            "buried_indices",
            "exposed_indices",
            "n_buried",
            "n_exposed",
        ],
        data_dir / "sasa_features.parquet",
    )
    _save_parquet(
        ppi_records,
        [
            "pdb_id",
            "chain_id",
            "sequence",
            "seq_length",
            "interface_indices",
            "non_interface_indices",
            "n_interface",
            "n_non_interface",
        ],
        data_dir / "ppi_features.parquet",
    )
    _save_parquet(
        binding_records,
        [
            "pdb_id",
            "chain_id",
            "sequence",
            "seq_length",
            "binding_indices",
            "non_binding_indices",
            "n_binding",
            "n_non_binding",
        ],
        data_dir / "binding_features.parquet",
    )
    _save_parquet(
        ptm_records,
        [
            "pdb_id",
            "chain_id",
            "sequence",
            "seq_length",
            "ptm_indices",
            "non_ptm_indices",
            "n_ptm",
            "n_non_ptm",
        ],
        data_dir / "ptm_features.parquet",
    )
    _save_parquet(
        disorder_records,
        [
            "pdb_id",
            "chain_id",
            "sequence",
            "seq_length",
            "disordered_indices",
            "ordered_indices",
            "n_disordered",
            "n_ordered",
        ],
        data_dir / "disorder_features.parquet",
    )

    volume.commit()
    print("\nDone!")
    return {
        "processed": len(cif_files),
        "disulfide": len(disulfide_records),
        "ss": len(ss_records),
        "sasa": len(sasa_records),
        "ppi": len(ppi_records),
        "binding": len(binding_records),
        "ptm": len(ptm_records),
        "disorder": len(disorder_records),
    }


# ---------------------------------------------------------------------------
# Extract concept vector (GPU)
# ---------------------------------------------------------------------------


@app.function(
    volumes={DATA_PATH: volume},
    gpu="T4",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def extract_concept_vector() -> dict:
    """Extract disulfide concept vector from ESM3 on GPU."""
    import sys

    sys.path.insert(0, "/root")
    import torch
    import pyarrow.parquet as pq
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein

    device = torch.device("cuda")
    parquet_path = Path(DATA_PATH) / "disulfide_features.parquet"
    output_path = Path(DATA_PATH) / "disulfide_concept_vector.pt"

    print("Loading ESM3...")
    model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
    model.eval()

    n_layers = len(model.transformer.blocks)
    hook_layer_idx = n_layers // 2
    print(f"Model has {n_layers} transformer layers, hooking layer {hook_layer_idx}")

    table = pq.read_table(parquet_path)
    n_rows = table.num_rows
    print(f"Loaded {n_rows} chains from {parquet_path}")

    captured_hidden = {}

    def hook_fn(module, input, output):
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
                protein = ESMProtein(sequence=sequence)
                protein_tensor = model.encode(protein)

                with torch.no_grad(), torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    model.forward(
                        sequence_tokens=protein_tensor.sequence.unsqueeze(0),
                    )

                hidden = captured_hidden["states"]
                for idx in disulfide_indices:
                    token_idx = idx + 1
                    vec = hidden[0, token_idx, :].float().cpu()
                    positive_vectors.append(vec)

                for idx in free_indices:
                    token_idx = idx + 1
                    vec = hidden[0, token_idx, :].float().cpu()
                    negative_vectors.append(vec)

            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"    Skipped {pdb_id}:{chain_id}: {e}")

    finally:
        handle.remove()

    print("\nCollection complete:")
    print(f"  Positive vectors (disulfide): {len(positive_vectors)}")
    print(f"  Negative vectors (free):      {len(negative_vectors)}")
    print(f"  Skipped:                       {skipped}")

    if not positive_vectors or not negative_vectors:
        raise ValueError("Not enough vectors to compute concept direction.")

    pos_tensor = torch.stack(positive_vectors)
    neg_tensor = torch.stack(negative_vectors)
    pos_mean = pos_tensor.mean(dim=0)
    neg_mean = neg_tensor.mean(dim=0)
    concept_vector = pos_mean - neg_mean
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

    torch.save(save_dict, output_path)
    volume.commit()

    print(f"\nConcept vector saved to {output_path}")
    print(f"  Dimension: {concept_vector.shape[0]}")
    print(f"  Norm (raw): {concept_vector.norm().item():.4f}")

    return {
        "output_path": str(output_path),
        "d_model": concept_vector.shape[0],
        "n_positive": len(positive_vectors),
        "n_negative": len(negative_vectors),
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def run_full_pipeline():
    """Run the full pipeline: download -> extract features -> concept vector."""
    print("=== Step 1: Download data ===")
    download_data.remote()
    print("\n=== Step 2: Extract features ===")
    extract_features.remote()
    print("\n=== Step 3: Extract concept vector ===")
    extract_concept_vector.remote()
    print("\n=== Pipeline complete ===")


@app.local_entrypoint()
def download_only():
    """Download PDB data only."""
    download_data.remote()


@app.local_entrypoint()
def extract_only():
    """Extract features only (requires CIF files in volume)."""
    extract_features.remote()


@app.local_entrypoint()
def concept_vector_only():
    """Extract concept vector only (requires disulfide_features.parquet in volume)."""
    extract_concept_vector.remote()
