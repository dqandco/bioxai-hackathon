# BioXAI Hackathon

A pipeline for **probing protein language model interpretability** using concept vectors. It downloads protein structures from RCSB PDB, extracts contrastive residue-level features (e.g., disulfide-bonded vs free cysteines), and computes concept directions in ESM3’s residual stream that separate positive from negative examples of each biological concept.

### Pipeline overview

1. **Download** — Search RCSB PDB and download mmCIF files for disulfide bonds, protein–protein interactions, ligand binding, and PTM-containing structures.
2. **Extract features** — Parse mmCIF files with BioPython and produce per-chain parquet datasets for 7 traits: disulfide bonds, secondary structure (helix/sheet/coil), solvent accessibility (buried/exposed), PPI interface residues, ligand-binding residues, PTM sites, and disorder.
3. **Concept vectors** — Run ESM3 on sequences, collect hidden states at cysteine (or other) positions, and compute concept vectors as the mean difference between positive and negative classes (e.g., disulfide-bonded vs free cysteines).
4. **Validation** — Evaluate concept vectors on held-out chains using projection scores, linear probe accuracy, and ROC AUC.

### Local scripts (optional)

- `extract_concept_vector.py` — Extract per-layer concept vectors for any supported concept (disulfide, ss_helix, sasa, ppi, binding, ptm, disorder).
- `validate_concept_vector.py` — Validate concept vectors on held-out data and report per-layer metrics.

---

## Modal Cloud Compute

This project uses [Modal](https://modal.com) for cloud compute. Data is stored in a Modal Volume (`bioxai-data`).

### Setup

1. Install dependencies: `uv sync`
2. Authenticate: `modal setup`
3. **Hugging Face (for ESM3):** Create a Modal secret with your HF token:
   ```bash
   modal secret create huggingface HF_TOKEN=hf_your_token_here
   ```
   Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Run on Modal

```bash
# Full pipeline (download → extract features → concept vector)
modal run modal_app.py::run_full_pipeline

# Individual steps
modal run modal_app.py::download_only
modal run modal_app.py::extract_only
modal run modal_app.py::concept_vector_only
```

### Sync data locally

To copy results from the Modal volume to your local `data/` directory:

```bash
modal volume get bioxai-data /data ./data
```
