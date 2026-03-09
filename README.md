# ESPRIT

**Extracting Spatial Protein Representations via Interpretability Tensors**

ESPRIT probes the internal representations of [ESM3](https://www.evolutionaryscale.ai/blog/esm3-release), a protein language model, by computing *concept vectors* — directions in hidden-state space that correspond to known biological properties of amino acid residues. Given a new protein sequence, ESPRIT runs ESM3 inference, projects each residue's hidden state onto these concept directions, and renders the results as an interactive 3D heatmap on the protein structure.

---

## Table of Contents

- [Overview](#overview)
- [Biological Concepts](#biological-concepts)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Data Pipeline](#running-the-data-pipeline)
  - [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)

---

## Overview

ESPRIT follows a three-stage workflow:

1. **Data collection** — Download ~2,750 protein structures from the RCSB Protein Data Bank (PDB) and extract per-residue biological features (secondary structure, solvent accessibility, binding sites, etc.) into Parquet datasets.
2. **Concept vector extraction** — Run ESM3 forward passes with hooks on every transformer layer, collect hidden states at labeled residue positions, and compute a concept direction per layer as the mean-difference between positive and negative classes.
3. **Interactive inference** — Serve a FastAPI backend that accepts arbitrary protein sequences, runs ESM3, projects residue hidden states onto concept vectors via cosine similarity, and streams the results to a React frontend that renders them on a 3Dmol.js structure viewer.

## Biological Concepts

ESPRIT ships with nine concept probes:

| Concept | Positive class | Negative class |
| --- | --- | --- |
| **Disulfide bonds** | Disulfide-bonded cysteines | Free cysteines |
| **Helix** | Helical residues | Coil residues |
| **Sheet** | Sheet residues | Coil residues |
| **Helix vs Sheet** | Helical residues | Sheet residues |
| **Solvent accessibility** | Buried residues (low SASA) | Exposed residues (high SASA) |
| **Protein-protein interaction** | Interface residues | Non-interface residues |
| **Ligand/metal binding** | Binding-site residues | Non-binding residues |
| **Post-translational modifications** | Modified residues | Unmodified residues |
| **Disorder** | Disordered residues | Ordered residues |

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  React + 3Dmol.js Frontend                             │
│  Interactive 3D protein viewer with concept heatmaps   │
└───────────────────┬────────────────────────────────────┘
                    │ REST
┌───────────────────▼────────────────────────────────────┐
│  FastAPI Backend (api.py)                              │
│  Serves concept vectors & runs live ESM3 inference     │
└───────────────────┬────────────────────────────────────┘
                    │
┌───────────────────▼────────────────────────────────────┐
│  ESM3 Inference Engine (inference.py)                  │
│  Forward hooks on all transformer layers               │
│  Cosine similarity projection onto concept directions  │
└───────────────────┬────────────────────────────────────┘
                    │
┌───────────────────▼────────────────────────────────────┐
│  Data Pipeline (Modal / local)                         │
│  download_structures → extract_features →              │
│  extract_concept_vector → validate_concept_vector      │
└────────────────────────────────────────────────────────┘
```

## Tech Stack

**Backend** — Python 3.12, FastAPI, PyTorch, ESM3, BioPython, PyArrow, Modal

**Frontend** — React 19, TypeScript, Vite, Tailwind CSS 4, 3Dmol.js

**Infrastructure** — Modal (GPU cloud compute), Hugging Face (ESM3 model weights), RCSB PDB (protein structures)

## Project Structure

```
├── api.py                        # FastAPI server
├── inference.py                  # ESM3 engine with forward hooks
├── extract_features.py           # BioPython feature extraction from mmCIF
├── extract_concept_vector.py     # Per-layer concept vector computation
├── validate_concept_vector.py    # ROC AUC & linear probe evaluation
├── download_structures.py        # RCSB PDB search & download
├── modal_app.py                  # Modal cloud orchestration
├── concepts.py                   # Concept definitions & utilities
├── main.py                       # CLI entry point
├── pyproject.toml                # Python dependencies
│
├── web/                          # Frontend
│   ├── src/
│   │   ├── App.tsx               # Main application component
│   │   ├── ProteinViewer.tsx     # 3Dmol.js 3D viewer
│   │   └── pdbParser.ts         # PDB file parser
│   ├── package.json
│   └── vite.config.ts
│
└── data/                         # Generated artifacts (not committed)
    ├── cif_files/                # Downloaded mmCIF structures
    ├── *_features.parquet        # Per-concept feature tables
    ├── *_concept_vectors.pt      # Per-layer concept direction tensors
    └── download_manifest.json    # PDB search results log
```

## Getting Started

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Node.js 18+ (for the frontend)
- A [Hugging Face](https://huggingface.co/) account with access to the ESM3 model
- A [Modal](https://modal.com/) account (for cloud pipeline; optional for local-only use)

### Installation

```bash
# Clone the repo
git clone <repo-url> && cd bioxai-hackathon

# Install Python dependencies
uv sync            # or: pip install -e .

# Install frontend dependencies
cd web
npm install        # or: bun install
cd ..
```

**Modal setup** (for cloud pipeline):

```bash
modal setup
modal secret create huggingface HF_TOKEN=hf_<your_token>
```

### Running the Data Pipeline

**Full pipeline on Modal (recommended):**

```bash
modal run modal_app.py::run_full_pipeline
```

Individual stages can also be run separately:

```bash
modal run modal_app.py::download_only          # Download PDB structures
modal run modal_app.py::extract_only           # Extract features to Parquet
modal run modal_app.py::concept_vector_only    # Compute concept vectors
```

**Local feature extraction (no GPU required):**

```bash
python extract_features.py                       # Extract all features
python extract_concept_vector.py <concept>       # Compute vectors for a concept
python validate_concept_vector.py <concept>      # Evaluate with ROC AUC
```

Where `<concept>` is one of: `disulfide`, `ss_helix`, `ss_sheet`, `ss_helix_sheet`, `sasa`, `ppi`, `binding`, `ptm`, `disorder`.

**Sync data from Modal to local:**

```bash
modal volume get bioxai-data /data ./data
```

### Running the Application

Start the backend and frontend in separate terminals:

```bash
# Terminal 1 — API server (port 8000)
python api.py

# Terminal 2 — Frontend dev server
cd web
npm run dev
```

Open the URL printed by Vite (typically `http://localhost:5173`). Paste a protein sequence, select a concept and layer, and view the cosine-similarity heatmap on the 3D structure.

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Model readiness status |
| `GET` | `/concepts` | List available concepts with metadata |
| `GET` | `/concepts/{name}/vectors` | Per-layer concept vector norms |
| `GET` | `/concepts/{name}/features` | Preview of the feature Parquet |
| `POST` | `/inference/project` | Run ESM3 inference and project onto concept vectors |

**`POST /inference/project`** accepts a JSON body:

```json
{
  "sequence": "MKTLLILAVL...",
  "concept": "ss_helix",
  "layer": 12
}
```

Returns per-residue cosine similarity scores between hidden states and the selected concept direction.

## How It Works

### Concept Vector Extraction

For each concept (e.g., "helix"), the pipeline:

1. Loads labeled residue indices from the feature Parquet (positive = helix, negative = coil).
2. Runs ESM3 on each chain, capturing hidden states at every transformer layer via forward hooks.
3. Collects hidden-state vectors at positive and negative residue positions.
4. Computes the concept direction per layer: **v = mean(h_pos) - mean(h_neg)**.
5. L2-normalizes the direction for stable cosine projections.

### Live Inference

Given a new sequence:

1. Tokenize and run ESM3 with hooks active.
2. For each residue at the selected layer, compute **cos(h_residue, v_concept)**.
3. Return the similarity vector to the frontend, which maps it to a color scale on the 3D structure.

### Validation

Each concept vector is evaluated on held-out chains using:

- **ROC AUC** — treating cosine similarity as a binary classifier score.
- **Linear probe accuracy** — fitting a logistic regression on the projected scalar.

These metrics measure how well the concept direction separates positive from negative residues in unseen proteins.
