# BioXAI Hackathon

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
