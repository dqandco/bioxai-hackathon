"""FastAPI application serving protein concept vector data and live ESM3 inference."""

import asyncio
import re
from contextlib import asynccontextmanager

import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from concepts import CONCEPTS, DATA_DIR, list_computed_concepts, load_concept_vectors
from inference import ESM3Engine, MAX_SEQUENCE_LENGTH

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
engine = ESM3Engine()
_inference_lock = asyncio.Lock()
_parquet_cache: dict[str, pq.ParquetFile] = {}

AMINO_ACID_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_parquet(concept_name: str):
    """Return a cached pyarrow Table for the concept's parquet file."""
    if concept_name not in _parquet_cache:
        parquet_file, _, _ = CONCEPTS[concept_name]
        path = DATA_DIR / parquet_file
        if not path.exists():
            raise HTTPException(404, f"Feature file not found: {parquet_file}")
        _parquet_cache[concept_name] = pq.read_table(path)
    return _parquet_cache[concept_name]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    engine.load_concept_vectors()
    yield


app = FastAPI(title="BioXAI Concept Vectors API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": engine.model is not None,
        "n_concepts_loaded": len(engine.concept_vectors),
        "device": str(engine.device),
    }


@app.get("/concepts")
def get_concepts():
    computed = set(list_computed_concepts())
    items = []
    for name, (_, pos_col, neg_col) in CONCEPTS.items():
        items.append({
            "name": name,
            "positive_class": pos_col,
            "negative_class": neg_col,
            "has_vectors": name in computed,
        })
    return {"concepts": items}


@app.get("/concepts/{name}/vectors")
def get_concept_vectors(name: str):
    if name not in CONCEPTS:
        raise HTTPException(404, f"Unknown concept: {name}")

    path = DATA_DIR / f"{name}_concept_vectors.pt"
    if not path.exists():
        raise HTTPException(404, f"Concept vectors not computed for: {name}")

    saved = load_concept_vectors(name)
    n_layers = saved["n_layers"]
    layers = []
    for layer_idx in range(n_layers):
        layers.append({
            "layer": layer_idx,
            "norm": saved["layers"][layer_idx]["norm"],
        })

    return {
        "concept": name,
        "n_positive": saved["n_positive"],
        "n_negative": saved["n_negative"],
        "n_layers": n_layers,
        "d_model": saved["d_model"],
        "layers": layers,
    }


@app.get("/concepts/{name}/features")
def get_concept_features(
    name: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    if name not in CONCEPTS:
        raise HTTPException(404, f"Unknown concept: {name}")

    table = _get_parquet(name)
    total_rows = table.num_rows

    end = min(offset + limit, total_rows)
    sliced = table.slice(offset, end - offset)
    rows = sliced.to_pydict()
    # Convert columnar dict to list of row dicts
    row_list = [
        {col: rows[col][i] for col in rows}
        for i in range(sliced.num_rows)
    ]

    return {
        "concept": name,
        "total_rows": total_rows,
        "offset": offset,
        "limit": limit,
        "rows": row_list,
    }


class InferenceRequest(BaseModel):
    sequence: str
    concepts: list[str] | None = None


@app.post("/inference/project")
async def inference_project(req: InferenceRequest):
    if engine.model is None:
        raise HTTPException(503, "Model not loaded")

    sequence = req.sequence.strip().upper()
    if not sequence:
        raise HTTPException(400, "Empty sequence")
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        raise HTTPException(
            400,
            f"Sequence length {len(sequence)} exceeds maximum {MAX_SEQUENCE_LENGTH}",
        )
    if not AMINO_ACID_PATTERN.match(sequence):
        raise HTTPException(400, "Sequence contains invalid amino acid characters")

    async with _inference_lock:
        try:
            result = await asyncio.to_thread(
                engine.run_inference, sequence, req.concepts
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000)
