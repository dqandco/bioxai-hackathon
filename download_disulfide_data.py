"""
Download protein structures from RCSB PDB for contrastive feature extraction.

Downloads structures for multiple trait categories:
  - Disulfide bonds (existing)
  - Protein-protein interaction sites (multi-chain complexes)
  - Ligand/metal binding sites (structures with non-polymer entities)
  - PTM sites (structures with modified residues)
  - Secondary structure, solvent accessibility, disorder are extracted
    from all structures (no special search needed)
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path


DATA_DIR = Path("data")
CIF_DIR = DATA_DIR / "cif_files"

# Target counts per category (will be deduplicated)
TARGETS = {
    "disulfide": 750,
    "ppi": 500,
    "ligand_binding": 500,
    "ptm": 500,
}


def _run_search(query: dict, label: str) -> list[str]:
    """Execute an RCSB search query and return PDB IDs."""
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    data = json.dumps(query).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req) as resp:
            results = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"  WARNING: {label} search failed ({e.code}): {body[:200]}")
        return []

    pdb_ids = [hit["identifier"] for hit in results.get("result_set", [])]
    print(f"  {label}: {len(pdb_ids)} results (total_count={results.get('total_count', '?')})")
    return pdb_ids


def _base_protein_filter() -> list[dict]:
    """Common filters: protein polymer + resolution <= 2.5 Å."""
    return [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match",
                "value": "Protein",
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": 2.5,
            },
        },
    ]


def search_disulfide_proteins(target: int) -> list[str]:
    return _run_search({
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.disulfide_bond_count",
                        "operator": "greater",
                        "value": 0,
                    },
                },
                *_base_protein_filter(),
            ],
        },
        "request_options": {"paginate": {"start": 0, "rows": target}},
        "return_type": "entry",
    }, "disulfide")


def search_ppi_proteins(target: int) -> list[str]:
    """Multi-chain protein complexes (>= 2 protein entities)."""
    return _run_search({
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "value": 2,
                    },
                },
                *_base_protein_filter(),
            ],
        },
        "request_options": {"paginate": {"start": 0, "rows": target}},
        "return_type": "entry",
    }, "ppi")


def search_ligand_proteins(target: int) -> list[str]:
    """Structures with non-polymer entities (ligands, metals, cofactors)."""
    return _run_search({
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater_or_equal",
                        "value": 1,
                    },
                },
                *_base_protein_filter(),
            ],
        },
        "request_options": {"paginate": {"start": 0, "rows": target}},
        "return_type": "entry",
    }, "ligand_binding")


def search_ptm_proteins(target: int) -> list[str]:
    """Structures with post-translational modifications.

    Searches for structures containing common modified amino acids:
    SEP (phosphoserine), TPO (phosphothreonine), PTR (phosphotyrosine),
    MLY (dimethyllysine), HYP (hydroxyproline), CSO (s-hydroxycysteine).
    """
    return _run_search({
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_chem_comp_container_identifiers.comp_id",
                        "operator": "in",
                        "value": ["SEP", "TPO", "PTR", "MLY", "HYP", "CSO"],
                    },
                },
                *_base_protein_filter(),
            ],
        },
        "request_options": {"paginate": {"start": 0, "rows": target}},
        "return_type": "entry",
    }, "ptm")


def download_cif(pdb_id: str, out_dir: Path) -> Path | None:
    """Download an mmCIF file from RCSB PDB."""
    out_path = out_dir / f"{pdb_id.lower()}.cif"
    if out_path.exists():
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, out_path)
            return out_path
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  {pdb_id}: not found (404), skipping.")
                return None
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"  {pdb_id}: download failed after retries ({e}).")
                return None
        except Exception as e:
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"  {pdb_id}: download failed ({e}).")
                return None


def main():
    CIF_DIR.mkdir(parents=True, exist_ok=True)

    print("Searching RCSB PDB for structures across all categories...")
    search_fns = {
        "disulfide": search_disulfide_proteins,
        "ppi": search_ppi_proteins,
        "ligand_binding": search_ligand_proteins,
        "ptm": search_ptm_proteins,
    }

    category_ids: dict[str, list[str]] = {}
    all_ids: set[str] = set()
    for name, fn in search_fns.items():
        ids = fn(TARGETS[name])
        category_ids[name] = ids
        all_ids.update(ids)

    print(f"\nTotal unique PDB IDs across all categories: {len(all_ids)}")

    # Save category membership for reference
    manifest_path = DATA_DIR / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({k: v for k, v in category_ids.items()}, f, indent=2)
    print(f"Saved category manifest to {manifest_path}")

    # Download all
    all_ids_sorted = sorted(all_ids)
    downloaded = 0
    failed = 0
    for i, pdb_id in enumerate(all_ids_sorted):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Downloading {i + 1}/{len(all_ids_sorted)}...")
        result = download_cif(pdb_id, CIF_DIR)
        if result is not None:
            downloaded += 1
        else:
            failed += 1

    print(f"\nDone!")
    print(f"  Downloaded: {downloaded}")
    print(f"  Failed:     {failed}")
    print(f"  Already cached: {downloaded - len([f for f in CIF_DIR.glob('*.cif')])}")
    for name, ids in category_ids.items():
        print(f"  {name}: {len(ids)} IDs")


if __name__ == "__main__":
    main()
