"""
Download proteins with disulfide bonds from RCSB PDB and extract:
  - Positive examples: cysteine residues engaged in disulfide bonds
  - Negative examples: free cysteine residues from the same proteins

Uses the RCSB Search API to find structures annotated with disulfide bonds,
then parses mmCIF files to classify each cysteine residue.
"""

import json
import os
import csv
import time
import urllib.request
import urllib.error
from pathlib import Path


DATA_DIR = Path("data")
CIF_DIR = DATA_DIR / "cif_files"
OUTPUT_CSV = DATA_DIR / "disulfide_cysteines.csv"
TARGET_PROTEINS = 750  # aim for 500-1000 range


def search_disulfide_proteins(target_count: int) -> list[str]:
    """Query RCSB Search API for proteins containing disulfide bonds."""
    query = {
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
                    # Restrict to X-ray structures with decent resolution
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5,
                    },
                },
            ],
        },
        "request_options": {
            "paginate": {"start": 0, "rows": target_count},
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "scoring_strategy": "combined",
        },
        "return_type": "entry",
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    data = json.dumps(query).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req) as resp:
        results = json.loads(resp.read().decode())

    pdb_ids = [hit["identifier"] for hit in results.get("result_set", [])]
    print(f"Search returned {len(pdb_ids)} PDB entries with disulfide bonds.")
    return pdb_ids[:target_count]


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


def parse_struct_conn_disulfides(cif_text: str) -> set[tuple[str, str, str]]:
    """
    Parse _struct_conn rows for disulfide bonds.
    Returns a set of (chain_id, residue_name, residue_seq_id) for each
    cysteine endpoint involved in a disulfide bond.
    """
    disulfide_residues = set()
    lines = cif_text.splitlines()

    # Find the _struct_conn loop
    in_struct_conn = False
    header_lines: list[str] = []
    data_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("_struct_conn."):
            in_struct_conn = True
            header_lines.append(line.strip())
        elif in_struct_conn:
            stripped = line.strip()
            if stripped.startswith("_struct_conn."):
                header_lines.append(stripped)
            elif stripped.startswith("#") or stripped.startswith("loop_") or stripped.startswith("_") and not stripped.startswith("_struct_conn"):
                break
            elif stripped:
                data_lines.append(stripped)
        i += 1

    if not header_lines:
        return disulfide_residues

    # Map column names to indices
    col_map = {}
    for idx, h in enumerate(header_lines):
        col_name = h.split()[0]  # e.g. _struct_conn.conn_type_id
        col_map[col_name] = idx

    needed = [
        "_struct_conn.conn_type_id",
        "_struct_conn.ptnr1_auth_asym_id",
        "_struct_conn.ptnr1_auth_comp_id",
        "_struct_conn.ptnr1_auth_seq_id",
        "_struct_conn.ptnr2_auth_asym_id",
        "_struct_conn.ptnr2_auth_comp_id",
        "_struct_conn.ptnr2_auth_seq_id",
    ]
    if not all(n in col_map for n in needed):
        # Try label_ columns as fallback
        needed = [
            "_struct_conn.conn_type_id",
            "_struct_conn.ptnr1_label_asym_id",
            "_struct_conn.ptnr1_label_comp_id",
            "_struct_conn.ptnr1_label_seq_id",
            "_struct_conn.ptnr2_label_asym_id",
            "_struct_conn.ptnr2_label_comp_id",
            "_struct_conn.ptnr2_label_seq_id",
        ]
        if not all(n in col_map for n in needed):
            return disulfide_residues

    type_idx = col_map[needed[0]]
    p1_chain_idx = col_map[needed[1]]
    p1_comp_idx = col_map[needed[2]]
    p1_seq_idx = col_map[needed[3]]
    p2_chain_idx = col_map[needed[4]]
    p2_comp_idx = col_map[needed[5]]
    p2_seq_idx = col_map[needed[6]]

    for row in data_lines:
        fields = _split_cif_row(row)
        if len(fields) <= max(type_idx, p1_chain_idx, p1_comp_idx, p1_seq_idx,
                              p2_chain_idx, p2_comp_idx, p2_seq_idx):
            continue
        if fields[type_idx] != "disulf":
            continue

        disulfide_residues.add((fields[p1_chain_idx], fields[p1_comp_idx], fields[p1_seq_idx]))
        disulfide_residues.add((fields[p2_chain_idx], fields[p2_comp_idx], fields[p2_seq_idx]))

    return disulfide_residues


def parse_all_cysteines(cif_text: str) -> list[tuple[str, str, str]]:
    """
    Parse _atom_site to find all unique cysteine residues (CYS).
    Returns list of (chain_id, comp_id, seq_id) tuples.
    """
    cysteines = set()
    lines = cif_text.splitlines()

    in_atom_site = False
    header_lines: list[str] = []
    i = 0

    # Find _atom_site loop header
    while i < len(lines):
        line = lines[i]
        if line.startswith("_atom_site."):
            in_atom_site = True
            header_lines.append(line.strip())
        elif in_atom_site and lines[i].strip().startswith("_atom_site."):
            header_lines.append(lines[i].strip())
        elif in_atom_site:
            break
        i += 1

    if not header_lines:
        return []

    col_map = {}
    for idx, h in enumerate(header_lines):
        col_map[h.split()[0]] = idx

    # Try auth columns first, fall back to label
    chain_col = col_map.get("_atom_site.auth_asym_id", col_map.get("_atom_site.label_asym_id"))
    comp_col = col_map.get("_atom_site.auth_comp_id", col_map.get("_atom_site.label_comp_id"))
    seq_col = col_map.get("_atom_site.auth_seq_id", col_map.get("_atom_site.label_seq_id"))

    if chain_col is None or comp_col is None or seq_col is None:
        return []

    max_col = max(chain_col, comp_col, seq_col)

    # Now read data lines
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or line.startswith("_") or line.startswith("loop_"):
            break
        if not line:
            i += 1
            continue

        fields = _split_cif_row(line)
        if len(fields) > max_col:
            comp = fields[comp_col]
            if comp == "CYS":
                cysteines.add((fields[chain_col], comp, fields[seq_col]))
        i += 1

    return list(cysteines)


def _split_cif_row(line: str) -> list[str]:
    """Split a CIF data row, handling quoted strings."""
    fields = []
    i = 0
    while i < len(line):
        if line[i] in (" ", "\t"):
            i += 1
            continue
        if line[i] in ("'", '"'):
            quote = line[i]
            i += 1
            start = i
            while i < len(line) and line[i] != quote:
                i += 1
            fields.append(line[start:i])
            i += 1  # skip closing quote
        else:
            start = i
            while i < len(line) and line[i] not in (" ", "\t"):
                i += 1
            fields.append(line[start:i])
    return fields


def process_structure(cif_path: Path, pdb_id: str) -> list[dict]:
    """
    Parse a single mmCIF file and return records for each cysteine,
    labeled as 'disulfide' or 'free'.
    """
    text = cif_path.read_text(encoding="utf-8", errors="replace")

    disulfide_cys = parse_struct_conn_disulfides(text)
    all_cys = parse_all_cysteines(text)

    records = []
    for chain, comp, seq in all_cys:
        label = "disulfide" if (chain, comp, seq) in disulfide_cys else "free"
        records.append({
            "pdb_id": pdb_id,
            "chain_id": chain,
            "residue_name": comp,
            "residue_seq_id": seq,
            "label": label,
        })
    return records


def main():
    CIF_DIR.mkdir(parents=True, exist_ok=True)

    print("Searching RCSB PDB for proteins with disulfide bonds...")
    pdb_ids = search_disulfide_proteins(TARGET_PROTEINS)
    print(f"Will process {len(pdb_ids)} structures.\n")

    all_records: list[dict] = []
    proteins_with_both = 0

    for i, pdb_id in enumerate(pdb_ids):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Processing {i + 1}/{len(pdb_ids)}...")

        cif_path = download_cif(pdb_id, CIF_DIR)
        if cif_path is None:
            continue

        records = process_structure(cif_path, pdb_id)
        if not records:
            continue

        has_disulfide = any(r["label"] == "disulfide" for r in records)
        has_free = any(r["label"] == "free" for r in records)
        if has_disulfide:
            all_records.extend(records)
            if has_free:
                proteins_with_both += 1

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pdb_id", "chain_id", "residue_name", "residue_seq_id", "label"])
        writer.writeheader()
        writer.writerows(all_records)

    disulfide_count = sum(1 for r in all_records if r["label"] == "disulfide")
    free_count = sum(1 for r in all_records if r["label"] == "free")
    unique_pdbs = len(set(r["pdb_id"] for r in all_records))

    print(f"\nDone!")
    print(f"  Proteins processed: {unique_pdbs}")
    print(f"  Proteins with both disulfide and free cysteines: {proteins_with_both}")
    print(f"  Disulfide cysteine residues (positive): {disulfide_count}")
    print(f"  Free cysteine residues (negative):      {free_count}")
    print(f"  Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
