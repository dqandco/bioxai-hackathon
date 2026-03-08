"""
Extract disulfide bond features from downloaded mmCIF files using BioPython.

For each protein chain, produces:
  - pdb_id, chain_id
  - Full amino acid sequence (one-letter codes)
  - List of 0-indexed sequence positions of disulfide-bonded cysteines
  - List of 0-indexed sequence positions of free cysteines

Output: data/disulfide_features.parquet
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Data.PDBData import protein_letters_3to1_extended


DATA_DIR = Path("data")
CIF_DIR = DATA_DIR / "cif_files"
OUTPUT_PARQUET = DATA_DIR / "disulfide_features.parquet"


def get_disulfide_residues(mmcif_dict: dict) -> set[tuple[str, str]]:
    """
    Return set of (auth_asym_id, auth_seq_id) tuples for residues
    involved in disulfide bonds.
    """
    disulfide = set()

    conn_types = mmcif_dict.get("_struct_conn.conn_type_id")
    if conn_types is None:
        return disulfide

    # MMCIF2Dict returns a string if there's only one row, list otherwise
    if isinstance(conn_types, str):
        conn_types = [conn_types]

    p1_chains = mmcif_dict.get("_struct_conn.ptnr1_auth_asym_id", [])
    p1_seqs = mmcif_dict.get("_struct_conn.ptnr1_auth_seq_id", [])
    p2_chains = mmcif_dict.get("_struct_conn.ptnr2_auth_asym_id", [])
    p2_seqs = mmcif_dict.get("_struct_conn.ptnr2_auth_seq_id", [])

    if isinstance(p1_chains, str):
        p1_chains, p1_seqs = [p1_chains], [p1_seqs]
        p2_chains, p2_seqs = [p2_chains], [p2_seqs]

    for i, ctype in enumerate(conn_types):
        if ctype == "disulf":
            disulfide.add((p1_chains[i], p1_seqs[i]))
            disulfide.add((p2_chains[i], p2_seqs[i]))

    return disulfide


def extract_chain_data(cif_path: Path) -> list[dict]:
    """
    Parse one mmCIF file and return a record per chain with:
      - sequence, disulfide_indices, free_indices
    """
    pdb_id = cif_path.stem.upper()

    mmcif_dict = MMCIF2Dict(str(cif_path))
    disulfide_set = get_disulfide_residues(mmcif_dict)

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(cif_path))

    records = []
    for model in structure:
        for chain in model:
            sequence_chars = []
            cys_positions_disulfide = []
            cys_positions_free = []
            pos = 0

            for residue in chain.get_residues():
                # Skip hetero residues (water, ligands, etc.)
                if residue.id[0] != " ":
                    continue

                resname = residue.resname.strip()
                one_letter = protein_letters_3to1_extended.get(resname, "X")
                sequence_chars.append(one_letter)

                if resname == "CYS":
                    auth_seq_id = str(residue.id[1])
                    if (chain.id, auth_seq_id) in disulfide_set:
                        cys_positions_disulfide.append(pos)
                    else:
                        cys_positions_free.append(pos)

                pos += 1

            sequence = "".join(sequence_chars)
            if not sequence:
                continue

            # Only include chains that have at least one cysteine
            if not cys_positions_disulfide and not cys_positions_free:
                continue

            records.append({
                "pdb_id": pdb_id,
                "chain_id": chain.id,
                "sequence": sequence,
                "seq_length": len(sequence),
                "disulfide_cys_indices": cys_positions_disulfide,
                "free_cys_indices": cys_positions_free,
                "n_disulfide_cys": len(cys_positions_disulfide),
                "n_free_cys": len(cys_positions_free),
            })

        break  # first model only

    return records


def main():
    cif_files = sorted(CIF_DIR.glob("*.cif"))
    print(f"Found {len(cif_files)} mmCIF files to process.")

    all_records: list[dict] = []
    errors = 0

    for i, cif_path in enumerate(cif_files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(cif_files)}...")

        try:
            records = extract_chain_data(cif_path)
            all_records.extend(records)
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  Warning: {cif_path.name} failed: {e}")

    if not all_records:
        print("No records extracted!")
        return

    # Build Arrow table with list columns for the index arrays
    table = pa.table({
        "pdb_id": [r["pdb_id"] for r in all_records],
        "chain_id": [r["chain_id"] for r in all_records],
        "sequence": [r["sequence"] for r in all_records],
        "seq_length": [r["seq_length"] for r in all_records],
        "disulfide_cys_indices": [r["disulfide_cys_indices"] for r in all_records],
        "free_cys_indices": [r["free_cys_indices"] for r in all_records],
        "n_disulfide_cys": [r["n_disulfide_cys"] for r in all_records],
        "n_free_cys": [r["n_free_cys"] for r in all_records],
    })

    pq.write_table(table, OUTPUT_PARQUET)

    # Summary stats
    total_disulfide = sum(r["n_disulfide_cys"] for r in all_records)
    total_free = sum(r["n_free_cys"] for r in all_records)
    unique_pdbs = len(set(r["pdb_id"] for r in all_records))
    chains_with_both = sum(
        1 for r in all_records if r["n_disulfide_cys"] > 0 and r["n_free_cys"] > 0
    )

    print(f"\nDone! Output: {OUTPUT_PARQUET}")
    print(f"  Unique PDB entries:  {unique_pdbs}")
    print(f"  Total chains:        {len(all_records)}")
    print(f"  Chains with both:    {chains_with_both}")
    print(f"  Disulfide cysteines: {total_disulfide}")
    print(f"  Free cysteines:      {total_free}")
    print(f"  Parse errors:        {errors}")
    print(f"\nParquet schema:")
    print(table.schema)


if __name__ == "__main__":
    main()
