"""
Extract contrastive features from downloaded mmCIF files using BioPython.

Produces per-chain parquet files for each trait:
  - disulfide_features.parquet:   disulfide-bonded vs free cysteines
  - ss_features.parquet:          helix vs sheet vs coil residue indices
  - sasa_features.parquet:        buried vs exposed residues
  - ppi_features.parquet:         interface vs non-interface residues
  - binding_features.parquet:     ligand/metal-proximal vs non-binding residues
  - ptm_features.parquet:         modified vs unmodified residues
  - disorder_features.parquet:    disordered vs ordered residues (uses full canonical seq)
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from Bio.Data.PDBData import protein_letters_3to1_extended


DATA_DIR = Path("data")
CIF_DIR = DATA_DIR / "cif_files"

# Max SASA values (Å²) for relative SASA, from Tien et al. 2013
MAX_SASA = {
    "ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167,
    "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197,
    "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159,
    "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174,
}

# Non-standard amino acids that represent PTMs, mapped to parent residue
PTM_RESIDUES = {
    "SEP": "SER", "TPO": "THR", "PTR": "TYR",  # phosphorylation
    "MLY": "LYS", "M3L": "LYS", "ALY": "LYS",  # methylation/acetylation
    "HYP": "PRO",  # hydroxylation
    "CSO": "CYS", "OCS": "CYS",  # cysteine oxidation
    "NEP": "HIS",  # nε-methylhistidine
    "SMC": "CYS",  # s-methylcysteine
    "CSD": "CYS",  # 3-sulfinoalanine
    "TYS": "TYR",  # sulfotyrosine
}

CONTACT_DISTANCE = 5.0  # Å for PPI and ligand contacts
SASA_BURIAL_THRESHOLD = 0.25  # relative SASA below this = buried


def _ensure_list(val):
    """MMCIF2Dict returns str for single values, list for multiple."""
    if val is None:
        return []
    return [val] if isinstance(val, str) else list(val)


def build_chain_data(structure, mmcif_dict):
    """
    Build per-chain data: sequence, residue list, and auth_seq_id -> index map.
    Returns {chain_id: {sequence, residues, seq_to_idx}}.
    """
    chains = {}
    for model in structure:
        for chain in model:
            residues = []
            seq_chars = []
            seq_to_idx = {}  # (chain_id, auth_seq_id_str) -> 0-based index
            pos = 0
            for residue in chain.get_residues():
                het_flag = residue.id[0]
                resname = residue.resname.strip()
                # Include standard residues and known PTM-modified residues
                if het_flag != " " and resname not in PTM_RESIDUES:
                    continue
                # For PTM residues, use the parent amino acid's one-letter code
                if resname in PTM_RESIDUES:
                    one_letter = protein_letters_3to1_extended.get(PTM_RESIDUES[resname], "X")
                else:
                    one_letter = protein_letters_3to1_extended.get(resname, "X")
                seq_chars.append(one_letter)
                residues.append(residue)
                seq_to_idx[(chain.id, str(residue.id[1]))] = pos
                pos += 1

            if seq_chars:
                chains[chain.id] = {
                    "sequence": "".join(seq_chars),
                    "residues": residues,
                    "seq_to_idx": seq_to_idx,
                }
        break  # first model only
    return chains


# ---------------------------------------------------------------------------
# Trait extractors — each returns {chain_id: dict_of_index_lists}
# ---------------------------------------------------------------------------


def extract_disulfide(mmcif_dict, chain_data):
    """Disulfide-bonded vs free cysteines."""
    conn_types = _ensure_list(mmcif_dict.get("_struct_conn.conn_type_id"))
    p1_chains = _ensure_list(mmcif_dict.get("_struct_conn.ptnr1_auth_asym_id"))
    p1_seqs = _ensure_list(mmcif_dict.get("_struct_conn.ptnr1_auth_seq_id"))
    p2_chains = _ensure_list(mmcif_dict.get("_struct_conn.ptnr2_auth_asym_id"))
    p2_seqs = _ensure_list(mmcif_dict.get("_struct_conn.ptnr2_auth_seq_id"))

    disulfide_set = set()
    for i, ct in enumerate(conn_types):
        if ct == "disulf":
            disulfide_set.add((p1_chains[i], p1_seqs[i]))
            disulfide_set.add((p2_chains[i], p2_seqs[i]))

    results = {}
    for cid, cdata in chain_data.items():
        disulfide_idx, free_idx = [], []
        for pos, res in enumerate(cdata["residues"]):
            if res.resname.strip() == "CYS":
                key = (cid, str(res.id[1]))
                if key in disulfide_set:
                    disulfide_idx.append(pos)
                else:
                    free_idx.append(pos)
        if disulfide_idx or free_idx:
            results[cid] = {
                "disulfide_cys_indices": disulfide_idx,
                "free_cys_indices": free_idx,
            }
    return results


def extract_secondary_structure(mmcif_dict, chain_data):
    """Helix, sheet, and coil residue indices from _struct_conf / _struct_sheet_range."""
    # Parse helix ranges
    helix_ranges = []
    conf_types = _ensure_list(mmcif_dict.get("_struct_conf.conf_type_id"))
    conf_beg_chains = _ensure_list(mmcif_dict.get("_struct_conf.beg_auth_asym_id"))
    conf_beg_seqs = _ensure_list(mmcif_dict.get("_struct_conf.beg_auth_seq_id"))
    conf_end_chains = _ensure_list(mmcif_dict.get("_struct_conf.end_auth_asym_id"))
    conf_end_seqs = _ensure_list(mmcif_dict.get("_struct_conf.end_auth_seq_id"))

    for i, ct in enumerate(conf_types):
        if ct.startswith("HELX"):
            try:
                helix_ranges.append((
                    conf_beg_chains[i], int(conf_beg_seqs[i]),
                    conf_end_chains[i], int(conf_end_seqs[i]),
                ))
            except (ValueError, IndexError):
                continue

    # Parse sheet ranges
    sheet_ranges = []
    sheet_beg_chains = _ensure_list(mmcif_dict.get("_struct_sheet_range.beg_auth_asym_id"))
    sheet_beg_seqs = _ensure_list(mmcif_dict.get("_struct_sheet_range.beg_auth_seq_id"))
    sheet_end_chains = _ensure_list(mmcif_dict.get("_struct_sheet_range.end_auth_asym_id"))
    sheet_end_seqs = _ensure_list(mmcif_dict.get("_struct_sheet_range.end_auth_seq_id"))

    for i in range(len(sheet_beg_chains)):
        try:
            sheet_ranges.append((
                sheet_beg_chains[i], int(sheet_beg_seqs[i]),
                sheet_end_chains[i], int(sheet_end_seqs[i]),
            ))
        except (ValueError, IndexError):
            continue

    results = {}
    for cid, cdata in chain_data.items():
        helix_idx = set()
        sheet_idx = set()
        s2i = cdata["seq_to_idx"]

        for beg_c, beg_s, end_c, end_s in helix_ranges:
            if beg_c != cid:
                continue
            for seq_num in range(beg_s, end_s + 1):
                idx = s2i.get((cid, str(seq_num)))
                if idx is not None:
                    helix_idx.add(idx)

        for beg_c, beg_s, end_c, end_s in sheet_ranges:
            if beg_c != cid:
                continue
            for seq_num in range(beg_s, end_s + 1):
                idx = s2i.get((cid, str(seq_num)))
                if idx is not None:
                    sheet_idx.add(idx)

        coil_idx = set(range(len(cdata["sequence"]))) - helix_idx - sheet_idx
        if helix_idx or sheet_idx:
            results[cid] = {
                "helix_indices": sorted(helix_idx),
                "sheet_indices": sorted(sheet_idx),
                "coil_indices": sorted(coil_idx),
            }
    return results


def extract_solvent_accessibility(structure, chain_data):
    """Buried vs exposed residues based on relative SASA."""
    try:
        sr = ShrakeRupley()
        sr.compute(structure[0], level="R")
    except Exception:
        return {}

    results = {}
    for cid, cdata in chain_data.items():
        buried_idx, exposed_idx = [], []
        for pos, res in enumerate(cdata["residues"]):
            resname = res.resname.strip()
            max_sasa = MAX_SASA.get(resname)
            if max_sasa is None or max_sasa == 0:
                continue
            try:
                rel_sasa = res.sasa / max_sasa
            except AttributeError:
                continue
            if rel_sasa < SASA_BURIAL_THRESHOLD:
                buried_idx.append(pos)
            else:
                exposed_idx.append(pos)
        if buried_idx and exposed_idx:
            results[cid] = {
                "buried_indices": buried_idx,
                "exposed_indices": exposed_idx,
            }
    return results


def extract_ppi_sites(structure, chain_data):
    """Interface vs non-interface residues at chain-chain contacts."""
    model = structure[0]
    chain_ids = list(chain_data.keys())
    if len(chain_ids) < 2:
        return {}

    # Build atom list per chain
    chain_atoms = {}
    for chain in model:
        if chain.id in chain_data:
            atoms = [a for r in chain_data[chain.id]["residues"] for a in r.get_atoms()]
            if atoms:
                chain_atoms[chain.id] = atoms

    if len(chain_atoms) < 2:
        return {}

    # For each chain, find residues with cross-chain contacts
    results = {}
    for cid in chain_atoms:
        # Build neighbor search from all OTHER chains' atoms
        other_atoms = []
        for other_cid, atoms in chain_atoms.items():
            if other_cid != cid:
                other_atoms.extend(atoms)
        if not other_atoms:
            continue

        ns = NeighborSearch(other_atoms)
        interface_residues = set()
        for res in chain_data[cid]["residues"]:
            for atom in res.get_atoms():
                neighbors = ns.search(atom.coord, CONTACT_DISTANCE, "A")
                if neighbors:
                    interface_residues.add(res)
                    break

        interface_idx = []
        non_interface_idx = []
        for pos, res in enumerate(chain_data[cid]["residues"]):
            if res in interface_residues:
                interface_idx.append(pos)
            else:
                non_interface_idx.append(pos)

        if interface_idx and non_interface_idx:
            results[cid] = {
                "interface_indices": interface_idx,
                "non_interface_indices": non_interface_idx,
            }
    return results


def extract_binding_sites(structure, chain_data):
    """Residues near ligands/metals vs non-binding residues."""
    model = structure[0]

    # Collect all HET atoms (non-standard, non-water)
    het_atoms = []
    for chain in model:
        for residue in chain.get_residues():
            het_flag = residue.id[0]
            if het_flag not in (" ", "W"):  # HET and not water
                het_atoms.extend(residue.get_atoms())

    if not het_atoms:
        return {}

    ns = NeighborSearch(het_atoms)
    results = {}
    for cid, cdata in chain_data.items():
        binding_idx, non_binding_idx = [], []
        for pos, res in enumerate(cdata["residues"]):
            is_binding = False
            for atom in res.get_atoms():
                if ns.search(atom.coord, CONTACT_DISTANCE, "A"):
                    is_binding = True
                    break
            if is_binding:
                binding_idx.append(pos)
            else:
                non_binding_idx.append(pos)

        if binding_idx and non_binding_idx:
            results[cid] = {
                "binding_indices": binding_idx,
                "non_binding_indices": non_binding_idx,
            }
    return results


def extract_ptm_sites(mmcif_dict, chain_data):
    """Modified vs unmodified residues from _pdbx_struct_mod_residue."""
    mod_chains = _ensure_list(mmcif_dict.get("_pdbx_struct_mod_residue.auth_asym_id"))
    mod_seqs = _ensure_list(mmcif_dict.get("_pdbx_struct_mod_residue.auth_seq_id"))
    mod_set = set(zip(mod_chains, mod_seqs))

    # Also detect non-standard amino acids that are common PTMs.
    # These show up as standard residues in BioPython but with non-standard
    # resnames in the raw mmCIF. Check _atom_site for PTM_RESIDUES.
    comp_ids = _ensure_list(mmcif_dict.get("_atom_site.auth_comp_id"))
    asym_ids = _ensure_list(mmcif_dict.get("_atom_site.auth_asym_id"))
    seq_ids = _ensure_list(mmcif_dict.get("_atom_site.auth_seq_id"))
    for i, comp in enumerate(comp_ids):
        if comp in PTM_RESIDUES:
            mod_set.add((asym_ids[i], seq_ids[i]))

    if not mod_set:
        return {}

    results = {}
    for cid, cdata in chain_data.items():
        ptm_idx, non_ptm_idx = [], []
        for pos, res in enumerate(cdata["residues"]):
            key = (cid, str(res.id[1]))
            if key in mod_set:
                ptm_idx.append(pos)
            else:
                non_ptm_idx.append(pos)
        if ptm_idx and non_ptm_idx:
            results[cid] = {
                "ptm_indices": ptm_idx,
                "non_ptm_indices": non_ptm_idx,
            }
    return results


def extract_disorder(mmcif_dict, chain_data):
    """
    Disordered vs ordered residues.

    Uses the full canonical sequence from _entity_poly and identifies
    positions missing from the resolved structure as disordered.

    Returns records with the FULL canonical sequence (not the resolved-only
    sequence used by other traits), since ESM3 needs the complete sequence
    to produce hidden states at disordered positions.
    """
    # Build entity -> canonical sequence mapping
    entity_ids = _ensure_list(mmcif_dict.get("_entity_poly.entity_id"))
    raw_seqs = _ensure_list(mmcif_dict.get("_entity_poly.pdbx_seq_one_letter_code_can"))
    entity_seq = {}
    for eid, seq in zip(entity_ids, raw_seqs):
        entity_seq[eid] = seq.replace("\n", "").replace(" ", "")

    # Build chain (auth) -> entity mapping via _struct_asym
    # _struct_asym.id = label_asym_id, but we need auth_asym_id
    # Use _atom_site to find the mapping: label_asym_id -> auth_asym_id
    label_asym_ids = _ensure_list(mmcif_dict.get("_struct_asym.id"))
    entity_for_asym = _ensure_list(mmcif_dict.get("_struct_asym.entity_id"))
    label_to_entity = dict(zip(label_asym_ids, entity_for_asym))

    # Map label_asym_id -> auth_asym_id from atom_site
    atom_label_asym = _ensure_list(mmcif_dict.get("_atom_site.label_asym_id"))
    atom_auth_asym = _ensure_list(mmcif_dict.get("_atom_site.auth_asym_id"))
    label_to_auth = {}
    for la, aa in zip(atom_label_asym, atom_auth_asym):
        if la not in label_to_auth:
            label_to_auth[la] = aa

    # Build auth_chain -> canonical sequence
    chain_canon_seq = {}
    for label_id, eid in label_to_entity.items():
        auth_id = label_to_auth.get(label_id)
        if auth_id and eid in entity_seq and auth_id in chain_data:
            chain_canon_seq[auth_id] = entity_seq[eid]

    # Get resolved auth_seq_ids per chain
    resolved_per_chain: dict[str, set[str]] = {}
    for cid, cdata in chain_data.items():
        resolved_per_chain[cid] = {str(r.id[1]) for r in cdata["residues"]}

    # Use _pdbx_poly_seq_scheme to map entity positions to auth positions
    scheme_asym = _ensure_list(mmcif_dict.get("_pdbx_poly_seq_scheme.pdb_strand_id"))
    scheme_seq_id = _ensure_list(mmcif_dict.get("_pdbx_poly_seq_scheme.pdb_seq_num"))
    scheme_mon_id = _ensure_list(mmcif_dict.get("_pdbx_poly_seq_scheme.mon_id"))

    if not scheme_asym:
        return {}

    results = {}
    for cid in chain_canon_seq:
        if cid not in resolved_per_chain:
            continue

        resolved = resolved_per_chain[cid]
        canon_seq = chain_canon_seq[cid]

        # Build full sequence and index mapping from poly_seq_scheme
        full_seq_chars = []
        ordered_idx = []
        disordered_idx = []

        chain_scheme = [
            (scheme_seq_id[i], scheme_mon_id[i])
            for i in range(len(scheme_asym))
            if scheme_asym[i] == cid
        ]

        if not chain_scheme:
            continue

        for pos, (auth_seq, mon_id) in enumerate(chain_scheme):
            one_letter = protein_letters_3to1_extended.get(mon_id, "X")
            full_seq_chars.append(one_letter)
            # Check if residue was resolved (exists in atom_site)
            if auth_seq in resolved and auth_seq != "?":
                ordered_idx.append(pos)
            else:
                disordered_idx.append(pos)

        full_seq = "".join(full_seq_chars)
        if disordered_idx and ordered_idx and len(full_seq) > 0:
            results[cid] = {
                "sequence": full_seq,  # override with full canonical
                "disordered_indices": disordered_idx,
                "ordered_indices": ordered_idx,
            }
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_parquet(records: list[dict], columns: list[str], path: Path):
    """Save records to parquet with list<int64> index columns."""
    if not records:
        print(f"  No records for {path.name}, skipping.")
        return

    col_data = {c: [r[c] for r in records] for c in columns}
    table = pa.table(col_data)
    pq.write_table(table, path)

    n = len(records)
    unique = len(set(r["pdb_id"] for r in records))
    print(f"  {path.name}: {n} chains from {unique} structures")


# ---------------------------------------------------------------------------
# Single-file processing (for parallelization)
# ---------------------------------------------------------------------------


def process_single_cif(cif_path: Path) -> dict[str, list[dict]]:
    """
    Process one mmCIF file and return records for all traits.
    Returns dict with keys: disulfide, ss, sasa, ppi, binding, ptm, disorder.
    Each value is a list of record dicts (or empty list).
    """
    pdb_id = cif_path.stem.upper()
    parser = MMCIFParser(QUIET=True)

    try:
        mmcif_dict = MMCIF2Dict(str(cif_path))
        structure = parser.get_structure(pdb_id, str(cif_path))
        chain_data = build_chain_data(structure, mmcif_dict)

        if not chain_data:
            return {k: [] for k in ("disulfide", "ss", "sasa", "ppi", "binding", "ptm", "disorder")}

        disulfide_records = []
        ss_records = []
        sasa_records = []
        ppi_records = []
        binding_records = []
        ptm_records = []
        disorder_records = []

        for cid, idx_data in extract_disulfide(mmcif_dict, chain_data).items():
            disulfide_records.append({
                "pdb_id": pdb_id, "chain_id": cid,
                "sequence": chain_data[cid]["sequence"],
                "seq_length": len(chain_data[cid]["sequence"]),
                **idx_data,
                "n_disulfide_cys": len(idx_data["disulfide_cys_indices"]),
                "n_free_cys": len(idx_data["free_cys_indices"]),
            })

        for cid, idx_data in extract_secondary_structure(mmcif_dict, chain_data).items():
            ss_records.append({
                "pdb_id": pdb_id, "chain_id": cid,
                "sequence": chain_data[cid]["sequence"],
                "seq_length": len(chain_data[cid]["sequence"]),
                **idx_data,
                "n_helix": len(idx_data["helix_indices"]),
                "n_sheet": len(idx_data["sheet_indices"]),
                "n_coil": len(idx_data["coil_indices"]),
            })

        for cid, idx_data in extract_solvent_accessibility(structure, chain_data).items():
            sasa_records.append({
                "pdb_id": pdb_id, "chain_id": cid,
                "sequence": chain_data[cid]["sequence"],
                "seq_length": len(chain_data[cid]["sequence"]),
                **idx_data,
                "n_buried": len(idx_data["buried_indices"]),
                "n_exposed": len(idx_data["exposed_indices"]),
            })

        for cid, idx_data in extract_ppi_sites(structure, chain_data).items():
            ppi_records.append({
                "pdb_id": pdb_id, "chain_id": cid,
                "sequence": chain_data[cid]["sequence"],
                "seq_length": len(chain_data[cid]["sequence"]),
                **idx_data,
                "n_interface": len(idx_data["interface_indices"]),
                "n_non_interface": len(idx_data["non_interface_indices"]),
            })

        for cid, idx_data in extract_binding_sites(structure, chain_data).items():
            binding_records.append({
                "pdb_id": pdb_id, "chain_id": cid,
                "sequence": chain_data[cid]["sequence"],
                "seq_length": len(chain_data[cid]["sequence"]),
                **idx_data,
                "n_binding": len(idx_data["binding_indices"]),
                "n_non_binding": len(idx_data["non_binding_indices"]),
            })

        for cid, idx_data in extract_ptm_sites(mmcif_dict, chain_data).items():
            ptm_records.append({
                "pdb_id": pdb_id, "chain_id": cid,
                "sequence": chain_data[cid]["sequence"],
                "seq_length": len(chain_data[cid]["sequence"]),
                **idx_data,
                "n_ptm": len(idx_data["ptm_indices"]),
                "n_non_ptm": len(idx_data["non_ptm_indices"]),
            })

        for cid, idx_data in extract_disorder(mmcif_dict, chain_data).items():
            disorder_records.append({
                "pdb_id": pdb_id, "chain_id": cid,
                "sequence": idx_data.pop("sequence"),
                "seq_length": len(idx_data["disordered_indices"]) + len(idx_data["ordered_indices"]),
                **idx_data,
                "n_disordered": len(idx_data["disordered_indices"]),
                "n_ordered": len(idx_data["ordered_indices"]),
            })

        return {
            "disulfide": disulfide_records,
            "ss": ss_records,
            "sasa": sasa_records,
            "ppi": ppi_records,
            "binding": binding_records,
            "ptm": ptm_records,
            "disorder": disorder_records,
        }
    except Exception:
        return {k: [] for k in ("disulfide", "ss", "sasa", "ppi", "binding", "ptm", "disorder")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cif_files = sorted(CIF_DIR.glob("*.cif"))
    print(f"Found {len(cif_files)} mmCIF files to process.\n")

    parser = MMCIFParser(QUIET=True)

    # Accumulators per trait
    disulfide_records = []
    ss_records = []
    sasa_records = []
    ppi_records = []
    binding_records = []
    ptm_records = []
    disorder_records = []

    errors = 0

    for i, cif_path in enumerate(cif_files):
        pdb_id = cif_path.stem.upper()

        if (i + 1) % 100 == 0 or i == 0:
            print(f"Processing {i + 1}/{len(cif_files)} ({pdb_id})...")

        try:
            mmcif_dict = MMCIF2Dict(str(cif_path))
            structure = parser.get_structure(pdb_id, str(cif_path))
            chain_data = build_chain_data(structure, mmcif_dict)

            if not chain_data:
                continue

            # --- Disulfide ---
            for cid, idx_data in extract_disulfide(mmcif_dict, chain_data).items():
                disulfide_records.append({
                    "pdb_id": pdb_id, "chain_id": cid,
                    "sequence": chain_data[cid]["sequence"],
                    "seq_length": len(chain_data[cid]["sequence"]),
                    **idx_data,
                    "n_disulfide_cys": len(idx_data["disulfide_cys_indices"]),
                    "n_free_cys": len(idx_data["free_cys_indices"]),
                })

            # --- Secondary Structure ---
            for cid, idx_data in extract_secondary_structure(mmcif_dict, chain_data).items():
                ss_records.append({
                    "pdb_id": pdb_id, "chain_id": cid,
                    "sequence": chain_data[cid]["sequence"],
                    "seq_length": len(chain_data[cid]["sequence"]),
                    **idx_data,
                    "n_helix": len(idx_data["helix_indices"]),
                    "n_sheet": len(idx_data["sheet_indices"]),
                    "n_coil": len(idx_data["coil_indices"]),
                })

            # --- Solvent Accessibility ---
            for cid, idx_data in extract_solvent_accessibility(structure, chain_data).items():
                sasa_records.append({
                    "pdb_id": pdb_id, "chain_id": cid,
                    "sequence": chain_data[cid]["sequence"],
                    "seq_length": len(chain_data[cid]["sequence"]),
                    **idx_data,
                    "n_buried": len(idx_data["buried_indices"]),
                    "n_exposed": len(idx_data["exposed_indices"]),
                })

            # --- PPI Sites ---
            for cid, idx_data in extract_ppi_sites(structure, chain_data).items():
                ppi_records.append({
                    "pdb_id": pdb_id, "chain_id": cid,
                    "sequence": chain_data[cid]["sequence"],
                    "seq_length": len(chain_data[cid]["sequence"]),
                    **idx_data,
                    "n_interface": len(idx_data["interface_indices"]),
                    "n_non_interface": len(idx_data["non_interface_indices"]),
                })

            # --- Binding Sites ---
            for cid, idx_data in extract_binding_sites(structure, chain_data).items():
                binding_records.append({
                    "pdb_id": pdb_id, "chain_id": cid,
                    "sequence": chain_data[cid]["sequence"],
                    "seq_length": len(chain_data[cid]["sequence"]),
                    **idx_data,
                    "n_binding": len(idx_data["binding_indices"]),
                    "n_non_binding": len(idx_data["non_binding_indices"]),
                })

            # --- PTM Sites ---
            for cid, idx_data in extract_ptm_sites(mmcif_dict, chain_data).items():
                ptm_records.append({
                    "pdb_id": pdb_id, "chain_id": cid,
                    "sequence": chain_data[cid]["sequence"],
                    "seq_length": len(chain_data[cid]["sequence"]),
                    **idx_data,
                    "n_ptm": len(idx_data["ptm_indices"]),
                    "n_non_ptm": len(idx_data["non_ptm_indices"]),
                })

            # --- Disorder ---
            for cid, idx_data in extract_disorder(mmcif_dict, chain_data).items():
                disorder_records.append({
                    "pdb_id": pdb_id, "chain_id": cid,
                    "sequence": idx_data.pop("sequence"),  # full canonical seq
                    "seq_length": len(idx_data["disordered_indices"]) + len(idx_data["ordered_indices"]),
                    **idx_data,
                    "n_disordered": len(idx_data["disordered_indices"]),
                    "n_ordered": len(idx_data["ordered_indices"]),
                })

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  Warning: {cif_path.name} failed: {e}")

    # Save all parquets
    print(f"\nSaving parquet files (errors: {errors})...")

    _save_parquet(disulfide_records, [
        "pdb_id", "chain_id", "sequence", "seq_length",
        "disulfide_cys_indices", "free_cys_indices",
        "n_disulfide_cys", "n_free_cys",
    ], DATA_DIR / "disulfide_features.parquet")

    _save_parquet(ss_records, [
        "pdb_id", "chain_id", "sequence", "seq_length",
        "helix_indices", "sheet_indices", "coil_indices",
        "n_helix", "n_sheet", "n_coil",
    ], DATA_DIR / "ss_features.parquet")

    _save_parquet(sasa_records, [
        "pdb_id", "chain_id", "sequence", "seq_length",
        "buried_indices", "exposed_indices",
        "n_buried", "n_exposed",
    ], DATA_DIR / "sasa_features.parquet")

    _save_parquet(ppi_records, [
        "pdb_id", "chain_id", "sequence", "seq_length",
        "interface_indices", "non_interface_indices",
        "n_interface", "n_non_interface",
    ], DATA_DIR / "ppi_features.parquet")

    _save_parquet(binding_records, [
        "pdb_id", "chain_id", "sequence", "seq_length",
        "binding_indices", "non_binding_indices",
        "n_binding", "n_non_binding",
    ], DATA_DIR / "binding_features.parquet")

    _save_parquet(ptm_records, [
        "pdb_id", "chain_id", "sequence", "seq_length",
        "ptm_indices", "non_ptm_indices",
        "n_ptm", "n_non_ptm",
    ], DATA_DIR / "ptm_features.parquet")

    _save_parquet(disorder_records, [
        "pdb_id", "chain_id", "sequence", "seq_length",
        "disordered_indices", "ordered_indices",
        "n_disordered", "n_ordered",
    ], DATA_DIR / "disorder_features.parquet")

    print("\nDone!")


if __name__ == "__main__":
    main()
