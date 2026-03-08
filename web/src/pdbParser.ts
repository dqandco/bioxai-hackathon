/**
 * Parse PDB file to extract sequence and residue index mapping.
 * PDB format: ATOM lines have resi at cols 23-26, resn at cols 18-20.
 * Some files use 4-char residue names for alternates (e.g. ALYS, BLYS).
 */

const RESIDUE_3TO1: Record<string, string> = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E',
  GLY: 'G', HIS: 'H', ILE: 'I', LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F',
  PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V', UNK: 'X',
  // PTM / non-standard
  SEP: 'S', TPO: 'T', PTR: 'Y', HYP: 'P', CSO: 'C', OCS: 'C',
  MLY: 'K', ALY: 'K', M3L: 'K', NEP: 'H', SMC: 'C', CSD: 'C', TYS: 'Y',
  // Alternate location variants (e.g. disordered residues in 1DPX)
  ALYS: 'K', BLYS: 'K', CLYS: 'K', DLYS: 'K',
  ASER: 'S', BSER: 'S', ATHR: 'T', BTHR: 'T', AARG: 'R', BARG: 'R',
}

// Non-polymer residues to skip (water, ligands, ions)
const SKIP_RESIDUES = new Set(['HOH', 'WAT', 'H2O', 'MPD', 'GOL', 'EDO', 'ACT', 'SO4', 'CL', 'NA', 'CA', 'MG', 'ZN', 'FE'])

export interface ParsePdbResult {
  sequence: string;
  resiToSeqIdx: Record<number, number>;
  seqIdxToResi: number[];
}

export function parsePdbSequence(pdb: string): ParsePdbResult {
  const resiToSeqIdx: Record<number, number> = {};
  const seqIdxToResi: number[] = [];
  const seen = new Set<number>();
  const seqChars: string[] = [];

  const lines = pdb.split('\n');
  for (const line of lines) {
    const isHetatm = line.startsWith('HETATM');
    if (!line.startsWith('ATOM') && !isHetatm) continue;

    // Residue name: cols 18-20 (3 chars), some files use 17-20 (4 chars) for alternates
    const resn3 = line.slice(17, 20).trim();
    const resn4 = line.slice(17, 21).trim();
    const resn = (RESIDUE_3TO1[resn4] ? resn4 : resn3) || resn3;

    if (isHetatm && SKIP_RESIDUES.has(resn)) continue;

    const resiStr = line.slice(22, 26).trim();
    const resi = parseInt(resiStr, 10);
    if (isNaN(resi)) continue;

    const oneLetter = RESIDUE_3TO1[resn] ?? RESIDUE_3TO1[resn3];
    if (!oneLetter || oneLetter === 'X') continue; // Skip unknown residues

    if (!seen.has(resi)) {
      seen.add(resi);
      resiToSeqIdx[resi] = seqChars.length;
      seqIdxToResi.push(resi);
      seqChars.push(oneLetter);
    }
  }

  return {
    sequence: seqChars.join(''),
    resiToSeqIdx,
    seqIdxToResi,
  };
}
