/**
 * Parse PDB file to extract sequence and residue index mapping.
 * PDB format: ATOM lines have chain at col 22, resi at cols 23-26, resn at cols 18-20.
 * Uses (chain, resi) as unique key to support multi-chain proteins.
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

/** Composite key for a residue: "chain:resi" (e.g. "A:42", "B:1") */
export function chainResiKey(chain: string, resi: number): string {
  return `${chain || ' '}:${resi}`
}

export interface ChainResi {
  chain: string
  resi: number
}

export interface ParsePdbResult {
  sequence: string;
  resiToSeqIdx: Record<string, number>;
  seqIdxToResi: ChainResi[];
}

export function parsePdbSequence(pdb: string): ParsePdbResult {
  const resiToSeqIdx: Record<string, number> = {};
  const seqIdxToResi: ChainResi[] = [];
  const seen = new Set<string>();
  const seqChars: string[] = [];

  const lines = pdb.split('\n');
  for (const line of lines) {
    const isHetatm = line.startsWith('HETATM');
    if (!line.startsWith('ATOM') && !isHetatm) continue;

    // Chain ID: col 22 (0-indexed: 21)
    const chain = line[21]?.trim() || ' ';

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

    const key = chainResiKey(chain, resi);
    if (!seen.has(key)) {
      seen.add(key);
      resiToSeqIdx[key] = seqChars.length;
      seqIdxToResi.push({ chain, resi });
      seqChars.push(oneLetter);
    }
  }

  return {
    sequence: seqChars.join(''),
    resiToSeqIdx,
    seqIdxToResi,
  };
}
