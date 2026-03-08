import { useEffect, useRef } from 'react';

export const RESIDUE_3TO1: Record<string, string> = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E',
  GLY: 'G', HIS: 'H', ILE: 'I', LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F',
  PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V', UNK: 'X',
};

const CONCEPT_COLORS: Record<string, string> = {
  disulfide: '#3b82f6',
  ss_helix: '#22c55e',
  ss_sheet: '#eab308',
  ss_helix_sheet: '#f97316',
  sasa: '#8b5cf6',
  ppi: '#ec4899',
  binding: '#06b6d4',
  ptm: '#84cc16',
  disorder: '#64748b',
};

const DEFAULT_COLOR = '#94a3b8';

declare global {
  interface Window {
    $3Dmol: {
      createViewer: (element: HTMLElement, options?: object) => {
        addModel: (pdb: string, format: string) => unknown;
        setStyle: (selector: object, style: object) => unknown;
        setClickable: (selector: object, clickable: boolean, callback: (atom: { resi: number; resn: string }) => void) => void;
        render: () => void;
        clear: () => void;
      };
    };
  }
}

interface ProteinViewerProps {
  pdb: string;
  residueConcepts: Record<number | string, string>;
  residueProjections?: Record<string, Record<string, number>>;
  onResidueSelect?: (resi: number, resn: string, oneLetter: string, conceptScores: Record<string, number>) => void;
  conceptColors?: Record<string, string>;
}

export function ProteinViewer({
  pdb,
  residueConcepts,
  residueProjections = {},
  onResidueSelect,
  conceptColors = CONCEPT_COLORS,
}: ProteinViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = containerRef.current;
    const $3Dmol = window.$3Dmol;

    if (!element || !$3Dmol || !pdb) return;

    const viewer = $3Dmol.createViewer(element, { backgroundColor: '0x1e293b' });
    viewer.addModel(pdb, 'pdb');

    // Default style for residues not in the map
    viewer.setStyle({}, { cartoon: { color: DEFAULT_COLOR } });

    for (const [resi, concept] of Object.entries(residueConcepts)) {
      const color = conceptColors[concept] ?? DEFAULT_COLOR;
      viewer.setStyle({ resi: parseInt(String(resi), 10) }, { cartoon: { color } });
    }

    if (onResidueSelect) {
      viewer.setClickable({}, true, (atom: { resi: number; resn: string }) => {
        const resi = atom.resi;
        const resn = atom.resn;
        const oneLetter = RESIDUE_3TO1[resn] ?? 'X';
        const conceptScores = residueProjections[String(resi)] ?? {};
        onResidueSelect(resi, resn, oneLetter, conceptScores);
      });
    }

    viewer.render();

    return () => {
      viewer.clear();
      element.innerHTML = '';
    };
  }, [pdb, residueConcepts, residueProjections, onResidueSelect, conceptColors]);

  if (!pdb) {
    return (
      <div className="flex h-96 min-h-[384px] w-full items-center justify-center rounded-lg border border-dashed border-gray-600 bg-gray-800/30 text-gray-500">
        No PDB structure loaded — use Load demo or Upload PDB
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="h-96 w-full min-h-[384px] rounded-lg border border-gray-600 bg-gray-800 overflow-hidden"
      style={{ position: 'relative' }}
    />
  );
}

export { CONCEPT_COLORS };
