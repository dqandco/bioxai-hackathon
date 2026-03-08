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

export type ViewMode = 'cartoon' | 'stick' | 'line' | 'ballstick';

interface ProteinViewerProps {
  pdb: string;
  residueConcepts: Record<number | string, string>;
  residueProjections?: Record<string, Record<string, number>>;
  onResidueSelect?: (resi: number, resn: string, oneLetter: string, conceptScores: Record<string, number>) => void;
  conceptColors?: Record<string, string>;
  selectedResidue?: number | null;
  viewMode?: ViewMode;
  activeConcepts?: Set<string>;
}

const HIGHLIGHT_COLOR = '#2d2a26';

function getBaseStyle(viewMode: ViewMode, color: string, opacity?: number): Record<string, unknown> {
  const opacityProp = opacity != null ? { opacity } : {};
  switch (viewMode) {
    case 'stick':
      return { stick: { color, radius: 0.3, ...opacityProp } };
    case 'line':
      return { line: { color, ...opacityProp } };
    case 'ballstick':
      return { stick: { color, radius: 0.15, ...opacityProp }, sphere: { color, scale: 0.25, ...opacityProp } };
    default:
      return { cartoon: { color, ...opacityProp } };
  }
}

export function ProteinViewer({
  pdb,
  residueConcepts,
  residueProjections = {},
  onResidueSelect,
  conceptColors = CONCEPT_COLORS,
  selectedResidue = null,
  viewMode = 'cartoon',
  activeConcepts,
}: ProteinViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<ReturnType<typeof window.$3Dmol.createViewer> | null>(null);
  const activeSet = activeConcepts ?? new Set(Object.keys(CONCEPT_COLORS));

  // Create/destroy viewer only when pdb changes
  useEffect(() => {
    const element = containerRef.current;
    const $3Dmol = window.$3Dmol;

    if (!element || !$3Dmol) return;

    if (!pdb) {
      if (viewerRef.current) {
        viewerRef.current.clear();
        element.innerHTML = '';
        viewerRef.current = null;
      }
      return;
    }

    const viewer = $3Dmol.createViewer(element, { backgroundColor: '0xf5f3ef' });
    viewer.addModel(pdb, 'pdb');
    viewerRef.current = viewer;

    return () => {
      if (viewerRef.current) {
        viewerRef.current.clear();
        element.innerHTML = '';
        viewerRef.current = null;
      }
    };
  }, [pdb]);

  // Update styles when residueConcepts, viewMode, selectedResidue, or activeConcepts change (preserves camera)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !pdb) return;

    if (onResidueSelect) {
      viewer.setClickable({}, true, (atom: { resi: number; resn: string }) => {
        const resi = atom.resi;
        const resn = atom.resn;
        const oneLetter = RESIDUE_3TO1[resn] ?? 'X';
        const conceptScores = residueProjections[String(resi)] ?? {};
        onResidueSelect(resi, resn, oneLetter, conceptScores);
      });
    }

    const dimOpacity = selectedResidue != null ? 0.5 : undefined;

    // Default style for residues not in the map
    viewer.setStyle({}, getBaseStyle(viewMode, DEFAULT_COLOR, dimOpacity));

    for (const [resi, concept] of Object.entries(residueConcepts)) {
      const isActive = activeSet.has(concept);
      const color = isActive ? (conceptColors[concept] ?? DEFAULT_COLOR) : DEFAULT_COLOR;
      const resiNum = parseInt(String(resi), 10);
      const isSelected = resiNum === selectedResidue;
      const opacity = isSelected ? undefined : dimOpacity;

      if (isSelected) {
        viewer.setStyle(
          { resi: resiNum },
          { ...getBaseStyle(viewMode, color), stick: { color: HIGHLIGHT_COLOR, radius: 0.35 } }
        );
      } else {
        viewer.setStyle({ resi: resiNum }, getBaseStyle(viewMode, color, opacity));
      }
    }

    viewer.render();
  }, [pdb, residueConcepts, residueProjections, onResidueSelect, viewMode, selectedResidue, conceptColors, activeSet]);

  if (!pdb) {
    return (
      <div
        className="flex h-96 min-h-[384px] w-full items-center justify-center rounded text-[13px]"
        style={{ backgroundColor: 'var(--bg-soft)', color: 'var(--text-muted)' }}
      >
        No PDB structure loaded — use Load demo or Upload PDB
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="h-96 w-full min-h-[384px] rounded overflow-hidden"
      style={{ position: 'relative', backgroundColor: 'var(--bg-soft)' }}
    />
  );
}

export { CONCEPT_COLORS };
