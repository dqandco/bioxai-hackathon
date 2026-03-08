import { useEffect, useRef } from 'react';
import { chainResiKey } from './pdbParser';

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
        zoomTo: (selector?: object, duration?: number) => void;
      };
      assignPDBBonds?: (viewer: unknown) => void;
    };
  }
}

export type ViewMode = 'cartoon' | 'stick' | 'line' | 'ballstick';

export interface SelectedResidueSpec {
  chain: string;
  resi: number;
}

interface ProteinViewerProps {
  pdb: string;
  residueConcepts: Record<number | string, string>;
  residueProjections?: Record<string, Record<string, number>>;
  onResidueSelect?: (chain: string, resi: number, resn: string, oneLetter: string, conceptScores: Record<string, number>) => void;
  onResidueDeselect?: () => void;
  conceptColors?: Record<string, string>;
  selectedResidue?: SelectedResidueSpec | null;
  /** When a residue's dominant concept is unchecked, use this rank (0=2nd best, 1=3rd best, etc.) */
  uncheckedFallbackRank?: number;
  viewMode?: ViewMode;
  activeConcepts?: Set<string>;
}

/** Get concept at rank index (0=best, 1=2nd best, etc.) from scores. */
function getConceptAtRank(scores: Record<string, number>, rank: number): string {
  const sorted = Object.entries(scores).sort(([, a], [, b]) => b - a);
  return sorted[rank]?.[0] ?? sorted[0]?.[0] ?? '';
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
  onResidueDeselect,
  conceptColors = CONCEPT_COLORS,
  selectedResidue = null,
  uncheckedFallbackRank = 0,
  viewMode = 'cartoon',
  activeConcepts,
}: ProteinViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<ReturnType<typeof window.$3Dmol.createViewer> | null>(null);
  const clickedResidueRef = useRef(false);
  const mouseDownRef = useRef<{ x: number; y: number } | null>(null);
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
    if (typeof $3Dmol.assignPDBBonds === 'function') {
      $3Dmol.assignPDBBonds(viewer);
    }
    viewer.zoomTo();
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
      viewer.setClickable({}, true, (atom: { chain?: string; resi: number; resn: string }) => {
        clickedResidueRef.current = true;
        const chain = atom.chain ?? ' ';
        const resi = atom.resi;
        const resn = atom.resn;
        const oneLetter = RESIDUE_3TO1[resn] ?? 'X';
        const key = chainResiKey(chain, resi);
        const conceptScores = residueProjections[key] ?? {};
        onResidueSelect(chain, resi, resn, oneLetter, conceptScores);
      });
    }

    const dimOpacity = selectedResidue != null ? 0.5 : undefined;
    const selectedKey = selectedResidue ? chainResiKey(selectedResidue.chain, selectedResidue.resi) : null;

    // Default style for residues not in the map
    viewer.setStyle({}, getBaseStyle(viewMode, DEFAULT_COLOR, dimOpacity));

    for (const [key, concept] of Object.entries(residueConcepts)) {
      // Parse "chain:resi" format
      const colonIdx = key.indexOf(':');
      const chain = colonIdx >= 0 ? key.slice(0, colonIdx) : ' ';
      const resiNum = parseInt(colonIdx >= 0 ? key.slice(colonIdx + 1) : key, 10);

      const isSelected = key === selectedKey;
      const opacity = isSelected ? undefined : dimOpacity;

      // If dominant concept is unchecked, use next-best fallback (2nd, 3rd, etc.)
      const scores = residueProjections[key];
      let displayConcept = concept;
      if (scores && !activeSet.has(concept)) {
        const fallback = getConceptAtRank(scores, 1 + uncheckedFallbackRank);
        if (fallback) displayConcept = fallback;
      }

      const isActive = activeSet.has(displayConcept);
      const color = isActive ? (conceptColors[displayConcept] ?? DEFAULT_COLOR) : DEFAULT_COLOR;

      const selector = chain.trim() ? { chain, resi: resiNum } : { resi: resiNum };

      if (isSelected) {
        viewer.setStyle(
          selector,
          { ...getBaseStyle(viewMode, color), stick: { color: HIGHLIGHT_COLOR, radius: 0.35 } }
        );
      } else {
        viewer.setStyle(selector, getBaseStyle(viewMode, color, opacity));
      }
    }

    viewer.render();
  }, [pdb, residueConcepts, residueProjections, onResidueSelect, viewMode, selectedResidue, uncheckedFallbackRank, conceptColors, activeSet]);

  // Zoom to selected residue only when selection changes (not when layer/concepts change)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !pdb || selectedResidue == null || !viewer.zoomTo) return;
    const sel = selectedResidue.chain.trim()
      ? { chain: selectedResidue.chain, resi: selectedResidue.resi }
      : { resi: selectedResidue.resi };
    viewer.zoomTo(sel, 400);
  }, [pdb, selectedResidue]);

  // Click-off to deselect: only when clicking empty space (outside protein), not when rotating
  useEffect(() => {
    const element = containerRef.current;
    if (!element || !pdb || !onResidueDeselect) return;

    const DRAG_THRESHOLD = 5;

    const handleMouseDown = (e: MouseEvent) => {
      mouseDownRef.current = { x: e.clientX, y: e.clientY };
    };

    const handleClick = (e: MouseEvent) => {
      if (selectedResidue == null) return;
      const wasResidue = clickedResidueRef.current;
      clickedResidueRef.current = false;

      // Don't deselect if we were dragging (rotate) - only on actual clicks
      const start = mouseDownRef.current;
      mouseDownRef.current = null;
      if (start) {
        const dx = e.clientX - start.x;
        const dy = e.clientY - start.y;
        if (dx * dx + dy * dy > DRAG_THRESHOLD * DRAG_THRESHOLD) return; // was a drag
      }

      if (!wasResidue) {
        onResidueDeselect();
      }
    };

    element.addEventListener('mousedown', handleMouseDown);
    element.addEventListener('click', handleClick);
    return () => {
      element.removeEventListener('mousedown', handleMouseDown);
      element.removeEventListener('click', handleClick);
    };
  }, [pdb, onResidueDeselect, selectedResidue]);

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
