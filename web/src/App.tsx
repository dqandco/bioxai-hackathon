import { useState, useCallback, useMemo } from 'react'
import { ProteinViewer, CONCEPT_COLORS, type ViewMode } from './ProteinViewer'
import { parsePdbSequence, chainResiKey, type ChainResi } from './pdbParser'

const API_BASE = '/api'
const DEMO_PDB_URL = 'https://files.rcsb.org/view/1CRN.pdb'

type ResidueConcepts = Record<number | string, string>
type ResidueProjections = Record<string, Record<string, number>>

interface InferenceResponse {
  sequence: string
  n_residues: number
  n_layers: number
  concepts: string[]
  projections: Record<string, number[][]>  // [concept][seq_idx][layer_idx]
}

interface SelectedResidue {
  chain: string
  resi: number
  resn: string
  oneLetter: string
  conceptScores: Record<string, number>
}

const CONCEPT_LABELS: Record<string, string> = {
  disulfide: 'Disulfide-bonded vs free cysteines',
  ss_helix: 'Helix vs coil',
  ss_sheet: 'Sheet vs coil',
  ss_helix_sheet: 'Helix vs sheet',
  sasa: 'Buried vs exposed',
  ppi: 'Interface vs non-interface',
  binding: 'Ligand/metal-binding vs non-binding',
  ptm: 'PTM-modified vs unmodified',
  disorder: 'Disordered vs ordered',
}

function App() {
  const [pdb, setPdb] = useState<string>('')
  const [pdbName, setPdbName] = useState<string>('')
  const [inferenceData, setInferenceData] = useState<InferenceResponse | null>(null)
  const [seqIdxToResi, setSeqIdxToResi] = useState<ChainResi[]>([])
  const [residueConcepts, setResidueConcepts] = useState<ResidueConcepts>({})
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [selectedResidue, setSelectedResidue] = useState<SelectedResidue | null>(null)
  const [uncheckedFallbackRank, setUncheckedFallbackRank] = useState(0)
  const [viewMode, setViewMode] = useState<ViewMode>('cartoon')
  const [activeConcepts, setActiveConcepts] = useState<Set<string>>(() => new Set(Object.keys(CONCEPT_COLORS)))
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState(false)

  const nLayers = inferenceData?.n_layers ?? 0

  const runInference = useCallback(async (sequence: string) => {
    setError('')
    setLoading(true)
    setInferenceData(null)
    try {
      const res = await fetch(`${API_BASE}/inference/project`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        const detail = err.detail
        const msg = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((d: { msg?: string }) => d.msg).join(', ') : `API error: ${res.status}`
        throw new Error(msg || `API error: ${res.status}`)
      }
      const data = (await res.json()) as InferenceResponse
      setInferenceData(data)
      setSelectedLayer(data.n_layers > 0 ? Math.floor(data.n_layers / 2) : 0)
      setActiveConcepts(new Set(data.concepts))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Inference failed')
    } finally {
      setLoading(false)
    }
  }, [])

  const loadDemo = useCallback(async () => {
    setError('')
    setLoading(true)
    try {
      const res = await fetch(DEMO_PDB_URL)
      if (!res.ok) throw new Error('Failed to fetch demo PDB')
      const text = await res.text()
      setPdb(text)
      setPdbName('1CRN')
      const { sequence, seqIdxToResi: seqToResi } = parsePdbSequence(text)
      setSeqIdxToResi(seqToResi)
      await runInference(sequence)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load demo')
    } finally {
      setLoading(false)
    }
  }, [runInference])

  const handlePdbUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (!file) return
      setError('')
      const reader = new FileReader()
      reader.onload = async () => {
        const text = String(reader.result)
        const name = file.name.replace(/\.pdb$/i, '').toUpperCase() || 'PDB'
        setPdb(text)
        setPdbName(name)
        try {
          const { sequence, seqIdxToResi: seqToResi } = parsePdbSequence(text)
          setSeqIdxToResi(seqToResi)
          setInferenceData(null)
          setResidueConcepts({})
          await runInference(sequence)
        } catch {
          setError('Failed to parse PDB or run inference')
        }
      }
      reader.readAsText(file)
      e.target.value = ''
    },
    [runInference]
  )

  const handleJsonUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError('')
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const data = JSON.parse(String(reader.result)) as ResidueConcepts
        setResidueConcepts(data)
        setInferenceData(null)
        setActiveConcepts(new Set(Object.values(data)))
      } catch {
        setError('Invalid JSON file')
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }, [])

  const { residueConceptsForLayer, residueProjectionsForLayer } = useMemo(() => {
    if (!inferenceData || seqIdxToResi.length === 0) {
      return { residueConceptsForLayer: residueConcepts, residueProjectionsForLayer: {} as ResidueProjections }
    }

    const layer = Math.min(selectedLayer, inferenceData.n_layers - 1)
    const concepts = inferenceData.concepts
    const projections = inferenceData.projections

    const outConcepts: ResidueConcepts = {}
    const outProjections: ResidueProjections = {}

    for (let seqIdx = 0; seqIdx < inferenceData.n_residues; seqIdx++) {
      const cr = seqIdxToResi[seqIdx]
      if (!cr) continue

      const key = chainResiKey(cr.chain, cr.resi)

      let maxScore = -Infinity
      let maxConcept = ''
      const scores: Record<string, number> = {}

      for (const concept of concepts) {
        const layerScores = projections[concept]?.[seqIdx]
        const score = layerScores?.[layer] ?? 0
        scores[concept] = score
        if (score > maxScore) {
          maxScore = score
          maxConcept = concept
        }
      }

      outConcepts[key] = maxConcept
      outProjections[key] = scores
    }

    return { residueConceptsForLayer: outConcepts, residueProjectionsForLayer: outProjections }
  }, [inferenceData, seqIdxToResi, selectedLayer, residueConcepts])

  const handleResidueSelect = useCallback(
    (chain: string, resi: number, resn: string, oneLetter: string, conceptScores: Record<string, number>) => {
      setSelectedResidue({ chain, resi, resn, oneLetter, conceptScores })
    },
    []
  )

  const handleResidueDeselect = useCallback(() => {
    setSelectedResidue(null)
  }, [])

  const effectiveLayer = inferenceData ? Math.min(selectedLayer, inferenceData.n_layers - 1) : 0

  return (
    <div className="min-h-screen p-6" style={{ backgroundColor: 'var(--bg-soft)', color: 'var(--text-primary)' }}>
      <div className="mx-auto max-w-6xl space-y-5">
        <div className="flex items-baseline gap-3">
          <h1 className="text-xl font-semibold">Protein Concept Vector Viewer</h1>
          {pdbName && (
            <span className="text-[13px]" style={{ color: 'var(--text-muted)' }}>
              {pdbName}
            </span>
          )}
        </div>
        <p className="text-[13px]" style={{ color: 'var(--text-secondary)' }}>
          Visualize residues colored by their dominant concept. Click a residue to see amino acid and similarity scores.
        </p>

        <div className="flex flex-wrap gap-3">
          <button
            onClick={loadDemo}
            disabled={loading}
            className="px-3 py-1.5 text-[13px] disabled:opacity-50 rounded font-medium transition-opacity hover:opacity-80"
            style={{ backgroundColor: 'var(--bg-card)', color: 'var(--text-primary)' }}
          >
            {loading ? 'Loading...' : 'Load demo (1CRN)'}
          </button>
          <label className="px-3 py-1.5 text-[13px] rounded font-medium cursor-pointer transition-opacity hover:opacity-80" style={{ backgroundColor: 'var(--bg-card)', color: 'var(--text-primary)' }}>
            Upload PDB
            <input type="file" accept=".pdb" onChange={handlePdbUpload} className="hidden" />
          </label>
          <label className="px-3 py-1.5 text-[13px] rounded font-medium cursor-pointer transition-opacity hover:opacity-80" style={{ backgroundColor: 'var(--bg-card)', color: 'var(--text-primary)' }}>
            Upload residue map (JSON)
            <input type="file" accept=".json" onChange={handleJsonUpload} className="hidden" />
          </label>
        </div>

        {error && (
          <div className="rounded bg-red-100 px-3 py-2 text-[13px] text-red-800">
            {error}
          </div>
        )}

        {inferenceData && nLayers > 1 && (
          <div className="rounded p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
            <label className="block text-[12px] font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
              Layer {effectiveLayer + 1} / {nLayers}
            </label>
            <input
              type="range"
              min={0}
              max={nLayers - 1}
              value={effectiveLayer}
              onChange={(e) => setSelectedLayer(parseInt(e.target.value, 10))}
              className="w-full max-w-md"
              style={{ accentColor: 'var(--accent)' }}
            />
          </div>
        )}

        <div className="rounded overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
          <div className="flex flex-col lg:flex-row">
            <aside className="lg:w-56 shrink-0 p-4">
              <h2 className="text-[12px] font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Color key</h2>
              {inferenceData && (
                <div className="mb-3">
                  <button
                    type="button"
                    onClick={() => setUncheckedFallbackRank((r) => (r + 1) % 9)}
                    className="px-2 py-1 text-[11px] rounded transition-opacity hover:opacity-80"
                    style={{ backgroundColor: 'var(--bg-soft)', color: 'var(--text-secondary)' }}
                  >
                    {uncheckedFallbackRank === 0
                      ? 'Show next best for unchecked'
                      : `Next best for unchecked (${uncheckedFallbackRank === 1 ? '2nd' : uncheckedFallbackRank === 2 ? '3rd' : `${uncheckedFallbackRank + 1}th`})`}
                  </button>
                  <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
                    Unchecked residues are hidden by default. Click to show their next-best concept.
                  </p>
                </div>
              )}
              <div className="space-y-1.5">
                {Object.keys(CONCEPT_COLORS).map((concept) => (
                  <label
                    key={concept}
                    className="flex items-start gap-2 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={activeConcepts.has(concept)}
                      onChange={() => {
                        const isCheckingBack = !activeConcepts.has(concept)
                        setActiveConcepts((prev) => {
                          const next = new Set(prev)
                          if (next.has(concept)) next.delete(concept)
                          else next.add(concept)
                          return next
                        })
                        if (isCheckingBack) setUncheckedFallbackRank(0)
                      }}
                      className="mt-1 h-3 w-3 shrink-0 rounded-sm accent-[var(--accent)]"
                    />
                    <div className="flex items-center gap-2 min-w-0">
                      <div
                        className="h-3 w-3 shrink-0 rounded-sm"
                        style={{ backgroundColor: CONCEPT_COLORS[concept] ?? '#94a3b8' }}
                      />
                      <div>
                        <span className="text-[12px] font-medium">{concept}</span>
                        <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{CONCEPT_LABELS[concept] ?? ''}</p>
                      </div>
                    </div>
                  </label>
                ))}
              </div>
            </aside>
            <main className="flex-1 min-w-0 p-4 space-y-4">
              {pdb && (
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>View:</span>
                  {(['cartoon', 'stick', 'line', 'ballstick'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setViewMode(mode)}
                      className="px-2 py-1 text-[11px] rounded transition-opacity hover:opacity-80 capitalize"
                      style={{
                        backgroundColor: viewMode === mode ? 'var(--accent)' : 'var(--bg-soft)',
                        color: viewMode === mode ? 'white' : 'var(--text-secondary)',
                      }}
                    >
                      {mode === 'ballstick' ? 'Ball+Stick' : mode}
                    </button>
                  ))}
                </div>
              )}
              <ProteinViewer
                pdb={pdb}
                residueConcepts={residueConceptsForLayer}
                residueProjections={residueProjectionsForLayer}
                onResidueSelect={handleResidueSelect}
                onResidueDeselect={handleResidueDeselect}
                selectedResidue={selectedResidue}
                uncheckedFallbackRank={uncheckedFallbackRank}
                viewMode={viewMode}
                activeConcepts={activeConcepts}
              />
              {selectedResidue && (
                <div className="rounded p-4" style={{ backgroundColor: 'var(--bg-soft)' }}>
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-[12px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                      Residue {selectedResidue.chain.trim() ? `${selectedResidue.chain}:` : ''}{selectedResidue.resi} — {selectedResidue.resn} ({selectedResidue.oneLetter})
                    </h3>
                    <button
                      onClick={() => setSelectedResidue(null)}
                      className="text-[11px] hover:opacity-70"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      Clear
                    </button>
                  </div>
                  <p className="text-[11px] mb-2" style={{ color: 'var(--text-muted)' }}>Scores at layer {effectiveLayer + 1}</p>
                  <div className="space-y-0.5">
                    {Object.entries(selectedResidue.conceptScores)
                      .sort(([, a], [, b]) => b - a)
                      .map(([concept, score]) => (
                        <div key={concept} className="flex items-center justify-between text-[12px]">
                          <span style={{ color: 'var(--text-secondary)' }}>{CONCEPT_LABELS[concept] ?? concept}</span>
                          <span className="font-mono">{score.toFixed(3)}</span>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </main>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
