import { useState, useCallback, useMemo } from 'react'
import { ProteinViewer, CONCEPT_COLORS } from './ProteinViewer'
import { parsePdbSequence } from './pdbParser'

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
  const [inferenceData, setInferenceData] = useState<InferenceResponse | null>(null)
  const [seqIdxToResi, setSeqIdxToResi] = useState<number[]>([])
  const [residueConcepts, setResidueConcepts] = useState<ResidueConcepts>({})
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [selectedResidue, setSelectedResidue] = useState<SelectedResidue | null>(null)
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
        setPdb(text)
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
      const resi = seqIdxToResi[seqIdx]
      if (resi == null) continue

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

      outConcepts[String(resi)] = maxConcept
      outProjections[String(resi)] = scores
    }

    return { residueConceptsForLayer: outConcepts, residueProjectionsForLayer: outProjections }
  }, [inferenceData, seqIdxToResi, selectedLayer, residueConcepts])

  const handleResidueSelect = useCallback(
    (resi: number, resn: string, oneLetter: string, conceptScores: Record<string, number>) => {
      setSelectedResidue({ resi, resn, oneLetter, conceptScores })
    },
    []
  )

  const effectiveLayer = inferenceData ? Math.min(selectedLayer, inferenceData.n_layers - 1) : 0

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      <div className="mx-auto max-w-6xl space-y-6">
        <h1 className="text-3xl font-bold">Protein Concept Vector Viewer</h1>
        <p className="text-gray-400">
          Visualize residues colored by their dominant concept. Click a residue to see amino acid and similarity scores.
        </p>

        <div className="flex flex-wrap gap-4">
          <button
            onClick={loadDemo}
            disabled={loading}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded-lg font-medium transition-colors"
          >
            {loading ? 'Loading...' : 'Load demo (1CRN)'}
          </button>
          <label className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium cursor-pointer transition-colors">
            Upload PDB
            <input type="file" accept=".pdb" onChange={handlePdbUpload} className="hidden" />
          </label>
          <label className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium cursor-pointer transition-colors">
            Upload residue map (JSON)
            <input type="file" accept=".json" onChange={handleJsonUpload} className="hidden" />
          </label>
        </div>

        {error && (
          <div className="rounded-lg bg-red-900/50 border border-red-700 px-4 py-2 text-red-200">
            {error}
          </div>
        )}

        {inferenceData && nLayers > 1 && (
          <div className="rounded-lg border border-gray-700 bg-gray-900/50 p-4">
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              Layer {effectiveLayer + 1} / {nLayers}
            </label>
            <input
              type="range"
              min={0}
              max={nLayers - 1}
              value={effectiveLayer}
              onChange={(e) => setSelectedLayer(parseInt(e.target.value, 10))}
              className="w-full max-w-md accent-indigo-500"
            />
          </div>
        )}

        <div className="rounded-xl border border-gray-700 bg-gray-900/50 overflow-hidden shadow-xl">
          <div className="flex flex-col lg:flex-row">
            <aside className="lg:w-64 shrink-0 border-b lg:border-b-0 lg:border-r border-gray-700 p-4 bg-gray-900/80">
              <h2 className="text-sm font-semibold text-gray-300 mb-3">Color key</h2>
              <div className="space-y-2">
                {Object.keys(CONCEPT_COLORS).map((concept) => (
                  <div key={concept} className="flex items-start gap-2">
                    <div
                      className="mt-0.5 h-4 w-4 shrink-0 rounded"
                      style={{ backgroundColor: CONCEPT_COLORS[concept] ?? '#94a3b8' }}
                    />
                    <div>
                      <span className="text-sm font-medium text-white">{concept}</span>
                      <p className="text-xs text-gray-500">{CONCEPT_LABELS[concept] ?? ''}</p>
                    </div>
                  </div>
                ))}
              </div>
            </aside>
            <main className="flex-1 min-w-0 p-4 bg-gray-900/30 space-y-4">
              <ProteinViewer
                pdb={pdb}
                residueConcepts={residueConceptsForLayer}
                residueProjections={residueProjectionsForLayer}
                onResidueSelect={handleResidueSelect}
              />
              {selectedResidue && (
                <div className="rounded-lg border border-gray-600 bg-gray-800/50 p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-gray-300">
                      Residue {selectedResidue.resi}: {selectedResidue.resn} ({selectedResidue.oneLetter})
                    </h3>
                    <button
                      onClick={() => setSelectedResidue(null)}
                      className="text-xs text-gray-500 hover:text-white"
                    >
                      Clear
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mb-2">Scores at layer {effectiveLayer + 1}</p>
                  <div className="space-y-1">
                    {Object.entries(selectedResidue.conceptScores)
                      .sort(([, a], [, b]) => b - a)
                      .map(([concept, score]) => (
                        <div key={concept} className="flex items-center justify-between text-sm">
                          <span className="text-gray-300">{concept}</span>
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
