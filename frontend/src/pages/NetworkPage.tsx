import { useCallback, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ReactFlow, Background, Controls, MiniMap, type Node, type Edge } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { api } from '../lib/api'
import { Card } from '../components/ui/Card'

const ROLE_COLOR: Record<string, string> = {
  Age: '#2F6FED',
  History: '#2F6FED',
  Condition: '#7C5CFC',
  Severity: '#7C5CFC',
  ChestPain: '#14B8A6',
  Dyspnea: '#14B8A6',
  VisibleInjury: '#14B8A6',
  ECG_Abnormal: '#94A3B8',
  Troponin_High: '#94A3B8',
  ChestXray_Abnormal: '#94A3B8',
  Lactate_High: '#94A3B8',
}

function nodeColor(name: string): string {
  if (ROLE_COLOR[name]) return ROLE_COLOR[name]
  if (['HeartRate', 'SystolicBP', 'SpO2', 'RespRate', 'Temperature', 'Consciousness'].includes(name)) return '#F76B15'
  if (name === 'ArrivalMode') return '#30A46C'
  return '#5B6B82'
}

export default function NetworkPage() {
  const modelQuery = useQuery({ queryKey: ['model'], queryFn: api.getModel })

  const { nodes, edges } = useMemo(() => {
    const order = modelQuery.data?.topological_order ?? []
    const edgeList: Edge[] = (modelQuery.data?.edges ?? []).map((e, i) => ({
      id: `e${i}`,
      source: e.source,
      target: e.target,
      animated: false,
      style: { stroke: 'var(--color-border)' },
    }))
    const flowNodes: Node[] = order.map((id, i) => {
      const col = i % 4
      const row = Math.floor(i / 4)
      return {
        id,
        data: { label: id },
        position: { x: col * 180, y: row * 90 },
        style: {
          background: nodeColor(id),
          color: '#fff',
          border: 'none',
          borderRadius: 12,
          fontSize: 11,
          padding: '8px 12px',
        },
      }
    })
    return { nodes: flowNodes, edges: edgeList }
  }, [modelQuery.data])

  const onInit = useCallback(() => {}, [])

  return (
    <div className="space-y-6">
      <header>
        <p className="text-xs text-[var(--color-ink-muted)]">AI Engine · Network</p>
        <h1 className="text-2xl font-bold mt-1">Bayesian network structure</h1>
        <p className="text-sm text-[var(--color-ink-muted)]">Loaded from `/api/model`. Learning tabs will extend here per syllabus.</p>
      </header>
      <Card className="!p-0 overflow-hidden">
        <div className="h-[520px]">
          {modelQuery.isLoading ? (
            <p className="p-6 text-sm">Loading graph…</p>
          ) : (
            <ReactFlow nodes={nodes} edges={edges} onInit={onInit} fitView>
              <Background />
              <MiniMap />
              <Controls />
            </ReactFlow>
          )}
        </div>
      </Card>
      <p className="text-xs text-[var(--color-ink-muted)]">
        Node colors: demographic (blue), latent (purple), symptoms (teal), vitals (orange), tests (gray).
      </p>
    </div>
  )
}
