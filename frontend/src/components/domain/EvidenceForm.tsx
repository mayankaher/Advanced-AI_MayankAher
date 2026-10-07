import type { AssessmentState, VitalField } from '../../lib/evidence-form'
import { EVIDENCE_FIELD_COUNT, measuredCount } from '../../lib/evidence-form'
import { ageBandFromYears, discretizeHeartRate, discretizeRespRate, discretizeSpO2, discretizeSystolicBP, discretizeTemperature } from '../../lib/vitals'
import type { ModelResponse } from '../../lib/api'

function VitalRow({
  label,
  unit,
  field,
  discretize,
  onChange,
}: {
  label: string
  unit: string
  field: VitalField
  discretize: (v: number) => string
  onChange: (f: VitalField) => void
}) {
  const band = field.measured && field.value != null ? discretize(field.value) : null
  return (
    <div className="grid grid-cols-[1fr_auto] gap-2 items-end border-b border-[var(--color-border)] pb-3 last:border-0">
      <div>
        <label className="text-xs font-medium text-[var(--color-ink-muted)]">{label}</label>
        <div className="flex items-baseline gap-2 mt-1">
          <input
            type="number"
            disabled={!field.measured}
            value={field.value ?? ''}
            onChange={(e) =>
              onChange({ ...field, value: e.target.value === '' ? null : Number(e.target.value) })
            }
            className="w-24 text-2xl font-semibold tabular-nums rounded-[var(--radius-input)] border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1 disabled:opacity-40"
          />
          <span className="text-sm text-[var(--color-ink-muted)]">{unit}</span>
          {band && (
            <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-[var(--color-surface-2)] border border-[var(--color-border)]">
              {band}
            </span>
          )}
        </div>
      </div>
      <label className="flex items-center gap-1.5 text-xs text-[var(--color-ink-muted)] cursor-pointer">
        <input
          type="checkbox"
          checked={!field.measured}
          onChange={(e) => onChange({ measured: !e.target.checked, value: e.target.checked ? null : field.value })}
        />
        Not measured
      </label>
    </div>
  )
}

export function EvidenceForm({
  state,
  onChange,
  model,
}: {
  state: AssessmentState
  onChange: (s: AssessmentState) => void
  model?: ModelResponse
}) {
  const measured = measuredCount(state)
  const pct = Math.round((measured / EVIDENCE_FIELD_COUNT) * 100)
  const historyStates = model?.nodes.History?.states ?? ['None', 'Cardiac', 'Respiratory', 'Metabolic']
  const consciousnessStates = model?.nodes.Consciousness?.states ?? ['Alert', 'Altered', 'Unresponsive']
  const ageBand = state.ageYears != null ? ageBandFromYears(state.ageYears) : null

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-4">
        <div
          className="relative h-14 w-14 rounded-full flex items-center justify-center border-2 border-[var(--color-primary)]"
          style={{
            background: `conic-gradient(var(--color-primary) ${pct}%, var(--color-surface-2) 0)`,
          }}
          aria-label={`Evidence completeness ${measured} of ${EVIDENCE_FIELD_COUNT}`}
        >
          <div className="h-10 w-10 rounded-full bg-[var(--color-surface)] flex items-center justify-center text-xs font-mono-nums font-semibold">
            {measured}/{EVIDENCE_FIELD_COUNT}
          </div>
        </div>
        <div>
          <p className="text-sm font-medium">Evidence completeness</p>
          <p className="text-xs text-[var(--color-ink-muted)]">{pct}% of fields measured</p>
        </div>
      </div>

      <section>
        <h3 className="text-xs uppercase tracking-wide text-[var(--color-ink-muted)] mb-2">Demographics</h3>
        <div className="flex flex-wrap gap-3 items-end">
          <label className="text-sm">
            <span className="text-[var(--color-ink-muted)] text-xs">Age (years)</span>
            <input
              type="number"
              min={0}
              max={120}
              value={state.ageYears ?? ''}
              onChange={(e) =>
                onChange({ ...state, ageYears: e.target.value === '' ? null : Number(e.target.value) })
              }
              className="block mt-1 w-28 rounded-[var(--radius-input)] border border-[var(--color-border)] px-3 py-2 tabular-nums"
            />
          </label>
          {ageBand && (
            <span className="text-xs px-2 py-1 rounded-full bg-[var(--color-primary-soft)] text-[var(--color-primary)] font-medium">
              {ageBand}
            </span>
          )}
          <label className="text-sm flex-1 min-w-[10rem]">
            <span className="text-[var(--color-ink-muted)] text-xs">History</span>
            <select
              value={state.history ?? ''}
              onChange={(e) => onChange({ ...state, history: e.target.value || null })}
              className="block mt-1 w-full rounded-[var(--radius-input)] border border-[var(--color-border)] px-3 py-2 bg-[var(--color-surface)]"
            >
              <option value="">Not measured</option>
              {historyStates.map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </label>
        </div>
      </section>

      <section>
        <h3 className="text-xs uppercase tracking-wide text-[var(--color-ink-muted)] mb-2">Symptoms</h3>
        <div className="flex flex-wrap gap-2">
          {(
            [
              ['chestPain', 'Chest pain'],
              ['dyspnea', 'Dyspnea'],
              ['visibleInjury', 'Visible injury'],
            ] as const
          ).map(([key, label]) => {
            const val = state[key]
            return (
              <div key={key} className="flex rounded-full border border-[var(--color-border)] overflow-hidden text-sm">
                <button
                  type="button"
                  className={`px-3 py-1.5 ${val === true ? 'bg-[var(--color-primary)] text-white' : 'bg-[var(--color-surface)]'}`}
                  onClick={() => onChange({ ...state, [key]: true })}
                >
                  {label}: Yes
                </button>
                <button
                  type="button"
                  className={`px-3 py-1.5 border-l border-[var(--color-border)] ${val === false ? 'bg-[var(--color-surface-2)] font-medium' : ''}`}
                  onClick={() => onChange({ ...state, [key]: false })}
                >
                  No
                </button>
                <button
                  type="button"
                  className={`px-2 py-1.5 border-l border-[var(--color-border)] text-xs text-[var(--color-ink-muted)] ${val === null ? 'bg-[var(--color-surface-2)]' : ''}`}
                  onClick={() => onChange({ ...state, [key]: null })}
                >
                  —
                </button>
              </div>
            )
          })}
        </div>
      </section>

      <section className="space-y-3">
        <h3 className="text-xs uppercase tracking-wide text-[var(--color-ink-muted)]">Vitals</h3>
        <VitalRow
          label="Heart rate"
          unit="bpm"
          field={state.heartRate}
          discretize={discretizeHeartRate}
          onChange={(f) => onChange({ ...state, heartRate: f })}
        />
        <VitalRow
          label="Systolic BP"
          unit="mmHg"
          field={state.systolicBP}
          discretize={discretizeSystolicBP}
          onChange={(f) => onChange({ ...state, systolicBP: f })}
        />
        <VitalRow label="SpO₂" unit="%" field={state.spO2} discretize={discretizeSpO2} onChange={(f) => onChange({ ...state, spO2: f })} />
        <VitalRow
          label="Resp rate"
          unit="/min"
          field={state.respRate}
          discretize={discretizeRespRate}
          onChange={(f) => onChange({ ...state, respRate: f })}
        />
        <VitalRow
          label="Temperature"
          unit="°C"
          field={state.temperature}
          discretize={discretizeTemperature}
          onChange={(f) => onChange({ ...state, temperature: f })}
        />
        <label className="block text-sm">
          <span className="text-xs text-[var(--color-ink-muted)]">Consciousness</span>
          <select
            value={state.consciousness ?? ''}
            onChange={(e) => onChange({ ...state, consciousness: e.target.value || null })}
            className="block mt-1 w-full rounded-[var(--radius-input)] border border-[var(--color-border)] px-3 py-2"
          >
            <option value="">Not measured</option>
            {consciousnessStates.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section>
        <h3 className="text-xs uppercase tracking-wide text-[var(--color-ink-muted)] mb-2">Arrival</h3>
        <div className="inline-flex rounded-[var(--radius-input)] border border-[var(--color-border)] overflow-hidden">
          {(['WalkIn', 'Ambulance'] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              className={`px-4 py-2 text-sm ${state.arrivalMode === mode ? 'bg-[var(--color-primary)] text-white' : 'bg-[var(--color-surface)]'}`}
              onClick={() => onChange({ ...state, arrivalMode: mode })}
            >
              {mode === 'WalkIn' ? 'Walk-in' : 'Ambulance'}
            </button>
          ))}
          <button
            type="button"
            className={`px-3 py-2 text-xs border-l border-[var(--color-border)] ${state.arrivalMode === null ? 'bg-[var(--color-surface-2)]' : ''}`}
            onClick={() => onChange({ ...state, arrivalMode: null })}
          >
            Clear
          </button>
        </div>
      </section>
    </div>
  )
}
