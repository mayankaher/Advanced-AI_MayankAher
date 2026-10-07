import { createContext, useContext, useState, type ReactNode } from 'react'

const QueueContext = createContext<{ waitingCount: number; setWaitingCount: (n: number) => void; clinicians: number; setClinicians: (n: number) => void } | null>(null)

export function QueueProvider({ children }: { children: ReactNode }) {
  const [waitingCount, setWaitingCount] = useState(0)
  const [clinicians, setClinicians] = useState(3)
  return (
    <QueueContext.Provider value={{ waitingCount, setWaitingCount, clinicians, setClinicians }}>
      {children}
    </QueueContext.Provider>
  )
}

export function useQueueMeta() {
  const ctx = useContext(QueueContext)
  if (!ctx) throw new Error('useQueueMeta outside provider')
  return ctx
}
