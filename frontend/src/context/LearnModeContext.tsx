import { createContext, useContext, useState, type ReactNode } from 'react'

const LearnModeContext = createContext<{ learnMode: boolean; toggle: () => void } | null>(null)

export function LearnModeProvider({ children }: { children: ReactNode }) {
  const [learnMode, setLearnMode] = useState(false)
  return (
    <LearnModeContext.Provider value={{ learnMode, toggle: () => setLearnMode((v) => !v) }}>
      {children}
    </LearnModeContext.Provider>
  )
}

export function useLearnMode() {
  const ctx = useContext(LearnModeContext)
  if (!ctx) throw new Error('useLearnMode outside provider')
  return ctx
}
