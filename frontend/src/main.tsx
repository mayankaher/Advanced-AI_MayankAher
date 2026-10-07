import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from './context/ThemeContext'
import { LearnModeProvider } from './context/LearnModeContext'
import { QueueProvider } from './context/QueueContext'
import './index.css'
import App from './App.tsx'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, retry: 1 },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <LearnModeProvider>
          <QueueProvider>
            <App />
          </QueueProvider>
        </LearnModeProvider>
      </ThemeProvider>
    </QueryClientProvider>
  </StrictMode>,
)
