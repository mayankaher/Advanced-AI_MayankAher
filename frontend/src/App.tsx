import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import TriageDeskPage from './pages/TriageDeskPage'
import QueueBoardPage from './pages/QueueBoardPage'
import InferenceLabPage from './pages/InferenceLabPage'
import NetworkPage from './pages/NetworkPage'
import DecisionPage from './pages/DecisionPage'
import { SimulationPage, EvaluationPage, AboutPage } from './pages/PlaceholderPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<TriageDeskPage />} />
          <Route path="queue" element={<QueueBoardPage />} />
          <Route path="inference" element={<InferenceLabPage />} />
          <Route path="network" element={<NetworkPage />} />
          <Route path="decision" element={<DecisionPage />} />
          <Route path="simulation" element={<SimulationPage />} />
          <Route path="evaluation" element={<EvaluationPage />} />
          <Route path="about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
