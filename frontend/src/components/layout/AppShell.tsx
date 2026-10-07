import { NavLink, Outlet, useLocation } from 'react-router-dom'
import {
  Activity,
  Bell,
  BookOpen,
  Brain,
  ChevronLeft,
  FlaskConical,
  LayoutDashboard,
  Moon,
  Search,
  Sun,
  Users,
} from 'lucide-react'
import { useState } from 'react'
import { Logo } from '../brand/Logo'
import { useTheme } from '../../context/ThemeContext'
import { useLearnMode } from '../../context/LearnModeContext'
import { useQueueMeta } from '../../context/QueueContext'

const NAV = [
  {
    group: 'Clinical',
    items: [
      { to: '/', label: 'Triage Desk', icon: Activity },
      { to: '/queue', label: 'ER Queue Board', icon: Users, badge: true },
    ],
  },
  {
    group: 'AI Engine',
    items: [
      { to: '/inference', label: 'Inference Lab', icon: FlaskConical },
      { to: '/network', label: 'Network & Learning', icon: Brain },
      { to: '/decision', label: 'Decision & VOI', icon: LayoutDashboard },
    ],
  },
  {
    group: 'Evaluation',
    items: [
      { to: '/simulation', label: 'Policy Simulation', icon: Activity },
      { to: '/evaluation', label: 'Model Evaluation', icon: LayoutDashboard },
    ],
  },
  {
    group: 'Learn',
    items: [{ to: '/about', label: 'About & Syllabus', icon: BookOpen }],
  },
]

export function AppShell() {
  const [collapsed, setCollapsed] = useState(false)
  const { theme, toggle } = useTheme()
  const { learnMode, toggle: toggleLearn } = useLearnMode()
  const { waitingCount, clinicians } = useQueueMeta()
  const location = useLocation()

  return (
    <div className="min-h-screen flex flex-col bg-[var(--color-canvas)]">
      <header className="sticky top-0 z-30 border-b border-[var(--color-border)] bg-[var(--color-surface)]/95 backdrop-blur px-4 py-2.5 flex items-center gap-4">
        <Logo compact={false} />
        <div className="hidden md:flex flex-1 max-w-md items-center gap-2 rounded-[var(--radius-input)] border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2">
          <Search size={16} className="text-[var(--color-ink-muted)]" aria-hidden />
          <input
            type="search"
            placeholder="Search patient ID, demo case…"
            className="flex-1 bg-transparent text-sm outline-none"
            aria-label="Search"
          />
        </div>
        <div className="flex items-center gap-2 ml-auto flex-wrap justify-end">
          <span className="text-xs font-mono-nums px-2.5 py-1 rounded-full border border-[var(--color-border)] bg-[var(--color-surface-2)]">
            Waiting: {waitingCount} · {clinicians} clinicians
          </span>
          <button
            type="button"
            onClick={toggleLearn}
            className={`text-xs px-2.5 py-1 rounded-full border font-medium ${
              learnMode
                ? 'border-[var(--color-primary)] bg-[var(--color-primary-soft)] text-[var(--color-primary)]'
                : 'border-[var(--color-border)]'
            }`}
          >
            Learn mode {learnMode ? 'on' : 'off'}
          </button>
          <span className="hidden sm:inline text-[10px] px-2 py-1 rounded-full bg-amber-100 dark:bg-amber-950 text-amber-900 dark:text-amber-100 border border-amber-200 dark:border-amber-800">
            Educational prototype · synthetic data
          </span>
          <button type="button" className="p-2 rounded-lg hover:bg-[var(--color-surface-2)]" aria-label="Notifications">
            <Bell size={18} />
          </button>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        <aside
          className={`hidden md:flex flex-col border-r border-[var(--color-border)] bg-[var(--color-surface)] transition-all ${
            collapsed ? 'w-[72px]' : 'w-60'
          }`}
        >
          <div className="p-3">
            <NavLink
              to="/"
              className="flex items-center justify-center gap-2 w-full rounded-[var(--radius-input)] bg-[var(--color-primary)] text-white py-2.5 text-sm font-medium"
            >
              + New Patient
            </NavLink>
          </div>
          <nav className="flex-1 overflow-y-auto px-2 pb-2">
            {NAV.map((section) => (
              <div key={section.group} className="mb-4">
                {!collapsed && (
                  <p className="text-[10px] uppercase tracking-wide text-[var(--color-ink-muted)] px-2 mb-1">{section.group}</p>
                )}
                <ul className="space-y-0.5">
                  {section.items.map(({ to, label, icon: Icon, badge }) => (
                    <li key={to}>
                      <NavLink
                        to={to}
                        end={to === '/'}
                        className={({ isActive }) =>
                          `flex items-center gap-2.5 px-2.5 py-2 rounded-[var(--radius-input)] text-sm ${
                            isActive
                              ? 'bg-[var(--color-primary-soft)] text-[var(--color-primary)] font-medium'
                              : 'text-[var(--color-ink-muted)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-ink)]'
                          }`
                        }
                      >
                        <Icon size={18} aria-hidden />
                        {!collapsed && <span className="flex-1">{label}</span>}
                        {!collapsed && badge && waitingCount > 0 && (
                          <span className="text-[10px] font-mono-nums px-1.5 rounded-full bg-[var(--color-p2-soft)] text-[var(--color-p2)]">
                            {waitingCount}
                          </span>
                        )}
                      </NavLink>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </nav>
          <div className="p-3 border-t border-[var(--color-border)] space-y-2">
            <button
              type="button"
              onClick={toggle}
              className="flex w-full items-center gap-2 text-sm text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] px-2 py-1.5"
            >
              {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
              {!collapsed && (theme === 'dark' ? 'Light mode' : 'Dark mode')}
            </button>
            <button
              type="button"
              onClick={() => setCollapsed((c) => !c)}
              className="flex w-full items-center gap-2 text-xs text-[var(--color-ink-muted)] px-2"
            >
              <ChevronLeft size={14} className={collapsed ? 'rotate-180' : ''} />
              {!collapsed && 'Collapse'}
            </button>
          </div>
        </aside>

        <main className="flex-1 overflow-y-auto">
          <div className="md:hidden flex gap-1 overflow-x-auto p-2 border-b border-[var(--color-border)] bg-[var(--color-surface)]">
            {NAV.flatMap((s) => s.items).map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `whitespace-nowrap text-xs px-3 py-1.5 rounded-full ${isActive ? 'bg-[var(--color-primary)] text-white' : 'bg-[var(--color-surface-2)]'}`
                }
              >
                {label}
              </NavLink>
            ))}
          </div>
          <div className="p-6 max-w-[1440px] mx-auto">
            <Outlet key={location.pathname} />
          </div>
        </main>
      </div>

      <footer className="border-t border-[var(--color-border)] bg-[var(--color-surface-2)] px-4 py-2 text-center text-xs text-[var(--color-ink-muted)]">
        Decision support only. A clinician makes the final decision. Not clinically validated.
      </footer>
    </div>
  )
}
