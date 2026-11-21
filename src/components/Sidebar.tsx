import {
  Activity,
  BarChart3,
  FileSearch,
  History,
  LayoutDashboard,
  Upload,
  X,
} from 'lucide-react';
import { PageId } from '../types/api';

interface SidebarProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
  hasPrediction: boolean;
  historyCount: number;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

const NAV_ITEMS: {
  id: PageId;
  label: string;
  icon: typeof LayoutDashboard;
}[] = [
  { id: 'home', label: 'Overview', icon: LayoutDashboard },
  { id: 'upload', label: 'CT Scan Upload', icon: Upload },
  { id: 'results', label: 'Prediction Results', icon: FileSearch },
  { id: 'performance', label: 'Model Performance', icon: BarChart3 },
  { id: 'history', label: 'Prediction History', icon: History },
];

export function Sidebar({
  activePage,
  onNavigate,
  hasPrediction,
  historyCount,
  mobileOpen,
  onCloseMobile,
}: SidebarProps) {
  const content = (
    <div className="flex h-full flex-col bg-[#0B1324] text-slate-100 border-r border-slate-800/80">
      <div className="flex h-14 items-center justify-between px-5 border-b border-slate-800/80">
        <button
          type="button"
          onClick={() => {
            onNavigate('home');
            onCloseMobile();
          }}
          className="flex items-center gap-2.5 text-left focus-visible:outline-2 focus-visible:outline-sky-400 rounded"
        >
          <Activity className="h-5 w-5 text-sky-400 shrink-0" aria-hidden="true" />
          <span className="font-display text-base font-semibold tracking-tight text-white whitespace-nowrap">
            PulmoVision AI
          </span>
        </button>
        <button
          type="button"
          onClick={onCloseMobile}
          className="lg:hidden p-1.5 text-slate-400 hover:text-white rounded-md focus-visible:outline-2 focus-visible:outline-sky-400"
          aria-label="Close navigation menu"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      <nav className="flex-1 px-3 py-5 space-y-1" aria-label="Primary Workspace Navigation">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                onNavigate(item.id);
                onCloseMobile();
              }}
              className={`w-full flex items-center justify-between gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors duration-150 whitespace-nowrap focus-visible:outline-2 focus-visible:outline-sky-400 ${
                isActive
                  ? 'bg-sky-500/15 text-sky-300'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <span className="flex items-center gap-3 min-w-0">
                <Icon
                  className={`h-4 w-4 shrink-0 ${
                    isActive ? 'text-sky-400' : 'text-slate-400'
                  }`}
                  aria-hidden="true"
                />
                <span className="truncate">{item.label}</span>
              </span>
              {item.id === 'results' && hasPrediction && (
                <span className="text-xs font-mono text-sky-300">Ready</span>
              )}
              {item.id === 'history' && historyCount > 0 && (
                <span className="text-xs font-mono tabular-nums text-slate-400">
                  {historyCount}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="px-5 py-4 border-t border-slate-800/80 text-xs leading-relaxed text-slate-400">
        <p className="font-medium text-slate-200 mb-1">Academic Research Prototype</p>
        <p>
          Computer Vision, OpenCV, CNN & Transfer Learning system. Not validated
          for clinical diagnosis or patient treatment.
        </p>
      </div>
    </div>
  );

  return (
    <>
      <aside className="hidden lg:block w-64 shrink-0 h-screen sticky top-0">
        {content}
      </aside>
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-slate-950/60"
            onClick={onCloseMobile}
            aria-hidden="true"
          />
          <div className="relative w-64 max-w-[80vw] h-full z-10">{content}</div>
        </div>
      )}
    </>
  );
}
