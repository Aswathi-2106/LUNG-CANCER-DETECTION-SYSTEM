import { Menu } from 'lucide-react';
import { PageId } from '../types/api';

interface TopBarProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
  onOpenMobileMenu: () => void;
}

const PAGE_TITLES: Record<PageId, string> = {
  home: 'Overview',
  upload: 'CT Scan Upload',
  results: 'Prediction Results',
  performance: 'Model Performance',
  history: 'Prediction History',
};

export function TopBar({ activePage, onNavigate, onOpenMobileMenu }: TopBarProps) {
  return (
    <header className="sticky top-0 z-30 h-14 bg-white border-b border-slate-200 px-4 lg:px-8 flex items-center justify-between gap-8">
      <div className="flex items-center gap-3 min-w-0">
        <button
          type="button"
          onClick={onOpenMobileMenu}
          className="lg:hidden p-2 -ml-1 text-slate-600 hover:text-slate-900 rounded-md focus-visible:outline-2 focus-visible:outline-sky-500"
          aria-label="Open navigation menu"
        >
          <Menu className="h-5 w-5" />
        </button>
        <span className="font-display text-sm font-semibold text-slate-900 whitespace-nowrap shrink-0">
          {PAGE_TITLES[activePage]}
        </span>
      </div>

      <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600">
        <button
          type="button"
          onClick={() => onNavigate('home')}
          className={`whitespace-nowrap shrink-0 transition-colors hover:text-slate-900 ${
            activePage === 'home' ? 'text-sky-700 underline underline-offset-8 decoration-2' : ''
          }`}
        >
          Overview
        </button>
        <button
          type="button"
          onClick={() => onNavigate('upload')}
          className={`whitespace-nowrap shrink-0 transition-colors hover:text-slate-900 ${
            activePage === 'upload' ? 'text-sky-700 underline underline-offset-8 decoration-2' : ''
          }`}
        >
          Upload Scan
        </button>
        <button
          type="button"
          onClick={() => onNavigate('results')}
          className={`whitespace-nowrap shrink-0 transition-colors hover:text-slate-900 ${
            activePage === 'results' ? 'text-sky-700 underline underline-offset-8 decoration-2' : ''
          }`}
        >
          Results
        </button>
        <button
          type="button"
          onClick={() => onNavigate('performance')}
          className={`whitespace-nowrap shrink-0 transition-colors hover:text-slate-900 ${
            activePage === 'performance' ? 'text-sky-700 underline underline-offset-8 decoration-2' : ''
          }`}
        >
          Evaluation
        </button>
        <button
          type="button"
          onClick={() => onNavigate('history')}
          className={`whitespace-nowrap shrink-0 transition-colors hover:text-slate-900 ${
            activePage === 'history' ? 'text-sky-700 underline underline-offset-8 decoration-2' : ''
          }`}
        >
          History
        </button>
      </nav>

      <div className="shrink-0">
        <button
          type="button"
          onClick={() => onNavigate('upload')}
          className="px-4 py-2 text-xs font-semibold text-white bg-[#0B1324] hover:bg-slate-800 rounded-lg transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-sky-500"
        >
          Start Analysis
        </button>
      </div>
    </header>
  );
}
