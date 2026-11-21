import { useState } from 'react';
import { Eye, History, Search, Trash2, Upload } from 'lucide-react';
import { HistoryEntry, LungClass, PageId, PredictionResult } from '../types/api';

interface HistoryPageProps {
  history: HistoryEntry[];
  isLoading: boolean;
  onDeleteItem: (id: string) => Promise<void>;
  onClearAll: () => Promise<void>;
  onInspectResult: (result: PredictionResult) => void;
  onNavigate: (page: PageId) => void;
}

export function HistoryPage({
  history,
  isLoading,
  onDeleteItem,
  onClearAll,
  onInspectResult,
  onNavigate,
}: HistoryPageProps) {
  const [filterClass, setFilterClass] = useState<'All' | LungClass>('All');
  const [query, setQuery] = useState('');

  const filtered = history.filter((item) => {
    const matchesClass =
      filterClass === 'All' || item.predictedClass === filterClass;
    const q = query.trim().toLowerCase();
    const matchesQuery =
      !q ||
      item.scanLabel.toLowerCase().includes(q) ||
      item.modelDisplayName.toLowerCase().includes(q) ||
      item.predictedClass.toLowerCase().includes(q);
    return matchesClass && matchesQuery;
  });

  const handleInspect = (entry: HistoryEntry) => {
    const reconstructed: PredictionResult = {
      id: entry.id,
      timestamp: entry.timestamp,
      scanLabel: entry.scanLabel,
      modelUsed: entry.modelUsed,
      modelDisplayName: entry.modelDisplayName,
      predictedClass: entry.predictedClass,
      confidence: entry.confidence,
      probabilities: entry.probabilities,
      metadata: entry.metadata,
      features: entry.features,
      featureVector: [],
      segmentation: entry.segmentation,
      visualizations: {
        originalPngDataUrl: entry.thumbnailDataUrl,
        denoisedPngDataUrl: entry.preprocessedDataUrl,
        clahePngDataUrl: entry.preprocessedDataUrl,
        edgesPngDataUrl: entry.preprocessedDataUrl,
        roiMaskPngDataUrl: entry.segmentationOverlayDataUrl,
        gradcamOverlayPngDataUrl: entry.gradcamOverlayDataUrl,
        gradcamHeatmapPngDataUrl: entry.gradcamOverlayDataUrl,
        segmentationOverlayPngDataUrl: entry.segmentationOverlayDataUrl,
      },
      disclaimer:
        'Academic research output only. Grad-CAM visualizes spatial feature attention and does not confirm the clinical presence or exact anatomical boundary of malignancy.',
    };
    onInspectResult(reconstructed);
  };

  return (
    <div className="space-y-6 pb-12">
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="font-display text-2xl font-semibold text-slate-900">
            Anonymized Prediction History
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            Stored scan thumbnails, predicted categories, and Grad-CAM overlays. Patient
            identifiers are stripped prior to persistence.
          </p>
        </div>

        {history.length > 0 && (
          <button
            type="button"
            onClick={() => void onClearAll()}
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-semibold text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Clear All History</span>
          </button>
        )}
      </div>

      {/* Filter & Search Toolbar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg w-fit">
          {(['All', 'Normal', 'Benign', 'Malignant'] as const).map((cls) => (
            <button
              key={cls}
              type="button"
              onClick={() => setFilterClass(cls)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                filterClass === cls
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {cls}
            </button>
          ))}
        </div>

        <div className="relative w-full sm:w-64">
          <Search
            className="h-4 w-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2"
            aria-hidden="true"
          />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by scan or model..."
            aria-label="Filter prediction history"
            className="w-full pl-9 pr-3 py-2 text-xs bg-white border border-slate-200 rounded-lg text-slate-900 placeholder:text-slate-400 focus:outline-2 focus:outline-sky-500"
          />
        </div>
      </div>

      {isLoading ? (
        <div className="py-16 text-center text-sm text-slate-500">
          Loading prediction history...
        </div>
      ) : history.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-xl p-12 text-center space-y-4">
          <History className="h-10 w-10 text-slate-400 mx-auto" aria-hidden="true" />
          <h2 className="font-display text-lg font-semibold text-slate-900">
            No Predictions Recorded
          </h2>
          <p className="text-sm text-slate-600 max-w-md mx-auto">
            Analyze a thoracic CT scan on the upload page to record classification
            results, Grad-CAM maps, and timestamps here.
          </p>
          <button
            type="button"
            onClick={() => onNavigate('upload')}
            className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B1324] hover:bg-slate-800 text-white text-xs font-semibold rounded-lg transition-colors"
          >
            <Upload className="h-4 w-4" aria-hidden="true" />
            <span>Analyze First CT Scan</span>
          </button>
        </div>
      ) : filtered.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-xl p-8 text-center text-sm text-slate-500">
          No prediction records match your current filter.
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-slate-500">
                  <th className="py-3 px-4 font-medium">CT / Grad-CAM</th>
                  <th className="py-3 px-4 font-medium">Scan Label</th>
                  <th className="py-3 px-4 font-medium">Category</th>
                  <th className="py-3 px-4 text-right font-medium">Confidence</th>
                  <th className="py-3 px-4 font-medium">Model</th>
                  <th className="py-3 px-4 font-medium">Timestamp</th>
                  <th className="py-3 px-4 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filtered.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50/80">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <img
                          src={item.thumbnailDataUrl}
                          alt={item.scanLabel}
                          referrerPolicy="no-referrer"
                          className="h-11 w-11 rounded object-cover bg-black border border-slate-200"
                        />
                        <img
                          src={item.gradcamOverlayDataUrl}
                          alt={`${item.scanLabel} Grad-CAM`}
                          referrerPolicy="no-referrer"
                          className="h-11 w-11 rounded object-cover bg-black border border-slate-200"
                        />
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <p className="font-medium text-slate-900">{item.scanLabel}</p>
                      <p className="text-[11px] font-mono text-slate-500">
                        {item.metadata.format} · {item.metadata.originalWidth}×
                        {item.metadata.originalHeight}
                      </p>
                    </td>
                    <td className="py-3 px-4 font-semibold">
                      <span
                        className={
                          item.predictedClass === 'Malignant'
                            ? 'text-rose-700'
                            : item.predictedClass === 'Benign'
                            ? 'text-teal-700'
                            : 'text-sky-700'
                        }
                      >
                        {item.predictedClass}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-mono tabular-nums font-semibold text-slate-900">
                      {(item.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4 text-slate-600">
                      {item.modelDisplayName.split(' (')[0]}
                    </td>
                    <td className="py-3 px-4 font-mono tabular-nums text-slate-500 whitespace-nowrap">
                      {new Date(item.timestamp).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="inline-flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => handleInspect(item)}
                          className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-sky-700 hover:bg-sky-50 rounded-md transition-colors whitespace-nowrap"
                        >
                          <Eye className="h-3.5 w-3.5" aria-hidden="true" />
                          <span>Inspect</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => void onDeleteItem(item.id)}
                          aria-label={`Delete ${item.scanLabel}`}
                          className="p-1.5 text-slate-400 hover:text-rose-700 hover:bg-rose-50 rounded-md transition-colors"
                        >
                          <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
