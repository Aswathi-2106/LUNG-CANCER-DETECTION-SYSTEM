import { useCallback, useEffect, useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { HomeDashboard } from './pages/HomeDashboard';
import { UploadPage } from './pages/UploadPage';
import { ResultsPage } from './pages/ResultsPage';
import { PerformanceDashboard } from './pages/PerformanceDashboard';
import { HistoryPage } from './pages/HistoryPage';
import {
  clearHistory,
  deleteHistoryItem,
  fetchHistory,
  fetchModelInfo,
  fetchSamples,
  predictScan,
  rerunEvaluation,
} from './services/api';
import {
  FilterMode,
  HistoryEntry,
  ModelArchitectureId,
  ModelInfoResponse,
  PageId,
  PredictionResult,
  SampleScan,
} from './types/api';

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('home');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const [samples, setSamples] = useState<SampleScan[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loadingModelInfo, setLoadingModelInfo] = useState(true);

  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(true);

  const [selectedModel, setSelectedModel] =
    useState<ModelArchitectureId>('resnet50');
  const [filterMode, setFilterMode] = useState<FilterMode>('gaussian');
  const [preselectedSampleId, setPreselectedSampleId] = useState<string | null>(
    null
  );

  const [latestResult, setLatestResult] = useState<PredictionResult | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  const loadInitialData = useCallback(async () => {
    try {
      const [samplesData, infoData, historyData] = await Promise.all([
        fetchSamples(),
        fetchModelInfo(),
        fetchHistory(),
      ]);
      setSamples(samplesData);
      setModelInfo(infoData);
      setSelectedModel(infoData.bestModelId);
      setHistory(historyData);
    } catch (err) {
      console.error('Initialization error:', err);
    } finally {
      setLoadingModelInfo(false);
      setLoadingHistory(false);
    }
  }, []);

  useEffect(() => {
    void loadInitialData();
  }, [loadInitialData]);

  const handleRunPrediction = async (params: {
    file?: File;
    sampleId?: string;
    modelId: ModelArchitectureId;
    filterMode: FilterMode;
  }) => {
    setIsPredicting(true);
    setPredictionError(null);
    try {
      const result = await predictScan(params);
      setLatestResult(result);
      const updatedHistory = await fetchHistory();
      setHistory(updatedHistory);
      setActivePage('results');
    } catch (err) {
      setPredictionError(
        err instanceof Error ? err.message : 'CT scan classification failed.'
      );
    } finally {
      setIsPredicting(false);
    }
  };

  const handleQuickAnalyzeSample = (sampleId: string) => {
    setPreselectedSampleId(sampleId);
    setActivePage('upload');
  };

  const handleRerunEvaluation = async () => {
    await rerunEvaluation();
    const fresh = await fetchModelInfo();
    setModelInfo(fresh);
  };

  const handleDeleteHistoryItem = async (id: string) => {
    await deleteHistoryItem(id);
    setHistory((prev) => prev.filter((item) => item.id !== id));
  };

  const handleClearHistory = async () => {
    await clearHistory();
    setHistory([]);
  };

  const handleInspectHistoryResult = (result: PredictionResult) => {
    setLatestResult(result);
    setActivePage('results');
  };

  return (
    <div className="min-h-screen flex bg-[#F8FAFC] text-slate-900">
      <Sidebar
        activePage={activePage}
        onNavigate={setActivePage}
        hasPrediction={latestResult !== null}
        historyCount={history.length}
        mobileOpen={mobileMenuOpen}
        onCloseMobile={() => setMobileMenuOpen(false)}
      />

      <div className="flex-1 min-w-0 flex flex-col">
        <TopBar
          activePage={activePage}
          onNavigate={setActivePage}
          onOpenMobileMenu={() => setMobileMenuOpen(true)}
        />

        <main className="flex-1 w-full max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8 pt-7">
          {activePage === 'home' && (
            <HomeDashboard
              samples={samples}
              modelInfo={modelInfo}
              onNavigate={setActivePage}
              onQuickAnalyzeSample={handleQuickAnalyzeSample}
            />
          )}

          {activePage === 'upload' && (
            <UploadPage
              samples={samples}
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
              filterMode={filterMode}
              onSelectFilterMode={setFilterMode}
              onRunPrediction={handleRunPrediction}
              isPredicting={isPredicting}
              predictionError={predictionError}
              preselectedSampleId={preselectedSampleId}
            />
          )}

          {activePage === 'results' && (
            <ResultsPage result={latestResult} onNavigate={setActivePage} />
          )}

          {activePage === 'performance' && (
            <PerformanceDashboard
              modelInfo={modelInfo}
              isLoading={loadingModelInfo}
              onRerunEvaluation={handleRerunEvaluation}
            />
          )}

          {activePage === 'history' && (
            <HistoryPage
              history={history}
              isLoading={loadingHistory}
              onDeleteItem={handleDeleteHistoryItem}
              onClearAll={handleClearHistory}
              onInspectResult={handleInspectHistoryResult}
              onNavigate={setActivePage}
            />
          )}
        </main>
      </div>
    </div>
  );
}
