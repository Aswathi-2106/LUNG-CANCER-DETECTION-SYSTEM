import {
  FilterMode,
  HistoryEntry,
  ModelArchitectureId,
  ModelInfoResponse,
  PredictionResult,
  PreprocessResponse,
  SampleScan,
} from '../types/api';

async function handleJson<T>(res: Response): Promise<T> {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(
      (data as { error?: string }).error || `Request failed (${res.status})`
    );
  }
  return data as T;
}

export async function fetchHealth(): Promise<{
  status: string;
  service: string;
  modelsLoaded: string[];
  trainedAt: string;
}> {
  const res = await fetch('/api/health');
  return handleJson(res);
}

export async function fetchSamples(): Promise<SampleScan[]> {
  const res = await fetch('/api/samples');
  const data = await handleJson<{ samples: SampleScan[] }>(res);
  return data.samples;
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch('/api/model-info');
  return handleJson(res);
}

export async function rerunEvaluation(): Promise<{
  message: string;
  evaluation: ModelInfoResponse;
}> {
  const res = await fetch('/api/evaluate', { method: 'POST' });
  return handleJson(res);
}

export async function preprocessScan(params: {
  file?: File;
  sampleId?: string;
  filterMode: FilterMode;
}): Promise<PreprocessResponse> {
  if (params.file) {
    const fd = new FormData();
    fd.append('image', params.file);
    fd.append('filterMode', params.filterMode);
    const res = await fetch('/api/preprocess', { method: 'POST', body: fd });
    return handleJson(res);
  }
  const res = await fetch('/api/preprocess', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sampleId: params.sampleId,
      filterMode: params.filterMode,
    }),
  });
  return handleJson(res);
}

export async function predictScan(params: {
  file?: File;
  sampleId?: string;
  modelId: ModelArchitectureId;
  filterMode: FilterMode;
}): Promise<PredictionResult> {
  if (params.file) {
    const fd = new FormData();
    fd.append('image', params.file);
    fd.append('modelId', params.modelId);
    fd.append('filterMode', params.filterMode);
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    return handleJson(res);
  }
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sampleId: params.sampleId,
      modelId: params.modelId,
      filterMode: params.filterMode,
    }),
  });
  return handleJson(res);
}

export async function fetchHistory(): Promise<HistoryEntry[]> {
  const res = await fetch('/api/history');
  const data = await handleJson<{ history: HistoryEntry[] }>(res);
  return data.history;
}

export async function deleteHistoryItem(id: string): Promise<void> {
  const res = await fetch(`/api/history/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
  await handleJson(res);
}

export async function clearHistory(): Promise<void> {
  const res = await fetch('/api/history', { method: 'DELETE' });
  await handleJson(res);
}
