import { ChangeEvent, DragEvent, useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  ImagePlus,
  Loader2,
  RefreshCw,
  SlidersHorizontal,
  Trash2,
} from 'lucide-react';
import { preprocessScan } from '../services/api';
import {
  FilterMode,
  ModelArchitectureId,
  PreprocessResponse,
  SampleScan,
} from '../types/api';

interface UploadPageProps {
  samples: SampleScan[];
  selectedModel: ModelArchitectureId;
  onSelectModel: (model: ModelArchitectureId) => void;
  filterMode: FilterMode;
  onSelectFilterMode: (mode: FilterMode) => void;
  onRunPrediction: (params: {
    file?: File;
    sampleId?: string;
    modelId: ModelArchitectureId;
    filterMode: FilterMode;
  }) => Promise<void>;
  isPredicting: boolean;
  predictionError: string | null;
  preselectedSampleId?: string | null;
}

const MODEL_OPTIONS: {
  id: ModelArchitectureId;
  name: string;
  note: string;
}[] = [
  {
    id: 'resnet50',
    name: 'ResNet50 Transfer Learning',
    note: 'Highest test accuracy & macro AUC on patient-level split',
  },
  {
    id: 'efficientnet_b0',
    name: 'EfficientNetB0 Transfer Learning',
    note: 'Compound-scaled MBConv architecture with [-1, 1] normalization',
  },
  {
    id: 'vgg16',
    name: 'VGG16 Transfer Learning',
    note: 'Deep 3x3 convolutional stack with ImageNet Caffe normalization',
  },
  {
    id: 'custom_cnn',
    name: 'Custom 4-Block CNN',
    note: 'Baseline convolutional model with BatchNorm, Dropout, and GAP',
  },
];

const MAX_BYTES = 15 * 1024 * 1024;

export function UploadPage({
  samples,
  selectedModel,
  onSelectModel,
  filterMode,
  onSelectFilterMode,
  onRunPrediction,
  isPredicting,
  predictionError,
  preselectedSampleId,
}: UploadPageProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(
    preselectedSampleId || null
  );
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewLabel, setPreviewLabel] = useState<string>('');
  const [localError, setLocalError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const [prepPreview, setPrepPreview] = useState<PreprocessResponse | null>(null);
  const [isPreprocessing, setIsPreprocessing] = useState(false);

  useEffect(() => {
    if (preselectedSampleId && samples.length > 0) {
      const found = samples.find((s) => s.id === preselectedSampleId);
      if (found) {
        setSelectedFile(null);
        setSelectedSampleId(found.id);
        setPreviewUrl(found.imageUrl);
        setPreviewLabel(found.title);
        setLocalError(null);
        setPrepPreview(null);
      }
    }
  }, [preselectedSampleId, samples]);

  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const validateAndSetFile = (file: File) => {
    setLocalError(null);
    setPrepPreview(null);
    const lower = file.name.toLowerCase();
    const validExt =
      lower.endsWith('.jpg') ||
      lower.endsWith('.jpeg') ||
      lower.endsWith('.png') ||
      lower.endsWith('.dcm');

    if (!validExt) {
      setLocalError(
        'Unsupported format. Please select a JPG, JPEG, PNG, or uncompressed DICOM (.dcm) CT scan.'
      );
      return;
    }

    if (file.size > MAX_BYTES) {
      setLocalError('File exceeds the 15 MB upload limit.');
      return;
    }

    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setSelectedSampleId(null);
    setPreviewLabel(
      lower.endsWith('.dcm')
        ? `DICOM Slice (${(file.size / 1024).toFixed(0)} KB)`
        : `${file.name} (${(file.size / 1024).toFixed(0)} KB)`
    );

    if (lower.endsWith('.dcm')) {
      // Browser <img> cannot directly render raw .dcm bytes; run /api/preprocess immediately for preview!
      setPreviewUrl(null);
      void handlePreviewPreprocess(file, undefined, filterMode);
    } else {
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) validateAndSetFile(file);
    e.target.value = '';
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) validateAndSetFile(file);
  };

  const handleSelectSample = (sample: SampleScan) => {
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setLocalError(null);
    setPrepPreview(null);
    setSelectedFile(null);
    setSelectedSampleId(sample.id);
    setPreviewUrl(sample.imageUrl);
    setPreviewLabel(sample.title);
  };

  const handleClear = () => {
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setSelectedSampleId(null);
    setPreviewUrl(null);
    setPreviewLabel('');
    setLocalError(null);
    setPrepPreview(null);
  };

  const handlePreviewPreprocess = async (
    fileArg?: File,
    sampleArg?: string,
    modeArg: FilterMode = filterMode
  ) => {
    const targetFile = fileArg ?? selectedFile ?? undefined;
    const targetSample = sampleArg ?? selectedSampleId ?? undefined;
    if (!targetFile && !targetSample) return;

    setIsPreprocessing(true);
    setLocalError(null);
    try {
      const res = await preprocessScan({
        file: targetFile,
        sampleId: targetSample,
        filterMode: modeArg,
      });
      setPrepPreview(res);
      if (!previewUrl) {
        setPreviewUrl(res.outputs.originalPngDataUrl);
      }
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Preprocessing failed.');
    } finally {
      setIsPreprocessing(false);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile && !selectedSampleId) {
      setLocalError('Upload a thoracic CT scan or select one of the reference scans first.');
      return;
    }
    setLocalError(null);
    await onRunPrediction({
      file: selectedFile ?? undefined,
      sampleId: selectedSampleId ?? undefined,
      modelId: selectedModel,
      filterMode,
    });
  };

  const activeError = localError || predictionError;
  const hasSelection = Boolean(selectedFile || selectedSampleId);

  return (
    <div className="space-y-8 pb-12">
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="font-display text-2xl font-semibold text-slate-900">
            Thoracic CT Scan Upload & Preprocessing
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            Upload an axial lung CT slice (PNG, JPG, JPEG, or uncompressed DICOM .dcm)
            or load a reference scan to run OpenCV preprocessing and classification.
          </p>
        </div>
      </div>

      {activeError && (
        <div
          role="alert"
          className="bg-rose-50 border border-rose-200 text-rose-800 rounded-xl p-4 flex items-start gap-3 text-sm"
        >
          <AlertCircle className="h-5 w-5 text-rose-600 shrink-0 mt-0.5" aria-hidden="true" />
          <span>{activeError}</span>
        </div>
      )}

      <div className="grid lg:grid-cols-12 gap-8 items-start">
        {/* Left: Upload Dropzone & Preview */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.dcm,image/png,image/jpeg,application/dicom"
              onChange={handleFileChange}
              className="hidden"
            />

            {!hasSelection ? (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-xl p-10 text-center transition-colors ${
                  isDragging
                    ? 'border-sky-500 bg-sky-50'
                    : 'border-slate-300 bg-slate-50/70 hover:border-sky-400'
                }`}
              >
                <ImagePlus
                  className="h-10 w-10 text-slate-400 mx-auto mb-3"
                  aria-hidden="true"
                />
                <p className="font-display text-base font-semibold text-slate-900">
                  Drag & drop lung CT scan image here
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  Supports PNG, JPG, JPEG, and uncompressed DICOM (.dcm) up to 15 MB
                </p>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="mt-5 px-4 py-2 bg-[#0B1324] hover:bg-slate-800 text-white text-sm font-medium rounded-lg transition-colors focus-visible:outline-2 focus-visible:outline-sky-500"
                >
                  Browse Files
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-slate-900 min-w-0">
                    <CheckCircle2
                      className="h-4 w-4 text-sky-600 shrink-0"
                      aria-hidden="true"
                    />
                    <span className="truncate">{previewLabel}</span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors whitespace-nowrap"
                    >
                      <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
                      <span>Replace</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleClear}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-rose-700 bg-rose-50 hover:bg-rose-100 rounded-lg transition-colors whitespace-nowrap"
                    >
                      <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                      <span>Remove</span>
                    </button>
                  </div>
                </div>

                <div className="grid sm:grid-cols-2 gap-4">
                  <div className="bg-black rounded-lg overflow-hidden border border-slate-200 aspect-square flex items-center justify-center">
                    {previewUrl ? (
                      <img
                        src={previewUrl}
                        alt="Selected CT scan preview"
                        referrerPolicy="no-referrer"
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <div className="text-xs text-slate-400 p-4 text-center">
                        Decoding scan preview...
                      </div>
                    )}
                  </div>

                  <div className="bg-slate-50 rounded-lg border border-slate-200 p-4 flex flex-col justify-between gap-4">
                    <div className="space-y-2 text-xs text-slate-600">
                      <p className="font-semibold text-slate-900 text-sm">
                        Preprocessing Inspection
                      </p>
                      <p>
                        Preview the 224×224 resized,{' '}
                        {filterMode === 'gaussian' ? '5×5 Gaussian' : '3×3 Median'}{' '}
                        denoised, CLAHE-enhanced, and Otsu Lung ROI stages before
                        running full model classification.
                      </p>
                      {prepPreview && (
                        <div className="pt-2 space-y-1 font-mono text-[11px] text-slate-700">
                          <p>Format: {prepPreview.metadata.format}</p>
                          <p>
                            Original: {prepPreview.metadata.originalWidth}×
                            {prepPreview.metadata.originalHeight}
                          </p>
                          <p>PHI Headers: Stripped</p>
                        </div>
                      )}
                    </div>

                    <button
                      type="button"
                      disabled={isPreprocessing}
                      onClick={() => void handlePreviewPreprocess()}
                      className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 text-xs font-semibold text-slate-800 bg-white border border-slate-300 hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-50"
                    >
                      {isPreprocessing ? (
                        <>
                          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
                          <span>Running OpenCV Pipeline...</span>
                        </>
                      ) : (
                        <>
                          <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
                          <span>Preview Preprocessing Stages</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {prepPreview && (
                  <div className="pt-2 border-t border-slate-200">
                    <p className="text-xs font-medium text-slate-700 mb-3">
                      Side-by-Side OpenCV Preprocessing Stages
                    </p>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {[
                        { label: 'Denoised', src: prepPreview.outputs.denoisedPngDataUrl },
                        { label: 'CLAHE Enhanced', src: prepPreview.outputs.clahePngDataUrl },
                        { label: 'Canny Edges', src: prepPreview.outputs.edgesPngDataUrl },
                        { label: 'Lung ROI Mask', src: prepPreview.outputs.roiMaskPngDataUrl },
                      ].map((item) => (
                        <div key={item.label}>
                          <div className="aspect-square bg-black rounded-lg overflow-hidden border border-slate-200">
                            <img
                              src={item.src}
                              alt={item.label}
                              referrerPolicy="no-referrer"
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <p className="mt-1.5 text-xs text-slate-600 text-center">
                            {item.label}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Reference Sample Selector */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
            <div className="flex items-baseline justify-between gap-2">
              <h2 className="font-display text-base font-semibold text-slate-900">
                Reference Axial CT Samples
              </h2>
              <span className="text-xs text-slate-500">
                Click any scan to load real pixel input
              </span>
            </div>
            <div className="grid sm:grid-cols-3 gap-4">
              {samples.map((sample) => {
                const active = selectedSampleId === sample.id;
                return (
                  <button
                    key={sample.id}
                    type="button"
                    onClick={() => handleSelectSample(sample)}
                    className={`text-left rounded-xl border p-3 transition-colors focus-visible:outline-2 focus-visible:outline-sky-500 ${
                      active
                        ? 'border-sky-600 bg-sky-50/50'
                        : 'border-slate-200 hover:border-slate-300 bg-white'
                    }`}
                  >
                    <div className="aspect-square rounded-lg overflow-hidden bg-black mb-2.5">
                      <img
                        src={sample.imageUrl}
                        alt={sample.title}
                        referrerPolicy="no-referrer"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <p className="text-xs font-mono text-sky-700">{sample.categoryHint}</p>
                    <p className="mt-0.5 text-sm font-semibold text-slate-900 truncate">
                      {sample.title}
                    </p>
                    <p className="mt-1 text-xs text-slate-500 line-clamp-2">
                      {sample.description}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right: Inference Configuration & Submit */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 space-y-6">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-900">
              Inference Configuration
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Select the trained deep learning classifier and noise filtering kernel.
            </p>
          </div>

          <div className="space-y-2.5">
            <label className="block text-xs font-semibold text-slate-700">
              Classification Architecture
            </label>
            <div className="space-y-2">
              {MODEL_OPTIONS.map((opt) => {
                const active = selectedModel === opt.id;
                return (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => onSelectModel(opt.id)}
                    className={`w-full text-left p-3.5 rounded-lg border transition-colors ${
                      active
                        ? 'border-sky-600 bg-sky-50/60'
                        : 'border-slate-200 hover:border-slate-300 bg-white'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-semibold text-slate-900">
                        {opt.name}
                      </span>
                      {opt.id === 'resnet50' && (
                        <span className="text-xs font-mono text-sky-700">
                          Best Val/Test
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-xs text-slate-500">{opt.note}</p>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="space-y-2">
            <label className="block text-xs font-semibold text-slate-700">
              OpenCV Noise Reduction Kernel
            </label>
            <div className="grid grid-cols-2 gap-2 p-1 bg-slate-100 rounded-lg">
              <button
                type="button"
                onClick={() => onSelectFilterMode('gaussian')}
                className={`py-2 px-3 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                  filterMode === 'gaussian'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                5×5 Gaussian Blur
              </button>
              <button
                type="button"
                onClick={() => onSelectFilterMode('median')}
                className={`py-2 px-3 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                  filterMode === 'median'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                3×3 Median Filter
              </button>
            </div>
          </div>

          <div className="pt-2 border-t border-slate-200 space-y-3">
            <button
              type="button"
              disabled={!hasSelection || isPredicting}
              onClick={() => void handleAnalyze()}
              className="w-full inline-flex items-center justify-center gap-2 px-5 py-3 bg-[#0B1324] hover:bg-slate-800 disabled:bg-slate-300 text-white font-semibold text-sm rounded-lg transition-colors focus-visible:outline-2 focus-visible:outline-sky-500"
            >
              {isPredicting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                  <span>Analyzing CT Scan & Computing Grad-CAM...</span>
                </>
              ) : (
                <span>Analyze Scan</span>
              )}
            </button>
            <p className="text-xs text-slate-500 leading-relaxed">
              Uploaded images are processed in-memory with DICOM PHI tags stripped.
              Results include CLAHE enhancement, 24 radiomic/conv features, Grad-CAM
              attention, and candidate nodule segmentation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
