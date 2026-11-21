import { ArrowRight, Cpu, Eye, Layers, Microscope, ShieldAlert } from 'lucide-react';
import { ModelInfoResponse, PageId, SampleScan } from '../types/api';

interface HomeDashboardProps {
  samples: SampleScan[];
  modelInfo: ModelInfoResponse | null;
  onNavigate: (page: PageId) => void;
  onQuickAnalyzeSample: (sampleId: string) => void;
}

const PIPELINE_STEPS = [
  {
    step: '01. OpenCV CT Preprocessing',
    summary:
      'Resizes thoracic slices to 224x224, applies 5x5 Gaussian or 3x3 Median denoising, enhances parenchymal contrast with CLAHE, extracts bilateral lung ROI via Otsu thresholding, and maps edges with Canny.',
  },
  {
    step: '02. Radiomic & Deep Feature Extraction',
    summary:
      'Extracts 24-dimensional hybrid descriptors combining intensity histograms, 4-angle GLCM texture matrices, HOG gradients, nodule circularity/spiculation indices, and multi-scale Gabor/LoG pooling.',
  },
  {
    step: '03. CNN & Transfer Learning Classification',
    summary:
      'Evaluates Custom CNN alongside ImageNet-pretrained VGG16, ResNet50, and EfficientNetB0 transfer architectures using patient-level 70/15/15 splits to prevent slice data leakage.',
  },
  {
    step: '04. Grad-CAM & Nodule Segmentation',
    summary:
      'Generates spatial Grad-CAM attention overlays alongside morphological and U-Net candidate nodule segmentation masks, reporting Dice/IoU metrics while separating classification from localization.',
  },
];

const TECH_STACK = [
  { label: 'Frontend Interface', detail: 'React.js, TypeScript, Tailwind CSS, Chart.js' },
  { label: 'Web & API Backend', detail: 'Express REST API + Python Flask Reference Server' },
  { label: 'Computer Vision', detail: 'OpenCV (CLAHE, Canny, Gaussian, Otsu ROI), NumPy, PIL' },
  { label: 'Deep Learning', detail: 'TensorFlow / Keras, Custom CNN, VGG16, ResNet50, EfficientNetB0' },
  { label: 'Explainability & Seg', detail: 'Grad-CAM Gradient Attribution, U-Net Segmentation' },
  { label: 'Evaluation Metrics', detail: 'Scikit-learn, Sensitivity, Specificity, ROC-AUC, Confusion Matrix' },
];

const DATASETS = [
  {
    name: 'IQ-OTH/NCCD Lung Cancer Dataset',
    classes: 'Normal · Benign · Malignant',
    notes:
      'Primary 3-class thoracic CT classification corpus used for categorical training subject to academic dataset license terms.',
  },
  {
    name: 'LIDC-IDRI & LUNA16',
    classes: 'Nodule Detection · Segmentation Masks',
    notes:
      'DICOM/MHD volumetric thoracic CT scans with radiologist-annotated nodule coordinates and contour masks for U-Net segmentation.',
  },
];

export function HomeDashboard({
  samples,
  modelInfo,
  onNavigate,
  onQuickAnalyzeSample,
}: HomeDashboardProps) {
  const bestModel = modelInfo ? modelInfo.models[modelInfo.bestModelId] : null;
  const split = modelInfo?.splitStrategy;

  return (
    <div className="space-y-10 pb-12">
      {/* Hero Split Banner */}
      <section className="bg-[#0B1324] text-white rounded-xl border border-slate-800 overflow-hidden">
        <div className="grid lg:grid-cols-12 items-stretch">
          <div className="lg:col-span-7 p-8 lg:p-10 flex flex-col justify-between gap-8">
            <div className="space-y-4">
              <p className="text-xs font-mono tracking-wider uppercase text-sky-400">
                Computer Vision · OpenCV · Deep Learning
              </p>
              <h1 className="font-display text-3xl lg:text-4xl font-semibold tracking-tight leading-tight text-balance">
                Lung Cancer Detection System Using Thoracic CT Imaging
              </h1>
              <p className="text-slate-300 text-base leading-relaxed max-w-2xl">
                An academic and research platform integrating OpenCV image enhancement,
                GLCM/HOG radiomics, Custom CNN and Transfer Learning models (VGG16,
                ResNet50, EfficientNetB0), and Grad-CAM visual attribution for Normal,
                Benign, and Malignant pulmonary CT analysis.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-4">
              <button
                type="button"
                onClick={() => onNavigate('upload')}
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold text-sm rounded-lg transition-colors whitespace-nowrap focus-visible:outline-2 focus-visible:outline-sky-300"
              >
                <span>Start Analysis</span>
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </button>
              <button
                type="button"
                onClick={() => onNavigate('performance')}
                className="inline-flex items-center gap-2 px-5 py-2.5 border border-slate-700 hover:border-slate-500 text-slate-100 font-medium text-sm rounded-lg transition-colors whitespace-nowrap focus-visible:outline-2 focus-visible:outline-sky-300"
              >
                View Model Evaluation
              </button>
            </div>
          </div>

          <div className="lg:col-span-5 bg-slate-950/60 border-t lg:border-t-0 lg:border-l border-slate-800 p-6 flex flex-col justify-between gap-4">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Reference Axial CT Samples</span>
              <span className="font-mono">224×224 Lung Window</span>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {samples.length > 0 ? (
                samples.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => onQuickAnalyzeSample(s.id)}
                    className="group text-left focus-visible:outline-2 focus-visible:outline-sky-400 rounded-lg"
                  >
                    <div className="aspect-square rounded-lg overflow-hidden border border-slate-800 bg-black">
                      <img
                        src={s.imageUrl}
                        alt={s.title}
                        referrerPolicy="no-referrer"
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-150"
                      />
                    </div>
                    <p className="mt-2 text-xs font-medium text-slate-200 truncate">
                      {s.categoryHint.replace(' Reference', '')}
                    </p>
                  </button>
                ))
              ) : (
                <div className="col-span-3 py-10 text-center text-xs text-slate-400">
                  Loading thoracic CT reference slices...
                </div>
              )}
            </div>

            <p className="text-xs text-slate-400">
              Select any reference CT slice above to immediately run preprocessing,
              classification, and Grad-CAM attribution.
            </p>
          </div>
        </div>
      </section>

      {/* Benchmark KPI Strip */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-xs text-slate-500">Top Selected Architecture</p>
          <p className="mt-1.5 font-display text-lg font-semibold text-slate-900 truncate">
            {bestModel ? bestModel.displayName.split(' (')[0] : 'ResNet50'}
          </p>
          <p className="mt-1 text-xs font-mono tabular-nums text-sky-700">
            Test Acc: {bestModel ? `${(bestModel.accuracy * 100).toFixed(1)}%` : '--'}
          </p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-xs text-slate-500">Macro Sensitivity / Specificity</p>
          <p className="mt-1.5 font-mono tabular-nums text-2xl font-semibold text-slate-900">
            {bestModel
              ? `${(bestModel.macroSensitivity * 100).toFixed(1)}% / ${(
                  bestModel.macroSpecificity * 100
                ).toFixed(1)}%`
              : '--'}
          </p>
          <p className="mt-1 text-xs text-slate-500">Held-out 15% patient test</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-xs text-slate-500">Patient-Level Split Protocol</p>
          <p className="mt-1.5 font-mono tabular-nums text-2xl font-semibold text-slate-900">
            70 / 15 / 15
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {split
              ? `${split.totalPatients} patients · ${split.totalSlices} slices`
              : 'Leakage-free GroupSplit'}
          </p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-xs text-slate-500">Classification Categories</p>
          <p className="mt-1.5 font-display text-lg font-semibold text-slate-900">
            Normal · Benign · Malignant
          </p>
          <p className="mt-1 text-xs font-mono text-slate-500">
            Macro AUC: {bestModel ? bestModel.macroAuc.toFixed(4) : '--'}
          </p>
        </div>
      </section>

      {/* Introduction & How It Works */}
      <section className="grid lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-2 text-sky-700">
            <Microscope className="h-5 w-5" aria-hidden="true" />
            <h2 className="font-display text-lg font-semibold text-slate-900">
              Introduction to Lung Cancer CT Analysis
            </h2>
          </div>
          <p className="text-sm text-slate-600 leading-relaxed">
            Early detection of pulmonary nodules on low-dose thoracic Computed Tomography
            (CT) is critical for improving survival outcomes. Manual slice-by-slice
            inspection across hundreds of axial frames is time-intensive and prone to
            inter-observer variability for small or subtle lesions.
          </p>
          <p className="text-sm text-slate-600 leading-relaxed">
            This research platform pairs classical OpenCV preprocessing (CLAHE contrast
            equalization, morphological lung ROI masking, Canny edge detection) with
            convolutional neural networks and explainable Grad-CAM heatmaps to distinguish{' '}
            <strong className="font-medium text-slate-900">Normal parenchyma</strong>,{' '}
            <strong className="font-medium text-slate-900">Benign circumscribed nodules</strong>, and{' '}
            <strong className="font-medium text-slate-900">Malignant spiculated masses</strong>.
          </p>
        </div>

        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6">
          <div className="flex items-center gap-2 text-sky-700 mb-5">
            <Layers className="h-5 w-5" aria-hidden="true" />
            <h2 className="font-display text-lg font-semibold text-slate-900">
              How the AI Pipeline Works
            </h2>
          </div>
          <div className="divide-y divide-slate-200">
            {PIPELINE_STEPS.map((item) => (
              <div key={item.step} className="py-3.5 first:pt-0 last:pb-0">
                <h3 className="font-display text-sm font-semibold text-slate-900">
                  {item.step}
                </h3>
                <p className="mt-1 text-sm text-slate-600 leading-relaxed">
                  {item.summary}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technologies & Dataset Documentation */}
      <section className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <div className="flex items-center gap-2 text-sky-700 mb-4">
            <Cpu className="h-5 w-5" aria-hidden="true" />
            <h2 className="font-display text-lg font-semibold text-slate-900">
              Technologies & Architecture Stack
            </h2>
          </div>
          <div className="divide-y divide-slate-100">
            {TECH_STACK.map((t) => (
              <div
                key={t.label}
                className="py-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 text-sm"
              >
                <span className="font-medium text-slate-900">{t.label}</span>
                <span className="text-slate-600 text-xs sm:text-right">{t.detail}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
          <div className="flex items-center gap-2 text-sky-700">
            <Eye className="h-5 w-5" aria-hidden="true" />
            <h2 className="font-display text-lg font-semibold text-slate-900">
              Dataset Sources & Leakage Prevention
            </h2>
          </div>
          <div className="space-y-4">
            {DATASETS.map((ds) => (
              <div
                key={ds.name}
                className="pb-4 border-b border-slate-100 last:border-b-0 last:pb-0"
              >
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h3 className="text-sm font-semibold text-slate-900">{ds.name}</h3>
                  <span className="text-xs font-mono text-sky-700">{ds.classes}</span>
                </div>
                <p className="mt-1 text-sm text-slate-600 leading-relaxed">{ds.notes}</p>
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            All slices belonging to the same patient volume are strictly assigned to a
            single split (70% Train, 15% Validation, 15% Test) via patient-grouped
            splitting, preventing adjacent-slice correlation leakage.
          </p>
        </div>
      </section>

      {/* Academic Research Notice */}
      <section className="bg-slate-100 border border-slate-200 rounded-xl p-5 flex items-start gap-3.5">
        <ShieldAlert className="h-5 w-5 text-slate-700 shrink-0 mt-0.5" aria-hidden="true" />
        <div className="text-sm text-slate-700 leading-relaxed">
          <span className="font-semibold text-slate-900">
            Academic & Research Disclaimer:{' '}
          </span>
          This software is developed strictly for educational and computer vision research
          purposes. It has not undergone prospective clinical trials or regulatory
          clearance and must not be used for clinical diagnosis, screening, or medical
          decision-making.
        </div>
      </section>
    </div>
  );
}
