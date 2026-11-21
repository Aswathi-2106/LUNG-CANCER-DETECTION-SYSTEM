import {
  AlertTriangle,
  CheckCircle2,
  FileSearch,
  Info,
  ShieldCheck,
  Upload,
} from 'lucide-react';
import { ProbabilityBarChart } from '../components/Charts';
import { LungClass, PageId, PredictionResult } from '../types/api';

interface ResultsPageProps {
  result: PredictionResult | null;
  onNavigate: (page: PageId) => void;
}

const EDUCATIONAL_NOTES: Record<
  LungClass,
  {
    headline: string;
    morphology: string;
    radiomicsCue: string;
    caveat: string;
  }
> = {
  Normal: {
    headline: 'Normal Pulmonary Parenchyma Pattern',
    morphology:
      'Bilateral lung fields demonstrate expected low-attenuation aerated parenchyma with branching pulmonary vessels tapering symmetrically toward the pleura and no discrete focal hyper-dense mass.',
    radiomicsCue:
      'Low focal contrast excess, balanced bilateral mean intensity, and minimal peripheral spiculation or solitary lesion area within the segmented lung ROI.',
    caveat:
      'A Normal model prediction reflects statistical similarity to reference non-nodular CT slices and does not rule out sub-resolution micro-nodules, ground-glass opacities, or pathology outside this 2D slice.',
  },
  Benign: {
    headline: 'Benign Solitary Nodule Pattern',
    morphology:
      'Benign pulmonary nodules (such as hamartomas or healed granulomas) typically present with well-circumscribed, smooth rounded margins, uniform internal density, and minimal radial strand extension.',
    radiomicsCue:
      'Higher lesion circularity (4πA/P² approaching 1.0), lower perimeter spiculation index, and localized focal contrast against surrounding aerated lung tissue.',
    caveat:
      'Benign-appearing radiographic features in a single 2D slice still require longitudinal volumetric CT follow-up and clinical correlation by a board-certified radiologist.',
  },
  Malignant: {
    headline: 'Malignant / Suspicious Mass Pattern',
    morphology:
      'Malignant pulmonary lesions frequently exhibit irregular, lobulated, or spiculated margins ("corona radiata"), heterogeneous internal attenuation, pleural tethering, and asymmetric parenchymal distortion.',
    radiomicsCue:
      'Elevated spiculation index, reduced circularity, higher GLCM contrast/heterogeneity, and prominent bilateral asymmetry with multi-scale edge activation.',
    caveat:
      'High-attention regions indicate patterns correlated with malignancy in the training cohort; biopsy, PET-CT, and multidisciplinary clinical evaluation are required for any diagnosis.',
  },
};

export function ResultsPage({ result, onNavigate }: ResultsPageProps) {
  if (!result) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-12 text-center max-w-2xl mx-auto my-8 space-y-4">
        <FileSearch className="h-10 w-10 text-slate-400 mx-auto" aria-hidden="true" />
        <h1 className="font-display text-xl font-semibold text-slate-900">
          No CT Scan Analyzed Yet
        </h1>
        <p className="text-sm text-slate-600 leading-relaxed">
          Upload a thoracic CT scan or select one of the reference axial slices to view
          predicted classification probabilities, Grad-CAM attention overlays, OpenCV
          preprocessing stages, and extracted radiomic features.
        </p>
        <button
          type="button"
          onClick={() => onNavigate('upload')}
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-[#0B1324] hover:bg-slate-800 text-white text-sm font-semibold rounded-lg transition-colors"
        >
          <Upload className="h-4 w-4" aria-hidden="true" />
          <span>Go to CT Scan Upload</span>
        </button>
      </div>
    );
  }

  const edu = EDUCATIONAL_NOTES[result.predictedClass];

  const statusTone =
    result.predictedClass === 'Normal'
      ? {
          text: 'text-sky-800',
          bg: 'bg-sky-50 border-sky-200',
          icon: ShieldCheck,
        }
      : result.predictedClass === 'Benign'
      ? {
          text: 'text-teal-800',
          bg: 'bg-teal-50 border-teal-200',
          icon: CheckCircle2,
        }
      : {
          text: 'text-rose-800',
          bg: 'bg-rose-50 border-rose-200',
          icon: AlertTriangle,
        };

  const StatusIcon = statusTone.icon;

  return (
    <div className="space-y-8 pb-12">
      {/* Header Summary */}
      <section className="bg-white border border-slate-200 rounded-xl p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <span>{result.scanLabel || 'Thoracic CT Slice'}</span>
              <span aria-hidden="true">·</span>
              <span className="font-mono">{result.modelDisplayName}</span>
              <span aria-hidden="true">·</span>
              <span className="font-mono tabular-nums">
                {new Date(result.timestamp).toLocaleString()}
              </span>
            </div>

            <div className="flex items-center gap-3">
              <div
                className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg border text-sm font-semibold ${statusTone.bg} ${statusTone.text}`}
              >
                <StatusIcon className="h-4 w-4 shrink-0" aria-hidden="true" />
                <span>Predicted Category: {result.predictedClass}</span>
              </div>
              <span className="font-mono tabular-nums text-2xl font-bold text-slate-900">
                {(result.confidence * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-slate-500">Confidence</span>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-6 text-xs text-slate-600 border-t lg:border-t-0 pt-4 lg:pt-0 border-slate-200">
            <div>
              <p className="text-slate-400">Source Format</p>
              <p className="font-mono font-medium text-slate-900 mt-0.5">
                {result.metadata.format} ({result.metadata.originalWidth}×
                {result.metadata.originalHeight})
              </p>
            </div>
            <div>
              <p className="text-slate-400">Network Input</p>
              <p className="font-mono font-medium text-slate-900 mt-0.5">
                {result.metadata.processedWidth}×{result.metadata.processedHeight} px
              </p>
            </div>
            <div>
              <p className="text-slate-400">Privacy</p>
              <p className="font-mono font-medium text-slate-900 mt-0.5">
                {result.metadata.phiStripped ? 'PHI Stripped' : 'Standard'}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Probability Distribution & Grad-CAM Side-by-Side */}
      <section className="grid lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 space-y-5">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-900">
              Class Probability Distribution
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Softmax output across Normal, Benign, and Malignant categories.
            </p>
          </div>

          <ProbabilityBarChart probabilities={result.probabilities} />

          <div className="divide-y divide-slate-100 border-t border-slate-100 pt-2 text-sm">
            {(['Normal', 'Benign', 'Malignant'] as LungClass[]).map((cls) => (
              <div key={cls} className="py-2 flex items-center justify-between">
                <span className="font-medium text-slate-700">{cls}</span>
                <span className="font-mono tabular-nums font-semibold text-slate-900">
                  {(result.probabilities[cls] * 100).toFixed(2)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 space-y-5">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-900">
              Explainable AI: Original CT vs. Grad-CAM Attribution
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Highlights spatial lung regions weighted heavily by convolutional and
              radiomic feature maps for the predicted class.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <div className="aspect-square rounded-lg overflow-hidden bg-black border border-slate-200">
                <img
                  src={result.visualizations.originalPngDataUrl}
                  alt="Original resized thoracic CT scan"
                  referrerPolicy="no-referrer"
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="mt-2 text-xs font-medium text-slate-700 text-center">
                Original CT (224×224)
              </p>
            </div>
            <div>
              <div className="aspect-square rounded-lg overflow-hidden bg-black border border-slate-200">
                <img
                  src={result.visualizations.gradcamOverlayPngDataUrl}
                  alt="Grad-CAM spatial attention overlay on CT scan"
                  referrerPolicy="no-referrer"
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="mt-2 text-xs font-medium text-slate-700 text-center">
                Grad-CAM Overlay
              </p>
            </div>
            <div>
              <div className="aspect-square rounded-lg overflow-hidden bg-black border border-slate-200">
                <img
                  src={result.visualizations.gradcamHeatmapPngDataUrl}
                  alt="Raw Grad-CAM spatial heatmap"
                  referrerPolicy="no-referrer"
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="mt-2 text-xs font-medium text-slate-700 text-center">
                Class Activation Map
              </p>
            </div>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 flex items-start gap-2.5 text-xs text-slate-600">
            <Info className="h-4 w-4 text-sky-700 shrink-0 mt-0.5" aria-hidden="true" />
            <span>{result.disclaimer}</span>
          </div>
        </div>
      </section>

      {/* OpenCV Preprocessing & Candidate Nodule Segmentation */}
      <section className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-900">
              OpenCV Preprocessing & Candidate Nodule Segmentation
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Side-by-side comparison of noise filtering, CLAHE contrast equalization,
              Canny edges, bilateral Lung ROI mask, and morphological lesion segmentation.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500">
            Lung ROI Coverage: {result.segmentation.lungRoiAreaPct.toFixed(1)}%
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          {[
            { title: '1. Original', src: result.visualizations.originalPngDataUrl },
            { title: '2. Denoised', src: result.visualizations.denoisedPngDataUrl },
            { title: '3. CLAHE', src: result.visualizations.clahePngDataUrl },
            { title: '4. Canny Edges', src: result.visualizations.edgesPngDataUrl },
            { title: '5. Lung ROI', src: result.visualizations.roiMaskPngDataUrl },
            {
              title: '6. Nodule Seg',
              src: result.visualizations.segmentationOverlayPngDataUrl,
            },
          ].map((panel) => (
            <div key={panel.title}>
              <div className="aspect-square rounded-lg overflow-hidden bg-black border border-slate-200">
                <img
                  src={panel.src}
                  alt={panel.title}
                  referrerPolicy="no-referrer"
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="mt-2 text-xs font-medium text-slate-700 text-center">
                {panel.title}
              </p>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 pt-3 border-t border-slate-200 text-xs">
          <div>
            <p className="text-slate-500">Suspect Regions</p>
            <p className="mt-0.5 font-mono tabular-nums text-base font-semibold text-slate-900">
              {result.segmentation.suspectRegionCount}
            </p>
          </div>
          <div>
            <p className="text-slate-500">Primary Region Area</p>
            <p className="mt-0.5 font-mono tabular-nums text-base font-semibold text-slate-900">
              {result.segmentation.primaryNoduleAreaPct.toFixed(2)}% of ROI
            </p>
          </div>
          <div>
            <p className="text-slate-500">Circularity (4πA/P²)</p>
            <p className="mt-0.5 font-mono tabular-nums text-base font-semibold text-slate-900">
              {result.segmentation.primaryNoduleCircularity.toFixed(3)}
            </p>
          </div>
          <div>
            <p className="text-slate-500">Spiculation Index</p>
            <p className="mt-0.5 font-mono tabular-nums text-base font-semibold text-slate-900">
              {result.segmentation.spiculationScore.toFixed(3)}
            </p>
          </div>
          <div>
            <p className="text-slate-500">Mask Dice / IoU Consistency</p>
            <p className="mt-0.5 font-mono tabular-nums text-base font-semibold text-slate-900">
              {result.segmentation.pseudoDiceConsistency.toFixed(2)} /{' '}
              {result.segmentation.pseudoIoU.toFixed(2)}
            </p>
          </div>
        </div>
        <p className="text-xs text-slate-500">
          Note: Image classification alone does not perform clinical tumor localization.
          Candidate segmentation regions are derived from morphological lung-ROI
          thresholding; full pixel-level U-Net segmentation requires voxel-annotated
          datasets such as LIDC-IDRI/LUNA16 (provided in{' '}
          <span className="font-mono">training/segmentation.py</span>).
        </p>
      </section>

      {/* Educational Explanation & Extracted Radiomics */}
      <section className="grid lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <h2 className="font-display text-base font-semibold text-slate-900">
            Educational Category Interpretation
          </h2>
          <p className="text-sm font-semibold text-sky-800">{edu.headline}</p>
          <div className="space-y-3 text-sm text-slate-600 leading-relaxed">
            <p>
              <strong className="font-medium text-slate-900">
                Typical Radiographic Pattern:{' '}
              </strong>
              {edu.morphology}
            </p>
            <p>
              <strong className="font-medium text-slate-900">
                Observed Feature Cues:{' '}
              </strong>
              {edu.radiomicsCue}
            </p>
            <p className="text-xs text-slate-500 pt-2 border-t border-slate-100">
              {edu.caveat}
            </p>
          </div>
        </div>

        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-900">
              Extracted Radiomic & Convolutional Features
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Computed directly inside the segmented bilateral Lung ROI.
            </p>
          </div>
          <div className="grid sm:grid-cols-2 gap-x-8 divide-y divide-slate-100 text-xs">
            {[
              ['ROI Mean Intensity (CLAHE)', result.features.meanIntensity.toFixed(2)],
              ['ROI Std Intensity', result.features.stdIntensity.toFixed(2)],
              ['Shannon Entropy (bits)', result.features.entropy.toFixed(4)],
              ['Intensity Skewness', result.features.skewness.toFixed(4)],
              ['GLCM Contrast', result.features.glcmContrast.toFixed(4)],
              ['GLCM Homogeneity', result.features.glcmHomogeneity.toFixed(4)],
              ['GLCM Energy', result.features.glcmEnergy.toFixed(4)],
              ['GLCM Correlation', result.features.glcmCorrelation.toFixed(4)],
              ['Canny Edge Density', result.features.edgeDensity.toFixed(4)],
              ['HOG Dominant Energy', result.features.hogDominantEnergy.toFixed(4)],
              ['Focal Contrast Ratio', result.features.focalContrastRatio.toFixed(4)],
              ['Bilateral Asymmetry', result.features.bilateralAsymmetry.toFixed(4)],
              ['Conv Gabor Response', result.features.convGaborResponse.toFixed(4)],
              ['Conv Laplacian Peak', result.features.convLaplacianPeak.toFixed(4)],
            ].map(([label, val]) => (
              <div
                key={label}
                className="py-2 flex items-center justify-between gap-2"
              >
                <span className="text-slate-600">{label}</span>
                <span className="font-mono tabular-nums font-semibold text-slate-900">
                  {val}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
