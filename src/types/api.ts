export type LungClass = 'Normal' | 'Benign' | 'Malignant';
export const CLASS_LABELS: LungClass[] = ['Normal', 'Benign', 'Malignant'];
export type ModelArchitectureId = 'efficientnet_b0' | 'resnet50' | 'vgg16' | 'custom_cnn';
export type FilterMode = 'gaussian' | 'median';
export type PageId = 'home' | 'upload' | 'results' | 'performance' | 'history';

export interface ImageMetadata {
  format: 'PNG' | 'JPEG' | 'DICOM' | 'UNKNOWN';
  originalWidth: number;
  originalHeight: number;
  processedWidth: number;
  processedHeight: number;
  dicomModality?: string;
  sliceThicknessMm?: string;
  windowCenter?: number;
  windowWidth?: number;
  phiStripped: boolean;
}

export interface RadiomicFeatures {
  meanIntensity: number;
  stdIntensity: number;
  skewness: number;
  kurtosis: number;
  entropy: number;
  glcmContrast: number;
  glcmDissimilarity: number;
  glcmHomogeneity: number;
  glcmEnergy: number;
  glcmCorrelation: number;
  edgeDensity: number;
  hogDominantEnergy: number;
  hogSpread: number;
  roiCoverageRatio: number;
  noduleCandidateCount: number;
  maxLesionAreaRatio: number;
  lesionCircularity: number;
  spiculationIndex: number;
  focalContrastRatio: number;
  bilateralAsymmetry: number;
  convGaborResponse: number;
  convLaplacianPeak: number;
  convBlobMaxPool: number;
  convTextureMeanPool: number;
}

export interface SegmentationSummary {
  lungRoiAreaPct: number;
  suspectRegionCount: number;
  primaryNoduleAreaPct: number;
  primaryNoduleCircularity: number;
  spiculationScore: number;
  centroidNormalized: { x: number; y: number } | null;
  pseudoDiceConsistency: number;
  pseudoIoU: number;
}

export interface PreprocessingOutputs {
  originalPngDataUrl: string;
  denoisedPngDataUrl: string;
  clahePngDataUrl: string;
  edgesPngDataUrl: string;
  roiMaskPngDataUrl: string;
}

export interface PredictionResult {
  id: string;
  timestamp: string;
  scanLabel?: string;
  modelUsed: ModelArchitectureId;
  modelDisplayName: string;
  predictedClass: LungClass;
  confidence: number;
  probabilities: Record<LungClass, number>;
  metadata: ImageMetadata;
  features: RadiomicFeatures;
  featureVector: number[];
  segmentation: SegmentationSummary;
  visualizations: PreprocessingOutputs & {
    gradcamOverlayPngDataUrl: string;
    gradcamHeatmapPngDataUrl: string;
    segmentationOverlayPngDataUrl: string;
  };
  disclaimer: string;
}

export interface PreprocessResponse {
  metadata: ImageMetadata;
  filterMode: FilterMode;
  outputs: PreprocessingOutputs;
}

export interface SampleScan {
  id: string;
  title: string;
  categoryHint: string;
  description: string;
  imageUrl: string;
  filename: string;
}

export interface EpochHistory {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAccuracy: number;
  valAccuracy: number;
}

export interface ClassMetrics {
  className: LungClass;
  precision: number;
  recall: number;
  f1Score: number;
  sensitivity: number;
  specificity: number;
  auc: number;
  support: number;
  rocPoints: { fpr: number; tpr: number }[];
}

export interface ModelEvaluationReport {
  architecture: ModelArchitectureId;
  displayName: string;
  family: 'Custom CNN' | 'Transfer Learning';
  inputNormalization: string;
  optimizer: string;
  epochsTrained: number;
  bestEpoch: number;
  accuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  macroSensitivity: number;
  macroSpecificity: number;
  macroAuc: number;
  confusionMatrix: number[][];
  perClassMetrics: ClassMetrics[];
  history: EpochHistory[];
}

export interface ModelInfoResponse {
  version: string;
  trainedAt: string;
  datasetProtocol: string;
  splitStrategy: {
    trainRatio: number;
    valRatio: number;
    testRatio: number;
    patientLevelLeakageFree: boolean;
    totalPatients: number;
    totalSlices: number;
    trainPatients: number;
    valPatients: number;
    testPatients: number;
    trainSlices: number;
    valSlices: number;
    testSlices: number;
  };
  bestModelId: ModelArchitectureId;
  models: Record<ModelArchitectureId, ModelEvaluationReport>;
  featureNames: string[];
}

export interface HistoryEntry {
  id: string;
  timestamp: string;
  scanLabel: string;
  modelUsed: ModelArchitectureId;
  modelDisplayName: string;
  predictedClass: LungClass;
  confidence: number;
  probabilities: Record<LungClass, number>;
  thumbnailDataUrl: string;
  gradcamOverlayDataUrl: string;
  preprocessedDataUrl: string;
  segmentationOverlayDataUrl: string;
  metadata: ImageMetadata;
  features: RadiomicFeatures;
  segmentation: SegmentationSummary;
}
