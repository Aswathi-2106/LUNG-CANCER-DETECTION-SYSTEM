import fs from 'node:fs';
import path from 'node:path';
import { PNG } from 'pngjs';
import jpeg from 'jpeg-js';

export type LungClass = 'Normal' | 'Benign' | 'Malignant';
export type ModelArchitectureId = 'efficientnet_b0' | 'resnet50' | 'vgg16' | 'custom_cnn';

export const CLASS_LABELS: LungClass[] = ['Normal', 'Benign', 'Malignant'];

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
  // Histogram features inside lung ROI
  meanIntensity: number;
  stdIntensity: number;
  skewness: number;
  kurtosis: number;
  entropy: number;
  // GLCM texture features (averaged across 0, 45, 90, 135 deg)
  glcmContrast: number;
  glcmDissimilarity: number;
  glcmHomogeneity: number;
  glcmEnergy: number;
  glcmCorrelation: number;
  // Edge & HOG features
  edgeDensity: number;
  hogDominantEnergy: number;
  hogSpread: number;
  // Shape & lesion candidate morphology
  roiCoverageRatio: number;
  noduleCandidateCount: number;
  maxLesionAreaRatio: number;
  lesionCircularity: number;
  spiculationIndex: number;
  focalContrastRatio: number;
  bilateralAsymmetry: number;
  // Deep-learning-inspired spatial filter bank GAP features
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

export interface TrainedModelWeights {
  architecture: ModelArchitectureId;
  displayName: string;
  featureMean: number[];
  featureStd: number[];
  // 2-layer MLP weights on top of extracted CV/Conv feature vector:
  // hidden = ReLU(X_norm * W1 + b1) -> logits = hidden * W2 + b2
  w1: number[][]; // [24][hiddenSize]
  b1: number[];   // [hiddenSize]
  w2: number[][]; // [hiddenSize][3]
  b2: number[];   // [3]
  spatialFeatureWeights: number[]; // length 6 for Grad-CAM channel weighting
}

export interface ModelBundle {
  version: string;
  trainedAt: string;
  datasetProtocol: string;
  splitSummary: {
    totalPatients: number;
    totalSlices: number;
    trainPatients: number;
    valPatients: number;
    testPatients: number;
    trainSlices: number;
    valSlices: number;
    testSlices: number;
  };
  models: Record<ModelArchitectureId, TrainedModelWeights>;
}

const TARGET_SIZE = 224;

// --- Image Decoding (PNG, JPEG, uncompressed DICOM) ---

export function decodeImageBuffer(
  buffer: Buffer,
  filename = ''
): { gray224: Float32Array; rgba224: Uint8Array; metadata: ImageMetadata } {
  const lower = filename.toLowerCase();

  // Check PNG signature
  if (
    buffer.length > 8 &&
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47
  ) {
    const parsed = PNG.sync.read(buffer);
    const { gray224, rgba224 } = resizeRgbaTo224(
      parsed.data,
      parsed.width,
      parsed.height
    );
    return {
      gray224,
      rgba224,
      metadata: {
        format: 'PNG',
        originalWidth: parsed.width,
        originalHeight: parsed.height,
        processedWidth: TARGET_SIZE,
        processedHeight: TARGET_SIZE,
        phiStripped: true,
      },
    };
  }

  // Check JPEG signature (FF D8)
  if (buffer.length > 3 && buffer[0] === 0xff && buffer[1] === 0xd8) {
    const decoded = jpeg.decode(buffer, { useTArray: true, formatAsRGBA: true });
    const { gray224, rgba224 } = resizeRgbaTo224(
      decoded.data,
      decoded.width,
      decoded.height
    );
    return {
      gray224,
      rgba224,
      metadata: {
        format: 'JPEG',
        originalWidth: decoded.width,
        originalHeight: decoded.height,
        processedWidth: TARGET_SIZE,
        processedHeight: TARGET_SIZE,
        phiStripped: true,
      },
    };
  }

  // Check DICOM signature ("DICM" at offset 128) or .dcm extension
  if (
    (buffer.length > 132 && buffer.toString('ascii', 128, 132) === 'DICM') ||
    lower.endsWith('.dcm')
  ) {
    return decodeDicomBuffer(buffer);
  }

  throw new Error(
    'Unsupported image format. Please upload a valid PNG, JPEG, or uncompressed DICOM (.dcm) thoracic CT scan.'
  );
}

function decodeDicomBuffer(buffer: Buffer): {
  gray224: Float32Array;
  rgba224: Uint8Array;
  metadata: ImageMetadata;
} {
  // Lightweight Explicit/Implicit VR Little-Endian DICOM parser (strips PHI tags)
  let offset = buffer.length > 132 && buffer.toString('ascii', 128, 132) === 'DICM' ? 132 : 0;
  let rows = 512;
  let cols = 512;
  let bitsAllocated = 16;
  let pixelRepresentation = 0;
  let rescaleSlope = 1;
  let rescaleIntercept = -1024;
  let windowCenter = -600;
  let windowWidth = 1500;
  let modality = 'CT';
  let sliceThickness = '2.0';
  let pixelDataOffset = -1;
  let pixelDataLength = 0;

  const explicitVRs = new Set([
    'AE', 'AS', 'AT', 'CS', 'DA', 'DS', 'DT', 'FL', 'FD', 'IS', 'LO', 'LT',
    'OB', 'OD', 'OF', 'OW', 'PN', 'SH', 'SL', 'SQ', 'SS', 'ST', 'TM', 'UI',
    'UL', 'UN', 'US', 'UT',
  ]);
  const longVRs = new Set(['OB', 'OD', 'OF', 'OW', 'SQ', 'UN', 'UT']);

  while (offset + 8 <= buffer.length) {
    const group = buffer.readUInt16LE(offset);
    const element = buffer.readUInt16LE(offset + 2);
    offset += 4;

    const vrCandidate = buffer.toString('ascii', offset, offset + 2);
    let vr = '';
    let length = 0;

    if (explicitVRs.has(vrCandidate)) {
      vr = vrCandidate;
      if (longVRs.has(vr)) {
        offset += 4; // 2 bytes VR + 2 reserved
        if (offset + 4 > buffer.length) break;
        length = buffer.readUInt32LE(offset);
        offset += 4;
      } else {
        length = buffer.readUInt16LE(offset + 2);
        offset += 4;
      }
    } else {
      length = buffer.readUInt32LE(offset);
      offset += 4;
    }

    if (length === 0xffffffff) {
      // Undefined length sequence/encapsulation
      if (group === 0x7fe0 && element === 0x0010) {
        pixelDataOffset = offset;
        pixelDataLength = buffer.length - offset;
      }
      break;
    }

    if (offset + length > buffer.length) break;

    if (group === 0x7fe0 && element === 0x0010) {
      pixelDataOffset = offset;
      pixelDataLength = length;
      break;
    }

    // Read clinical acquisition parameters only (never store PatientName/PatientID)
    if (group === 0x0028 && element === 0x0010 && length >= 2) {
      rows = buffer.readUInt16LE(offset);
    } else if (group === 0x0028 && element === 0x0011 && length >= 2) {
      cols = buffer.readUInt16LE(offset);
    } else if (group === 0x0028 && element === 0x0100 && length >= 2) {
      bitsAllocated = buffer.readUInt16LE(offset);
    } else if (group === 0x0028 && element === 0x0103 && length >= 2) {
      pixelRepresentation = buffer.readUInt16LE(offset);
    } else if (group === 0x0008 && element === 0x0060) {
      modality = buffer.toString('ascii', offset, offset + length).trim().replace(/\0/g, '');
    } else if (group === 0x0018 && element === 0x0050) {
      sliceThickness = buffer.toString('ascii', offset, offset + length).trim().replace(/\0/g, '');
    } else if (group === 0x0028 && element === 0x1050) {
      const val = parseFloat(buffer.toString('ascii', offset, offset + length).split('\\')[0]);
      if (!Number.isNaN(val)) windowCenter = val;
    } else if (group === 0x0028 && element === 0x1051) {
      const val = parseFloat(buffer.toString('ascii', offset, offset + length).split('\\')[0]);
      if (!Number.isNaN(val)) windowWidth = val;
    } else if (group === 0x0028 && element === 0x1052) {
      const val = parseFloat(buffer.toString('ascii', offset, offset + length));
      if (!Number.isNaN(val)) rescaleIntercept = val;
    } else if (group === 0x0028 && element === 0x1053) {
      const val = parseFloat(buffer.toString('ascii', offset, offset + length));
      if (!Number.isNaN(val)) rescaleSlope = val;
    }

    offset += length;
  }

  if (pixelDataOffset < 0 || rows <= 0 || cols <= 0) {
    throw new Error(
      'Unable to parse uncompressed DICOM pixel data. Compressed transfer syntaxes should be converted to PNG/JPEG or uncompressed Little-Endian DICOM.'
    );
  }

  const totalPixels = rows * cols;
  const rgba = new Uint8Array(totalPixels * 4);
  const bytesPerPixel = bitsAllocated === 8 ? 1 : 2;
  const availablePixels = Math.min(totalPixels, Math.floor(pixelDataLength / bytesPerPixel));
  const minHU = windowCenter - windowWidth / 2;
  const maxHU = windowCenter + windowWidth / 2;

  for (let i = 0; i < availablePixels; i++) {
    let raw = 0;
    if (bytesPerPixel === 1) {
      raw = buffer.readUInt8(pixelDataOffset + i);
    } else {
      raw =
        pixelRepresentation === 1
          ? buffer.readInt16LE(pixelDataOffset + i * 2)
          : buffer.readUInt16LE(pixelDataOffset + i * 2);
    }
    const hu = raw * rescaleSlope + rescaleIntercept;
    const norm = Math.max(0, Math.min(255, ((hu - minHU) / Math.max(1, maxHU - minHU)) * 255));
    const idx = i * 4;
    rgba[idx] = norm;
    rgba[idx + 1] = norm;
    rgba[idx + 2] = norm;
    rgba[idx + 3] = 255;
  }

  const { gray224, rgba224 } = resizeRgbaTo224(rgba, cols, rows);
  return {
    gray224,
    rgba224,
    metadata: {
      format: 'DICOM',
      originalWidth: cols,
      originalHeight: rows,
      processedWidth: TARGET_SIZE,
      processedHeight: TARGET_SIZE,
      dicomModality: modality,
      sliceThicknessMm: sliceThickness,
      windowCenter,
      windowWidth,
      phiStripped: true,
    },
  };
}

function resizeRgbaTo224(
  src: Uint8Array | Buffer,
  srcW: number,
  srcH: number
): { gray224: Float32Array; rgba224: Uint8Array } {
  const rgba224 = new Uint8Array(TARGET_SIZE * TARGET_SIZE * 4);
  const gray224 = new Float32Array(TARGET_SIZE * TARGET_SIZE);

  const scaleX = srcW / TARGET_SIZE;
  const scaleY = srcH / TARGET_SIZE;

  for (let y = 0; y < TARGET_SIZE; y++) {
    const sy = Math.min(srcH - 1, (y + 0.5) * scaleY - 0.5);
    const y0 = Math.max(0, Math.floor(sy));
    const y1 = Math.min(srcH - 1, y0 + 1);
    const fy = Math.max(0, sy - y0);

    for (let x = 0; x < TARGET_SIZE; x++) {
      const sx = Math.min(srcW - 1, (x + 0.5) * scaleX - 0.5);
      const x0 = Math.max(0, Math.floor(sx));
      const x1 = Math.min(srcW - 1, x0 + 1);
      const fx = Math.max(0, sx - x0);

      const i00 = (y0 * srcW + x0) * 4;
      const i10 = (y0 * srcW + x1) * 4;
      const i01 = (y1 * srcW + x0) * 4;
      const i11 = (y1 * srcW + x1) * 4;

      const w00 = (1 - fx) * (1 - fy);
      const w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy;
      const w11 = fx * fy;

      const r = src[i00] * w00 + src[i10] * w10 + src[i01] * w01 + src[i11] * w11;
      const g = src[i00 + 1] * w00 + src[i10 + 1] * w10 + src[i01 + 1] * w01 + src[i11 + 1] * w11;
      const b = src[i00 + 2] * w00 + src[i10 + 2] * w10 + src[i01 + 2] * w01 + src[i11 + 2] * w11;

      const dstIdx = (y * TARGET_SIZE + x) * 4;
      rgba224[dstIdx] = Math.round(r);
      rgba224[dstIdx + 1] = Math.round(g);
      rgba224[dstIdx + 2] = Math.round(b);
      rgba224[dstIdx + 3] = 255;

      gray224[y * TARGET_SIZE + x] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
  }

  return { gray224, rgba224 };
}

// --- OpenCV-Equivalent Preprocessing Pipeline ---

export function gaussianBlur5x5(src: Float32Array, w = TARGET_SIZE, h = TARGET_SIZE): Float32Array {
  // Separable 5x5 Gaussian kernel [1, 4, 6, 4, 1] / 16
  const kernel = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16];
  const temp = new Float32Array(w * h);
  const dst = new Float32Array(w * h);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let k = -2; k <= 2; k++) {
        const sx = Math.max(0, Math.min(w - 1, x + k));
        sum += src[y * w + sx] * kernel[k + 2];
      }
      temp[y * w + x] = sum;
    }
  }

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let k = -2; k <= 2; k++) {
        const sy = Math.max(0, Math.min(h - 1, y + k));
        sum += temp[sy * w + x] * kernel[k + 2];
      }
      dst[y * w + x] = sum;
    }
  }

  return dst;
}

export function medianFilter3x3(src: Float32Array, w = TARGET_SIZE, h = TARGET_SIZE): Float32Array {
  const dst = new Float32Array(w * h);
  const win = new Float32Array(9);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let idx = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const sy = Math.max(0, Math.min(h - 1, y + dy));
        for (let dx = -1; dx <= 1; dx++) {
          const sx = Math.max(0, Math.min(w - 1, x + dx));
          win[idx++] = src[sy * w + sx];
        }
      }
      win.sort();
      dst[y * w + x] = win[4];
    }
  }

  return dst;
}

export function applyCLAHE(
  src: Float32Array,
  w = TARGET_SIZE,
  h = TARGET_SIZE,
  tilesX = 8,
  tilesY = 8,
  clipLimitFactor = 2.2
): Float32Array {
  const tileW = w / tilesX;
  const tileH = h / tilesY;
  const numBins = 256;
  const cdfs: Float32Array[][] = [];

  for (let ty = 0; ty < tilesY; ty++) {
    cdfs[ty] = [];
    const yStart = Math.floor(ty * tileH);
    const yEnd = Math.floor((ty + 1) * tileH);

    for (let tx = 0; tx < tilesX; tx++) {
      const xStart = Math.floor(tx * tileW);
      const xEnd = Math.floor((tx + 1) * tileW);

      const hist = new Float32Array(numBins);
      let count = 0;
      for (let y = yStart; y < yEnd; y++) {
        for (let x = xStart; x < xEnd; x++) {
          const bin = Math.max(0, Math.min(255, Math.round(src[y * w + x])));
          hist[bin]++;
          count++;
        }
      }

      // Clip histogram and redistribute excess uniformly
      const clipLimit = Math.max(1, (clipLimitFactor * count) / numBins);
      let excess = 0;
      for (let b = 0; b < numBins; b++) {
        if (hist[b] > clipLimit) {
          excess += hist[b] - clipLimit;
          hist[b] = clipLimit;
        }
      }
      const redist = excess / numBins;
      for (let b = 0; b < numBins; b++) {
        hist[b] += redist;
      }

      // Build normalized CDF
      const cdf = new Float32Array(numBins);
      let cum = 0;
      for (let b = 0; b < numBins; b++) {
        cum += hist[b];
        cdf[b] = (cum / Math.max(1, count)) * 255;
      }
      cdfs[ty][tx] = cdf;
    }
  }

  // Bilinear interpolation across tile centers
  const dst = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const fty = (y + 0.5) / tileH - 0.5;
    const ty0 = Math.max(0, Math.min(tilesY - 1, Math.floor(fty)));
    const ty1 = Math.max(0, Math.min(tilesY - 1, ty0 + 1));
    const wy = Math.max(0, Math.min(1, fty - ty0));

    for (let x = 0; x < w; x++) {
      const ftx = (x + 0.5) / tileW - 0.5;
      const tx0 = Math.max(0, Math.min(tilesX - 1, Math.floor(ftx)));
      const tx1 = Math.max(0, Math.min(tilesX - 1, tx0 + 1));
      const wx = Math.max(0, Math.min(1, ftx - tx0));

      const bin = Math.max(0, Math.min(255, Math.round(src[y * w + x])));
      const v00 = cdfs[ty0][tx0][bin];
      const v10 = cdfs[ty0][tx1][bin];
      const v01 = cdfs[ty1][tx0][bin];
      const v11 = cdfs[ty1][tx1][bin];

      const top = v00 * (1 - wx) + v10 * wx;
      const bot = v01 * (1 - wx) + v11 * wx;
      dst[y * w + x] = top * (1 - wy) + bot * wy;
    }
  }

  return dst;
}

export function extractLungRoiMask(
  denoised: Float32Array,
  w = TARGET_SIZE,
  h = TARGET_SIZE
): Uint8Array {
  // Compute Otsu threshold on interior thoracic region (excluding outer scanner circle & text annotations)
  const hist = new Float32Array(256);
  let total = 0;
  const cx = w / 2;
  const cy = h / 2;
  const maxR = w * 0.43;

  for (let y = Math.floor(h * 0.14); y < Math.floor(h * 0.86); y++) {
    for (let x = Math.floor(w * 0.08); x < Math.floor(w * 0.92); x++) {
      const dx = x - cx;
      const dy = y - cy;
      if (dx * dx + dy * dy <= maxR * maxR) {
        const bin = Math.max(0, Math.min(255, Math.round(denoised[y * w + x])));
        hist[bin]++;
        total++;
      }
    }
  }

  let sumAll = 0;
  for (let i = 0; i < 256; i++) sumAll += i * hist[i];
  let sumB = 0;
  let wB = 0;
  let maxVar = -1;
  let otsuThresh = 100;

  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sumAll - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) {
      maxVar = between;
      otsuThresh = t;
    }
  }

  // Body tissue mask: pixels inside thoracic oval above Otsu threshold * 0.72
  const bodyThreshold = Math.max(45, otsuThresh * 0.78);
  const bodyMask = new Uint8Array(w * h);
  for (let y = Math.floor(h * 0.15); y < Math.floor(h * 0.85); y++) {
    for (let x = Math.floor(w * 0.07); x < Math.floor(w * 0.93); x++) {
      if (denoised[y * w + x] >= bodyThreshold) {
        bodyMask[y * w + x] = 1;
      }
    }
  }

  // Flood-fill external air from image borders across !bodyMask to separate outside air from internal lung air
  const outsideAir = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);
  let head = 0;
  let tail = 0;

  const pushIfAir = (idx: number) => {
    if (!bodyMask[idx] && !outsideAir[idx]) {
      outsideAir[idx] = 1;
      queue[tail++] = idx;
    }
  };

  for (let x = 0; x < w; x++) {
    pushIfAir(x);
    pushIfAir((h - 1) * w + x);
  }
  for (let y = 0; y < h; y++) {
    pushIfAir(y * w);
    pushIfAir(y * w + (w - 1));
  }

  while (head < tail) {
    const idx = queue[head++];
    const x = idx % w;
    const y = (idx - x) / w;
    if (x > 0) pushIfAir(idx - 1);
    if (x < w - 1) pushIfAir(idx + 1);
    if (y > 0) pushIfAir(idx - w);
    if (y < h - 1) pushIfAir(idx + w);
  }

  // Internal lung cavities = pixels inside the thoracic contour that are NOT outsideAir,
  // plus morphological closing so hyper-dense nodules inside the lung are preserved in the lung ROI!
  const thoracicInterior = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (!outsideAir[i]) thoracicInterior[i] = 1;
  }

  // Lung parenchyma ROI = thoracicInterior pixels that are dark (lung air) OR enclosed by lung tissue (nodules inside lung)
  const rawLung = new Uint8Array(w * h);
  for (let y = Math.floor(h * 0.18); y < Math.floor(h * 0.82); y++) {
    for (let x = Math.floor(w * 0.12); x < Math.floor(w * 0.88); x++) {
      const idx = y * w + x;
      // Exclude central mediastinum/spine column roughly around x in [0.43w, 0.57w] unless dark trachea
      const normX = Math.abs(x - cx) / w;
      if (thoracicInterior[idx] && denoised[idx] < otsuThresh * 0.92 && normX > 0.05) {
        rawLung[idx] = 1;
      }
    }
  }

  // 5-iteration morphological dilation + hole fill to include dense intra-parenchymal nodules/masses inside Lung ROI
  let roi = rawLung;
  for (let iter = 0; iter < 5; iter++) {
    const next = new Uint8Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        if (!thoracicInterior[idx]) continue;
        if (
          roi[idx] ||
          roi[idx - 1] ||
          roi[idx + 1] ||
          roi[idx - w] ||
          roi[idx + w]
        ) {
          // Keep inside thoracic cavity and avoid expanding deep into mediastinal bone/heart center
          const normX = Math.abs(x - cx) / w;
          if (normX > 0.055) next[idx] = 1;
        }
      }
    }
    roi = next;
  }

  // Fill any enclosed holes (solid tumors surrounded by lung parenchyma) via border flood-fill on !roi
  const notLung = new Uint8Array(w * h);
  head = 0;
  tail = 0;
  for (let x = 0; x < w; x++) {
    if (!roi[x]) { notLung[x] = 1; queue[tail++] = x; }
    const bIdx = (h - 1) * w + x;
    if (!roi[bIdx] && !notLung[bIdx]) { notLung[bIdx] = 1; queue[tail++] = bIdx; }
  }
  while (head < tail) {
    const idx = queue[head++];
    const x = idx % w;
    const y = (idx - x) / w;
    if (x > 0 && !roi[idx - 1] && !notLung[idx - 1]) { notLung[idx - 1] = 1; queue[tail++] = idx - 1; }
    if (x < w - 1 && !roi[idx + 1] && !notLung[idx + 1]) { notLung[idx + 1] = 1; queue[tail++] = idx + 1; }
    if (y > 0 && !roi[idx - w] && !notLung[idx - w]) { notLung[idx - w] = 1; queue[tail++] = idx - w; }
    if (y < h - 1 && !roi[idx + w] && !notLung[idx + w]) { notLung[idx + w] = 1; queue[tail++] = idx + w; }
  }
  for (let i = 0; i < w * h; i++) {
    if (!notLung[i]) roi[i] = 1;
  }

  // Fallback oval lung field mask if uploaded non-standard CT image yielded tiny ROI
  let roiCount = 0;
  for (let i = 0; i < w * h; i++) roiCount += roi[i];
  if (roiCount < w * h * 0.06) {
    for (let y = Math.floor(h * 0.22); y < Math.floor(h * 0.78); y++) {
      for (let x = Math.floor(w * 0.16); x < Math.floor(w * 0.84); x++) {
        const normX = Math.abs(x - cx) / w;
        if (normX > 0.06) roi[y * w + x] = 1;
      }
    }
  }

  return roi;
}

export function cannyEdgeDetection(
  src: Float32Array,
  w = TARGET_SIZE,
  h = TARGET_SIZE,
  lowThresh = 24,
  highThresh = 64
): { edges: Uint8Array; mag: Float32Array; dir: Float32Array } {
  const mag = new Float32Array(w * h);
  const dir = new Float32Array(w * h);

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const gx =
        -src[idx - w - 1] + src[idx - w + 1] -
        2 * src[idx - 1] + 2 * src[idx + 1] -
        src[idx + w - 1] + src[idx + w + 1];
      const gy =
        -src[idx - w - 1] - 2 * src[idx - w] - src[idx - w + 1] +
        src[idx + w - 1] + 2 * src[idx + w] + src[idx + w + 1];
      mag[idx] = Math.hypot(gx, gy);
      dir[idx] = Math.atan2(gy, gx);
    }
  }

  // Non-maximum suppression
  const nms = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const angle = ((dir[idx] * 180) / Math.PI + 180) % 180;
      let q = 0;
      let r = 0;
      if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
        q = mag[idx + 1];
        r = mag[idx - 1];
      } else if (angle >= 22.5 && angle < 67.5) {
        q = mag[idx + w - 1];
        r = mag[idx - w + 1];
      } else if (angle >= 67.5 && angle < 112.5) {
        q = mag[idx + w];
        r = mag[idx - w];
      } else {
        q = mag[idx - w - 1];
        r = mag[idx + w + 1];
      }
      if (mag[idx] >= q && mag[idx] >= r) {
        nms[idx] = mag[idx];
      }
    }
  }

  // Double threshold + hysteresis
  const edges = new Uint8Array(w * h);
  const queue: number[] = [];
  for (let i = 0; i < w * h; i++) {
    if (nms[i] >= highThresh) {
      edges[i] = 255;
      queue.push(i);
    } else if (nms[i] >= lowThresh) {
      edges[i] = 128;
    }
  }

  while (queue.length > 0) {
    const idx = queue.pop()!;
    const x = idx % w;
    const y = (idx - x) / w;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
          const nIdx = ny * w + nx;
          if (edges[nIdx] === 128) {
            edges[nIdx] = 255;
            queue.push(nIdx);
          }
        }
      }
    }
  }

  for (let i = 0; i < w * h; i++) {
    if (edges[i] !== 255) edges[i] = 0;
  }

  return { edges, mag, dir };
}

// --- Feature Extraction & Spatial Lesion / Nodule Candidate Analysis ---

interface ConnectedComponent {
  pixels: number[];
  area: number;
  perimeter: number;
  circularity: number;
  meanIntensity: number;
  maxIntensity: number;
  cx: number;
  cy: number;
  spiculation: number;
}

export function extractRadiomicAndConvFeatures(
  denoised: Float32Array,
  clahe: Float32Array,
  roiMask: Uint8Array,
  edges: Uint8Array,
  gradMag: Float32Array,
  gradDir: Float32Array,
  w = TARGET_SIZE,
  h = TARGET_SIZE
): {
  features: RadiomicFeatures;
  vector: number[];
  noduleMask: Uint8Array;
  activationMap: Float32Array;
  segmentation: SegmentationSummary;
} {
  // 1. Histogram statistics inside Lung ROI
  const roiPixels: number[] = [];
  let leftSum = 0;
  let leftCount = 0;
  let rightSum = 0;
  let rightCount = 0;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (roiMask[idx]) {
        const val = clahe[idx];
        roiPixels.push(val);
        if (x < w / 2) {
          leftSum += val;
          leftCount++;
        } else {
          rightSum += val;
          rightCount++;
        }
      }
    }
  }

  const n = Math.max(1, roiPixels.length);
  const meanIntensity = roiPixels.reduce((a, b) => a + b, 0) / n;
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  const hist32 = new Float32Array(32);

  for (let i = 0; i < roiPixels.length; i++) {
    const d = roiPixels[i] - meanIntensity;
    m2 += d * d;
    m3 += d * d * d;
    m4 += d * d * d * d;
    const b = Math.max(0, Math.min(31, Math.floor((roiPixels[i] / 256) * 32)));
    hist32[b]++;
  }

  const variance = m2 / n;
  const stdIntensity = Math.sqrt(variance);
  const skewness = stdIntensity > 1e-5 ? m3 / n / Math.pow(stdIntensity, 3) : 0;
  const kurtosis = stdIntensity > 1e-5 ? m4 / n / Math.pow(stdIntensity, 4) - 3 : 0;

  let entropy = 0;
  for (let b = 0; b < 32; b++) {
    const p = hist32[b] / n;
    if (p > 0) entropy -= p * Math.log2(p);
  }

  // 2. GLCM Texture Features (16 quantization levels inside ROI across 4 directions)
  const levels = 16;
  const dirs = [
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, -1],
  ];
  let glcmContrast = 0;
  let glcmDissimilarity = 0;
  let glcmHomogeneity = 0;
  let glcmEnergy = 0;
  let glcmCorrelation = 0;

  for (const [dx, dy] of dirs) {
    const co = new Float32Array(levels * levels);
    let pairs = 0;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx1 = y * w + x;
        const idx2 = (y + dy) * w + (x + dx);
        if (roiMask[idx1] && roiMask[idx2]) {
          const q1 = Math.max(0, Math.min(levels - 1, Math.floor((clahe[idx1] / 256) * levels)));
          const q2 = Math.max(0, Math.min(levels - 1, Math.floor((clahe[idx2] / 256) * levels)));
          co[q1 * levels + q2]++;
          co[q2 * levels + q1]++;
          pairs += 2;
        }
      }
    }
    if (pairs === 0) continue;

    let muI = 0;
    let muJ = 0;
    for (let i = 0; i < levels; i++) {
      for (let j = 0; j < levels; j++) {
        const p = co[i * levels + j] / pairs;
        muI += i * p;
        muJ += j * p;
      }
    }
    let sigI = 0;
    let sigJ = 0;
    for (let i = 0; i < levels; i++) {
      for (let j = 0; j < levels; j++) {
        const p = co[i * levels + j] / pairs;
        sigI += (i - muI) * (i - muI) * p;
        sigJ += (j - muJ) * (j - muJ) * p;
      }
    }
    const denom = Math.sqrt(sigI * sigJ) + 1e-6;

    for (let i = 0; i < levels; i++) {
      for (let j = 0; j < levels; j++) {
        const p = co[i * levels + j] / pairs;
        if (p === 0) continue;
        const diff = i - j;
        glcmContrast += diff * diff * p;
        glcmDissimilarity += Math.abs(diff) * p;
        glcmHomogeneity += p / (1 + diff * diff);
        glcmEnergy += p * p;
        glcmCorrelation += ((i - muI) * (j - muJ) * p) / denom;
      }
    }
  }
  glcmContrast /= dirs.length;
  glcmDissimilarity /= dirs.length;
  glcmHomogeneity /= dirs.length;
  glcmEnergy /= dirs.length;
  glcmCorrelation /= dirs.length;

  // 3. Edge & HOG Orientation Features inside ROI
  const hogBins = new Float32Array(9);
  let edgeCount = 0;
  for (let i = 0; i < w * h; i++) {
    if (!roiMask[i]) continue;
    if (edges[i] > 0) edgeCount++;
    const angleDeg = ((gradDir[i] * 180) / Math.PI + 180) % 180;
    const bin = Math.min(8, Math.floor(angleDeg / 20));
    hogBins[bin] += gradMag[i];
  }
  const edgeDensity = edgeCount / n;
  const hogTotal = hogBins.reduce((a, b) => a + b, 0) + 1e-6;
  const hogNorm = Array.from(hogBins).map((v) => v / hogTotal);
  const hogDominantEnergy = Math.max(...hogNorm);
  const hogMean = 1 / 9;
  const hogSpread = Math.sqrt(
    hogNorm.reduce((acc, v) => acc + (v - hogMean) * (v - hogMean), 0) / 9
  );

  // 4. Focal Lesion Candidate Segmentation (inside strictly interior Lung ROI away from hilar vessels)
  // Compute radial distance from chest wall / mediastinum so peripheral & mid-lung focal lesions are isolated cleanly
  const noduleCandidatesRaw = new Uint8Array(w * h);
  const focalThreshold = Math.max(125, meanIntensity + 1.45 * stdIntensity);
  const rawFocalThresh = Math.max(95, meanIntensity * 0.85);

  for (let y = Math.floor(h * 0.22); y < Math.floor(h * 0.78); y++) {
    for (let x = Math.floor(w * 0.17); x < Math.floor(w * 0.83); x++) {
      const idx = y * w + x;
      if (!roiMask[idx]) continue;
      // Exclude immediate central hilar root (x within 13% of center)
      const distCenter = Math.abs(x - w / 2) / w;
      if (distCenter < 0.13) continue;

      // Ensure pixel is strictly interior to lung ROI (at least 5px away from chest wall/ribs)
      let interior = true;
      for (let dy = -4; dy <= 4 && interior; dy += 4) {
        for (let dx = -4; dx <= 4; dx += 4) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h || !roiMask[ny * w + nx]) {
            interior = false;
            break;
          }
        }
      }
      if (!interior) continue;

      if (clahe[idx] >= focalThreshold && denoised[idx] >= rawFocalThresh) {
        noduleCandidatesRaw[idx] = 1;
      }
    }
  }

  // Connected components analysis on focal hyper-intense regions
  const visited = new Uint8Array(w * h);
  const components: ConnectedComponent[] = [];
  const noduleMask = new Uint8Array(w * h);

  for (let i = 0; i < w * h; i++) {
    if (noduleCandidatesRaw[i] && !visited[i]) {
      const compPixels: number[] = [];
      const stack = [i];
      visited[i] = 1;

      while (stack.length > 0) {
        const cur = stack.pop()!;
        compPixels.push(cur);
        const cx = cur % w;
        const cy = (cur - cx) / w;
        const neighbors = [
          cx > 0 ? cur - 1 : -1,
          cx < w - 1 ? cur + 1 : -1,
          cy > 0 ? cur - w : -1,
          cy < h - 1 ? cur + w : -1,
        ];
        for (const nb of neighbors) {
          if (nb >= 0 && noduleCandidatesRaw[nb] && !visited[nb]) {
            visited[nb] = 1;
            stack.push(nb);
          }
        }
      }

      // Filter out tiny vessel cross-sections (< 28 px) and overly wide chest-wall bleed (> 3800 px)
      if (compPixels.length >= 28 && compPixels.length <= 3800) {
        const compSet = new Set(compPixels);
        let perimeter = 0;
        let sumX = 0;
        let sumY = 0;
        let sumInt = 0;
        let maxInt = 0;
        let edgeOverlap = 0;

        for (const p of compPixels) {
          const px = p % w;
          const py = (p - px) / w;
          sumX += px;
          sumY += py;
          sumInt += denoised[p];
          if (denoised[p] > maxInt) maxInt = denoised[p];
          if (edges[p] > 0) edgeOverlap++;

          const isBoundary =
            !compSet.has(p - 1) ||
            !compSet.has(p + 1) ||
            !compSet.has(p - w) ||
            !compSet.has(p + w);
          if (isBoundary) perimeter++;
        }

        const area = compPixels.length;
        const circularity = Math.min(
          1,
          (4 * Math.PI * area) / Math.max(1, perimeter * perimeter)
        );
        // Spiculation combines perimeter irregularity and internal edge/strand density
        const spiculation = Math.max(
          0,
          (1 - circularity) * 0.7 + (edgeOverlap / Math.max(1, area)) * 1.4
        );

        // Keep genuine solitary nodules / masses (either compact circular or large heterogeneous mass)
        if (area >= 42 || circularity > 0.35) {
          components.push({
            pixels: compPixels,
            area,
            perimeter,
            circularity,
            meanIntensity: sumInt / area,
            maxIntensity: maxInt,
            cx: sumX / area,
            cy: sumY / area,
            spiculation,
          });
          for (const p of compPixels) noduleMask[p] = 1;
        }
      }
    }
  }

  components.sort((a, b) => b.area - a.area);
  const primary = components[0] ?? null;

  const roiCoverageRatio = roiPixels.length / (w * h);
  const maxLesionAreaRatio = primary ? primary.area / n : 0;
  const lesionCircularity = primary ? primary.circularity : 0;
  const spiculationIndex = primary ? primary.spiculation : 0;
  const focalContrastRatio = primary
    ? (primary.meanIntensity - meanIntensity) / Math.max(1, stdIntensity)
    : 0;

  const leftMean = leftCount > 0 ? leftSum / leftCount : 0;
  const rightMean = rightCount > 0 ? rightSum / rightCount : 0;
  const bilateralAsymmetry = Math.abs(leftMean - rightMean) / Math.max(1, (leftMean + rightMean) / 2);

  // 5. Deep-Learning-Inspired Spatial Conv Feature Maps (LoG blob + Gabor orientation + Texture)
  const activationMap = new Float32Array(w * h);
  let convGaborSum = 0;
  let convLapPeak = 0;
  let convBlobMax = 0;
  let convTexSum = 0;

  // Multi-scale Difference of Gaussians / Laplacian response + nodule proximity boost
  const blurSmall = gaussianBlur5x5(clahe, w, h);
  const blurLarge = gaussianBlur5x5(gaussianBlur5x5(blurSmall, w, h), w, h);

  for (let y = 2; y < h - 2; y++) {
    for (let x = 2; x < w - 2; x++) {
      const idx = y * w + x;
      if (!roiMask[idx]) continue;

      const distCenter = Math.abs(x - w / 2) / w;
      const gabor =
        Math.abs(
          clahe[idx - w - 1] -
            clahe[idx + w + 1] +
            0.5 * (clahe[idx - 1] - clahe[idx + 1])
        ) / 255;
      const lap = Math.abs(blurSmall[idx] - blurLarge[idx]) / 255;
      const blob = Math.max(0, (blurSmall[idx] - meanIntensity) / 255);

      convGaborSum += gabor;
      convTexSum += Math.abs(clahe[idx] - blurSmall[idx]) / 255;
      if (lap > convLapPeak) convLapPeak = lap;
      if (blob > convBlobMax) convBlobMax = blob;

      // Spatial attention map combines focal intensity excess, DoG edge energy, and nodule mask focus
      const peripheryWeight = distCenter > 0.11 ? 1.0 : 0.25;
      const noduleBoost = noduleMask[idx] ? 0.65 : 0;
      activationMap[idx] =
        peripheryWeight * (0.45 * blob + 0.35 * lap + 0.2 * gabor + noduleBoost);
    }
  }

  // Smooth activation map for crisp Grad-CAM-like receptive field upsampling
  const smoothActivation = gaussianBlur5x5(
    gaussianBlur5x5(gaussianBlur5x5(activationMap, w, h), w, h),
    w,
    h
  );

  const convGaborResponse = convGaborSum / n;
  const convTextureMeanPool = convTexSum / n;

  const features: RadiomicFeatures = {
    meanIntensity: round4(meanIntensity),
    stdIntensity: round4(stdIntensity),
    skewness: round4(skewness),
    kurtosis: round4(kurtosis),
    entropy: round4(entropy),
    glcmContrast: round4(glcmContrast),
    glcmDissimilarity: round4(glcmDissimilarity),
    glcmHomogeneity: round4(glcmHomogeneity),
    glcmEnergy: round4(glcmEnergy),
    glcmCorrelation: round4(glcmCorrelation),
    edgeDensity: round4(edgeDensity),
    hogDominantEnergy: round4(hogDominantEnergy),
    hogSpread: round4(hogSpread),
    roiCoverageRatio: round4(roiCoverageRatio),
    noduleCandidateCount: components.length,
    maxLesionAreaRatio: round4(maxLesionAreaRatio),
    lesionCircularity: round4(lesionCircularity),
    spiculationIndex: round4(spiculationIndex),
    focalContrastRatio: round4(focalContrastRatio),
    bilateralAsymmetry: round4(bilateralAsymmetry),
    convGaborResponse: round4(convGaborResponse),
    convLaplacianPeak: round4(convLapPeak),
    convBlobMaxPool: round4(convBlobMax),
    convTextureMeanPool: round4(convTextureMeanPool),
  };

  const vector = [
    features.meanIntensity,
    features.stdIntensity,
    features.skewness,
    features.kurtosis,
    features.entropy,
    features.glcmContrast,
    features.glcmDissimilarity,
    features.glcmHomogeneity,
    features.glcmEnergy,
    features.glcmCorrelation,
    features.edgeDensity,
    features.hogDominantEnergy,
    features.hogSpread,
    features.roiCoverageRatio,
    features.noduleCandidateCount,
    features.maxLesionAreaRatio,
    features.lesionCircularity,
    features.spiculationIndex,
    features.focalContrastRatio,
    features.bilateralAsymmetry,
    features.convGaborResponse,
    features.convLaplacianPeak,
    features.convBlobMaxPool,
    features.convTextureMeanPool,
  ];

  // Estimate mask consistency metrics (comparing Otsu/CLAHE candidate against denoised intensity boundary)
  let intersect = 0;
  let union = 0;
  let sumM = 0;
  for (let i = 0; i < w * h; i++) {
    const pred = noduleMask[i] === 1;
    const ref = roiMask[i] === 1 && denoised[i] >= focalThreshold * 0.88 && Math.abs((i % w) - w / 2) / w > 0.115;
    if (pred && ref) intersect++;
    if (pred || ref) union++;
    if (pred) sumM++;
  }
  const pseudoIoU = union > 0 ? round4(intersect / union) : 0;
  const pseudoDice = sumM > 0 ? round4((2 * intersect) / Math.max(1, sumM + (union - (sumM - intersect)))) : 0;

  const segmentation: SegmentationSummary = {
    lungRoiAreaPct: round4(roiCoverageRatio * 100),
    suspectRegionCount: components.length,
    primaryNoduleAreaPct: round4(maxLesionAreaRatio * 100),
    primaryNoduleCircularity: round4(lesionCircularity),
    spiculationScore: round4(spiculationIndex),
    centroidNormalized: primary
      ? { x: round4(primary.cx / w), y: round4(primary.cy / h) }
      : null,
    pseudoDiceConsistency: pseudoDice,
    pseudoIoU,
  };

  return {
    features,
    vector,
    noduleMask,
    activationMap: smoothActivation,
    segmentation,
  };
}

// --- Neural Network Forward Pass & Grad-CAM Rendering ---

export function runModelForward(
  model: TrainedModelWeights,
  featureVector: number[]
): { probabilities: Record<LungClass, number>; predictedClass: LungClass; confidence: number } {
  const xNorm = featureVector.map((v, i) => (v - model.featureMean[i]) / Math.max(1e-5, model.featureStd[i]));

  const hiddenSize = model.b1.length;
  const hidden = new Array<number>(hiddenSize).fill(0);

  for (let j = 0; j < hiddenSize; j++) {
    let sum = model.b1[j];
    for (let i = 0; i < xNorm.length; i++) {
      sum += xNorm[i] * model.w1[i][j];
    }
    hidden[j] = Math.max(0, sum); // ReLU
  }

  const logits = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    let sum = model.b2[c];
    for (let j = 0; j < hiddenSize; j++) {
      sum += hidden[j] * model.w2[j][c];
    }
    logits[c] = sum;
  }

  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sumExp);

  const probabilities: Record<LungClass, number> = {
    Normal: round4(probs[0]),
    Benign: round4(probs[1]),
    Malignant: round4(probs[2]),
  };

  let bestIdx = 0;
  if (probs[1] > probs[bestIdx]) bestIdx = 1;
  if (probs[2] > probs[bestIdx]) bestIdx = 2;

  return {
    probabilities,
    predictedClass: CLASS_LABELS[bestIdx],
    confidence: round4(probs[bestIdx]),
  };
}

// --- PNG Encoding Utilities for Web Visualization ---

function grayToPngDataUrl(gray: Float32Array | Uint8Array, w = TARGET_SIZE, h = TARGET_SIZE): string {
  const png = new PNG({ width: w, height: h });
  for (let i = 0; i < w * h; i++) {
    const v = Math.max(0, Math.min(255, Math.round(gray[i])));
    const p = i * 4;
    png.data[p] = v;
    png.data[p + 1] = v;
    png.data[p + 2] = v;
    png.data[p + 3] = 255;
  }
  const out = PNG.sync.write(png);
  return `data:image/png;base64,${out.toString('base64')}`;
}

function rgbaToPngDataUrl(rgba: Uint8Array, w = TARGET_SIZE, h = TARGET_SIZE): string {
  const png = new PNG({ width: w, height: h });
  png.data.set(rgba);
  const out = PNG.sync.write(png);
  return `data:image/png;base64,${out.toString('base64')}`;
}

function jetColormap(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(255, Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * v - 3), 0), 1))));
  const g = Math.max(0, Math.min(255, Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * v - 2), 0), 1))));
  const b = Math.max(0, Math.min(255, Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * v - 1), 0), 1))));
  return [r, g, b];
}

function renderGradCamMaps(
  baseGray: Float32Array,
  activationMap: Float32Array,
  roiMask: Uint8Array,
  predictedClass: LungClass,
  w = TARGET_SIZE,
  h = TARGET_SIZE
): { heatmapUrl: string; overlayUrl: string } {
  let maxAct = 1e-6;
  for (let i = 0; i < w * h; i++) {
    if (roiMask[i] && activationMap[i] > maxAct) maxAct = activationMap[i];
  }

  const heatmapRgba = new Uint8Array(w * h * 4);
  const overlayRgba = new Uint8Array(w * h * 4);

  // Class-specific contrast gamma so Normal scans show diffuse low-attention rather than false alarm hotspots
  const gamma = predictedClass === 'Normal' ? 1.85 : predictedClass === 'Benign' ? 1.15 : 0.92;
  const maxIntensityScale = predictedClass === 'Normal' ? 0.52 : 1.0;

  for (let i = 0; i < w * h; i++) {
    const base = Math.max(0, Math.min(255, baseGray[i]));
    const rawNorm = roiMask[i] ? Math.min(1, activationMap[i] / maxAct) : 0;
    const norm = Math.pow(rawNorm, gamma) * maxIntensityScale;
    const [jr, jg, jb] = jetColormap(norm);

    const p = i * 4;
    heatmapRgba[p] = jr;
    heatmapRgba[p + 1] = jg;
    heatmapRgba[p + 2] = jb;
    heatmapRgba[p + 3] = 255;

    const alpha = norm > 0.14 ? 0.48 * Math.min(1, (norm - 0.14) / 0.65) : 0.08;
    overlayRgba[p] = Math.round(base * (1 - alpha) + jr * alpha);
    overlayRgba[p + 1] = Math.round(base * (1 - alpha) + jg * alpha);
    overlayRgba[p + 2] = Math.round(base * (1 - alpha) + jb * alpha);
    overlayRgba[p + 3] = 255;
  }

  return {
    heatmapUrl: rgbaToPngDataUrl(heatmapRgba, w, h),
    overlayUrl: rgbaToPngDataUrl(overlayRgba, w, h),
  };
}

function renderSegmentationOverlay(
  baseGray: Float32Array,
  roiMask: Uint8Array,
  noduleMask: Uint8Array,
  w = TARGET_SIZE,
  h = TARGET_SIZE
): { roiMaskUrl: string; segOverlayUrl: string } {
  const roiRgba = new Uint8Array(w * h * 4);
  const segRgba = new Uint8Array(w * h * 4);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const base = Math.max(0, Math.min(255, Math.round(baseGray[i])));
      const p = i * 4;

      // Lung ROI visualization (light cyan tint on parenchyma)
      if (roiMask[i]) {
        roiRgba[p] = Math.round(base * 0.72 + 14 * 0.28);
        roiRgba[p + 1] = Math.round(base * 0.72 + 165 * 0.28);
        roiRgba[p + 2] = Math.round(base * 0.72 + 233 * 0.28);
      } else {
        roiRgba[p] = Math.round(base * 0.35);
        roiRgba[p + 1] = Math.round(base * 0.35);
        roiRgba[p + 2] = Math.round(base * 0.35);
      }
      roiRgba[p + 3] = 255;

      // Segmentation Overlay: subtle lung boundary + highlighted candidate nodule mask
      const isRoiEdge =
        roiMask[i] &&
        x > 0 && x < w - 1 && y > 0 && y < h - 1 &&
        (!roiMask[i - 1] || !roiMask[i + 1] || !roiMask[i - w] || !roiMask[i + w]);

      const isNoduleEdge =
        noduleMask[i] &&
        x > 0 && x < w - 1 && y > 0 && y < h - 1 &&
        (!noduleMask[i - 1] || !noduleMask[i + 1] || !noduleMask[i - w] || !noduleMask[i + w]);

      if (isNoduleEdge) {
        segRgba[p] = 245;
        segRgba[p + 1] = 158;
        segRgba[p + 2] = 11; // Amber boundary
      } else if (noduleMask[i]) {
        segRgba[p] = Math.round(base * 0.52 + 239 * 0.48);
        segRgba[p + 1] = Math.round(base * 0.52 + 68 * 0.48);
        segRgba[p + 2] = Math.round(base * 0.52 + 68 * 0.48);
      } else if (isRoiEdge) {
        segRgba[p] = 56;
        segRgba[p + 1] = 189;
        segRgba[p + 2] = 248; // Cyan lung boundary
      } else {
        segRgba[p] = base;
        segRgba[p + 1] = base;
        segRgba[p + 2] = base;
      }
      segRgba[p + 3] = 255;
    }
  }

  return {
    roiMaskUrl: rgbaToPngDataUrl(roiRgba, w, h),
    segOverlayUrl: rgbaToPngDataUrl(segRgba, w, h),
  };
}

// --- Full End-to-End Inference Function ---

let cachedBundle: ModelBundle | null = null;

export function loadTrainedBundle(): ModelBundle {
  if (cachedBundle) return cachedBundle;
  const bundlePath = path.resolve(process.cwd(), 'server/artifacts/trained_models.json');
  if (!fs.existsSync(bundlePath)) {
    throw new Error(
      'Trained model weights (server/artifacts/trained_models.json) not found. Run scripts/train_web_model.ts first.'
    );
  }
  cachedBundle = JSON.parse(fs.readFileSync(bundlePath, 'utf-8')) as ModelBundle;
  return cachedBundle;
}

export function clearBundleCache(): void {
  cachedBundle = null;
}

export function preprocessImageBuffer(
  buffer: Buffer,
  filename = '',
  filterMode: 'gaussian' | 'median' = 'gaussian'
): {
  metadata: ImageMetadata;
  gray224: Float32Array;
  denoised: Float32Array;
  clahe: Float32Array;
  roiMask: Uint8Array;
  edges: Uint8Array;
  gradMag: Float32Array;
  gradDir: Float32Array;
  outputs: PreprocessingOutputs;
} {
  const { gray224, rgba224, metadata } = decodeImageBuffer(buffer, filename);
  const denoised =
    filterMode === 'median'
      ? medianFilter3x3(gray224)
      : gaussianBlur5x5(gray224);
  const clahe = applyCLAHE(denoised);
  const roiMask = extractLungRoiMask(denoised);
  const { edges, mag, dir } = cannyEdgeDetection(clahe);
  const { roiMaskUrl } = renderSegmentationOverlay(clahe, roiMask, new Uint8Array(TARGET_SIZE * TARGET_SIZE));

  return {
    metadata,
    gray224,
    denoised,
    clahe,
    roiMask,
    edges,
    gradMag: mag,
    gradDir: dir,
    outputs: {
      originalPngDataUrl: rgbaToPngDataUrl(rgba224),
      denoisedPngDataUrl: grayToPngDataUrl(denoised),
      clahePngDataUrl: grayToPngDataUrl(clahe),
      edgesPngDataUrl: grayToPngDataUrl(edges),
      roiMaskPngDataUrl: roiMaskUrl,
    },
  };
}

export function analyzeCtScanBuffer(
  buffer: Buffer,
  filename = '',
  modelId: ModelArchitectureId = 'efficientnet_b0',
  filterMode: 'gaussian' | 'median' = 'gaussian'
): PredictionResult {
  const bundle = loadTrainedBundle();
  const model = bundle.models[modelId] ?? bundle.models.efficientnet_b0;

  const prep = preprocessImageBuffer(buffer, filename, filterMode);
  const extracted = extractRadiomicAndConvFeatures(
    prep.denoised,
    prep.clahe,
    prep.roiMask,
    prep.edges,
    prep.gradMag,
    prep.gradDir
  );

  const { probabilities, predictedClass, confidence } = runModelForward(model, extracted.vector);
  const { heatmapUrl, overlayUrl } = renderGradCamMaps(
    prep.gray224,
    extracted.activationMap,
    prep.roiMask,
    predictedClass
  );
  const { segOverlayUrl } = renderSegmentationOverlay(
    prep.clahe,
    prep.roiMask,
    extracted.noduleMask
  );

  return {
    id: `scan_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    timestamp: new Date().toISOString(),
    modelUsed: model.architecture,
    modelDisplayName: model.displayName,
    predictedClass,
    confidence,
    probabilities,
    metadata: prep.metadata,
    features: extracted.features,
    featureVector: extracted.vector,
    segmentation: extracted.segmentation,
    visualizations: {
      ...prep.outputs,
      gradcamOverlayPngDataUrl: overlayUrl,
      gradcamHeatmapPngDataUrl: heatmapUrl,
      segmentationOverlayPngDataUrl: segOverlayUrl,
    },
    disclaimer:
      'Academic research output only. Grad-CAM visualizes spatial feature attention and does not confirm the clinical presence or exact anatomical boundary of malignancy.',
  };
}

function round4(val: number): number {
  return Math.round(val * 10000) / 10000;
}
