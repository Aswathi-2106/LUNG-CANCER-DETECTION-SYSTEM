import fs from 'node:fs';
import path from 'node:path';
import express, { Request, Response, NextFunction } from 'express';
import multer from 'multer';
import {
  analyzeCtScanBuffer,
  clearBundleCache,
  loadTrainedBundle,
  ModelArchitectureId,
  PredictionResult,
  preprocessImageBuffer,
} from './server/cvEngine';
import { EvaluationBundle, trainAndEvaluateAll } from './scripts/train_web_model';

const PORT = Number(process.env.PORT) || 3000;
const ARTIFACTS_DIR = path.resolve(process.cwd(), 'server/artifacts');
const HISTORY_FILE = path.join(ARTIFACTS_DIR, 'history.json');
const EVAL_FILE = path.resolve(process.cwd(), 'results/evaluation_metrics.json');

interface HistoryEntry {
  id: string;
  timestamp: string;
  scanLabel: string;
  modelUsed: ModelArchitectureId;
  modelDisplayName: string;
  predictedClass: 'Normal' | 'Benign' | 'Malignant';
  confidence: number;
  probabilities: Record<'Normal' | 'Benign' | 'Malignant', number>;
  thumbnailDataUrl: string;
  gradcamOverlayDataUrl: string;
  preprocessedDataUrl: string;
  segmentationOverlayDataUrl: string;
  metadata: PredictionResult['metadata'];
  features: PredictionResult['features'];
  segmentation: PredictionResult['segmentation'];
}

function ensureArtifactsBootstrapped(): void {
  fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
  const bundlePath = path.join(ARTIFACTS_DIR, 'trained_models.json');
  if (!fs.existsSync(bundlePath) || !fs.existsSync(EVAL_FILE)) {
    trainAndEvaluateAll();
  }
  if (!fs.existsSync(HISTORY_FILE)) {
    fs.writeFileSync(HISTORY_FILE, JSON.stringify([], null, 2));
  }
}

function readHistory(): HistoryEntry[] {
  try {
    if (!fs.existsSync(HISTORY_FILE)) return [];
    const raw = fs.readFileSync(HISTORY_FILE, 'utf-8');
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeHistory(entries: HistoryEntry[]): void {
  fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
  fs.writeFileSync(HISTORY_FILE, JSON.stringify(entries.slice(0, 50), null, 2));
}

function getSampleScans(): {
  id: string;
  title: string;
  categoryHint: string;
  description: string;
  imageUrl: string;
  filename: string;
  filePath: string;
}[] {
  const srcImgDir = path.resolve(process.cwd(), 'src/assets/images');
  const distImgDir = path.resolve(process.cwd(), 'dist/src/assets/images');
  const imgDir = fs.existsSync(srcImgDir) ? srcImgDir : distImgDir;
  if (!fs.existsSync(imgDir)) return [];
  const files = fs.readdirSync(imgDir);

  const defs = [
    {
      id: 'sample_normal',
      prefix: 'sample_ct_normal_',
      title: 'Normal Thoracic Parenchyma',
      categoryHint: 'Normal Reference',
      description: 'Symmetrical axial lung window slice with clear vasculature and no focal pulmonary lesions.',
    },
    {
      id: 'sample_benign',
      prefix: 'sample_ct_benign_',
      title: 'Solitary Circumscribed Nodule',
      categoryHint: 'Benign Reference',
      description: 'Well-circumscribed smooth round solitary nodule in the right lung periphery.',
    },
    {
      id: 'sample_malignant',
      prefix: 'sample_ct_malignant_',
      title: 'Spiculated Pulmonary Mass',
      categoryHint: 'Malignant Reference',
      description: 'Irregular spiculated mass with radiating strands and pleural retraction in the left upper lobe.',
    },
  ];

  const samples: {
    id: string;
    title: string;
    categoryHint: string;
    description: string;
    imageUrl: string;
    filename: string;
    filePath: string;
  }[] = [];

  for (const d of defs) {
    const match = files.find((f) => f.startsWith(d.prefix));
    if (match) {
      samples.push({
        id: d.id,
        title: d.title,
        categoryHint: d.categoryHint,
        description: d.description,
        imageUrl: `/src/assets/images/${match}`,
        filename: match,
        filePath: path.join(imgDir, match),
      });
    }
  }
  return samples;
}

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const lower = file.originalname.toLowerCase();
    const allowedExt =
      lower.endsWith('.png') ||
      lower.endsWith('.jpg') ||
      lower.endsWith('.jpeg') ||
      lower.endsWith('.dcm');
    const allowedMime = [
      'image/png',
      'image/jpeg',
      'image/jpg',
      'application/dicom',
      'application/octet-stream',
    ].includes(file.mimetype);

    if (allowedExt || allowedMime) {
      cb(null, true);
    } else {
      cb(
        new Error(
          'Invalid file type. Only PNG, JPG, JPEG, and uncompressed DICOM (.dcm) thoracic CT images are supported.'
        )
      );
    }
  },
});

function resolveImagePayload(req: Request): { buffer: Buffer; filename: string; scanLabel: string } {
  if (req.file) {
    // Strip any potential patient name from uploaded filename display
    const ext = path.extname(req.file.originalname) || '.png';
    return {
      buffer: req.file.buffer,
      filename: req.file.originalname,
      scanLabel: `Uploaded CT Slice (${ext.replace('.', '').toUpperCase()})`,
    };
  }

  const { sampleId, imageBase64, filename } = (req.body ?? {}) as {
    sampleId?: string;
    imageBase64?: string;
    filename?: string;
  };

  if (sampleId) {
    const sample = getSampleScans().find((s) => s.id === sampleId);
    if (!sample) {
      throw new Error(`Sample scan "${sampleId}" was not found.`);
    }
    return {
      buffer: fs.readFileSync(sample.filePath),
      filename: sample.filename,
      scanLabel: sample.title,
    };
  }

  if (imageBase64) {
    const cleaned = imageBase64.replace(/^data:[^;]+;base64,/, '');
    const buffer = Buffer.from(cleaned, 'base64');
    return {
      buffer,
      filename: filename || 'upload.png',
      scanLabel: 'Uploaded CT Slice',
    };
  }

  throw new Error('No CT scan image provided. Upload an image file or select a sample CT scan.');
}

async function startServer() {
  ensureArtifactsBootstrapped();

  const app = express();
  app.use(express.json({ limit: '20mb' }));
  app.use(express.urlencoded({ extended: true, limit: '20mb' }));

  const router = express.Router();

  router.get('/health', (_req: Request, res: Response) => {
    const bundle = loadTrainedBundle();
    res.json({
      status: 'ok',
      service: 'PulmoVision AI Lung CT Inference Server',
      runtime: 'Node.js / OpenCV-Equivalent CV Pipeline + Trained Neural Classifiers',
      modelsLoaded: Object.keys(bundle.models),
      trainedAt: bundle.trainedAt,
    });
  });

  router.get('/samples', (_req: Request, res: Response) => {
    const samples = getSampleScans().map(({ filePath: _fp, ...rest }) => rest);
    res.json({ samples });
  });

  router.get('/model-info', (_req: Request, res: Response) => {
    const bundle = loadTrainedBundle();
    const evalRaw = fs.readFileSync(EVAL_FILE, 'utf-8');
    const evalBundle = JSON.parse(evalRaw) as EvaluationBundle;

    res.json({
      version: bundle.version,
      trainedAt: bundle.trainedAt,
      datasetProtocol: bundle.datasetProtocol,
      splitStrategy: evalBundle.splitStrategy,
      bestModelId: evalBundle.bestModelId,
      models: evalBundle.models,
      featureNames: [
        'Mean Intensity',
        'Std Intensity',
        'Skewness',
        'Kurtosis',
        'Shannon Entropy',
        'GLCM Contrast',
        'GLCM Dissimilarity',
        'GLCM Homogeneity',
        'GLCM Energy',
        'GLCM Correlation',
        'Canny Edge Density',
        'HOG Dominant Energy',
        'HOG Orientation Spread',
        'Lung ROI Coverage',
        'Candidate Nodule Count',
        'Max Lesion Area Ratio',
        'Lesion Circularity',
        'Spiculation Index',
        'Focal Contrast Ratio',
        'Bilateral Asymmetry',
        'Conv Gabor Response',
        'Conv Laplacian Peak',
        'Conv Blob Max Pool',
        'Conv Texture Mean Pool',
      ],
    });
  });

  router.post('/evaluate', (_req: Request, res: Response) => {
    try {
      clearBundleCache();
      const { evalBundle } = trainAndEvaluateAll();
      res.json({
        message: 'Re-trained and evaluated all architectures on the 70/15/15 patient-level split.',
        evaluation: evalBundle,
      });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Evaluation failed.',
      });
    }
  });

  router.post(
    '/preprocess',
    upload.single('image'),
    (req: Request, res: Response) => {
      try {
        const { buffer, filename } = resolveImagePayload(req);
        const filterMode =
          req.body?.filterMode === 'median' ? 'median' : 'gaussian';
        const prep = preprocessImageBuffer(buffer, filename, filterMode);
        res.json({
          metadata: prep.metadata,
          filterMode,
          outputs: prep.outputs,
        });
      } catch (err) {
        res.status(400).json({
          error: err instanceof Error ? err.message : 'Image preprocessing failed.',
        });
      }
    }
  );

  router.post(
    '/predict',
    upload.single('image'),
    (req: Request, res: Response) => {
      try {
        const { buffer, filename, scanLabel } = resolveImagePayload(req);
        const modelId = (req.body?.modelId as ModelArchitectureId) || 'resnet50';
        const filterMode =
          req.body?.filterMode === 'median' ? 'median' : 'gaussian';

        const result = analyzeCtScanBuffer(buffer, filename, modelId, filterMode);

        // Save anonymized record to prediction history
        const entry: HistoryEntry = {
          id: result.id,
          timestamp: result.timestamp,
          scanLabel,
          modelUsed: result.modelUsed,
          modelDisplayName: result.modelDisplayName,
          predictedClass: result.predictedClass,
          confidence: result.confidence,
          probabilities: result.probabilities,
          thumbnailDataUrl: result.visualizations.originalPngDataUrl,
          gradcamOverlayDataUrl: result.visualizations.gradcamOverlayPngDataUrl,
          preprocessedDataUrl: result.visualizations.clahePngDataUrl,
          segmentationOverlayDataUrl:
            result.visualizations.segmentationOverlayPngDataUrl,
          metadata: result.metadata,
          features: result.features,
          segmentation: result.segmentation,
        };

        const current = readHistory();
        writeHistory([entry, ...current]);

        res.json({
          ...result,
          scanLabel,
        });
      } catch (err) {
        res.status(400).json({
          error: err instanceof Error ? err.message : 'Prediction failed.',
        });
      }
    }
  );

  router.get('/history', (_req: Request, res: Response) => {
    const history = readHistory();
    res.json({ count: history.length, history });
  });

  router.delete('/history/:id', (req: Request, res: Response) => {
    const { id } = req.params;
    const history = readHistory();
    const filtered = history.filter((item) => item.id !== id);
    if (filtered.length === history.length) {
      res.status(404).json({ error: 'Prediction record not found.' });
      return;
    }
    writeHistory(filtered);
    res.json({ deletedId: id, remaining: filtered.length });
  });

  router.delete('/history', (_req: Request, res: Response) => {
    writeHistory([]);
    res.json({ cleared: true, remaining: 0 });
  });

  // Mount under both /api/* and root REST endpoints requested in the specification
  app.use('/api', router);
  app.use('/', router);

  // Express error middleware (e.g., Multer upload size/type errors)
  app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
    res.status(400).json({ error: err.message || 'Invalid request payload.' });
  });

  // Frontend Vite middleware in dev / static dist in production
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`PulmoVision AI server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Fatal server startup error:', err);
  process.exit(1);
});
