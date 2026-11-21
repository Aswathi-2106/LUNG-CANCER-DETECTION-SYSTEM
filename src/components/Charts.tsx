import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  ChartOptions,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';
import {
  ClassMetrics,
  EpochHistory,
  LungClass,
  ModelArchitectureId,
  ModelEvaluationReport,
} from '../types/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler
);

const BASE_FONT = {
  family: "'DM Sans', system-ui, sans-serif",
  size: 12,
};

export function ProbabilityBarChart({
  probabilities,
}: {
  probabilities: Record<LungClass, number>;
}) {
  const labels: LungClass[] = ['Normal', 'Benign', 'Malignant'];
  const values = labels.map((c) => Number((probabilities[c] * 100).toFixed(2)));

  const data = {
    labels,
    datasets: [
      {
        label: 'Class Probability (%)',
        data: values,
        backgroundColor: ['#0284c7', '#0d9488', '#dc2626'],
        borderRadius: 6,
        barThickness: 28,
      },
    ],
  };

  const options: ChartOptions<'bar'> = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.parsed.x?.toFixed(2)}%`,
        },
      },
    },
    scales: {
      x: {
        min: 0,
        max: 100,
        grid: { color: '#e2e8f0' },
        ticks: {
          color: '#475569',
          font: BASE_FONT,
          callback: (val) => `${val}%`,
        },
      },
      y: {
        grid: { display: false },
        ticks: {
          color: '#0f172a',
          font: { ...BASE_FONT, weight: 600 },
        },
      },
    },
  };

  return (
    <div className="h-44 w-full">
      <Bar data={data} options={options} />
    </div>
  );
}

export function ModelComparisonChart({
  models,
}: {
  models: Record<ModelArchitectureId, ModelEvaluationReport>;
}) {
  const order: ModelArchitectureId[] = [
    'custom_cnn',
    'vgg16',
    'resnet50',
    'efficientnet_b0',
  ];
  const shortNames: Record<ModelArchitectureId, string> = {
    custom_cnn: 'Custom CNN',
    vgg16: 'VGG16',
    resnet50: 'ResNet50',
    efficientnet_b0: 'EfficientNetB0',
  };

  const labels = order.map((id) => shortNames[id]);

  const data = {
    labels,
    datasets: [
      {
        label: 'Accuracy (%)',
        data: order.map((id) => Number((models[id].accuracy * 100).toFixed(2))),
        backgroundColor: '#0b1324',
        borderRadius: 4,
      },
      {
        label: 'Sensitivity (%)',
        data: order.map((id) => Number((models[id].macroSensitivity * 100).toFixed(2))),
        backgroundColor: '#0284c7',
        borderRadius: 4,
      },
      {
        label: 'Specificity (%)',
        data: order.map((id) => Number((models[id].macroSpecificity * 100).toFixed(2))),
        backgroundColor: '#38bdf8',
        borderRadius: 4,
      },
      {
        label: 'F1-Score (%)',
        data: order.map((id) => Number((models[id].macroF1 * 100).toFixed(2))),
        backgroundColor: '#0d9488',
        borderRadius: 4,
      },
    ],
  };

  const options: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#334155',
          font: BASE_FONT,
          boxWidth: 12,
          padding: 16,
        },
      },
    },
    scales: {
      y: {
        min: 50,
        max: 100,
        grid: { color: '#e2e8f0' },
        ticks: {
          color: '#475569',
          font: BASE_FONT,
          callback: (val) => `${val}%`,
        },
      },
      x: {
        grid: { display: false },
        ticks: {
          color: '#0f172a',
          font: { ...BASE_FONT, weight: 600 },
        },
      },
    },
  };

  return (
    <div className="h-72 w-full">
      <Bar data={data} options={options} />
    </div>
  );
}

export function TrainingCurveChart({
  history,
  metric,
}: {
  history: EpochHistory[];
  metric: 'accuracy' | 'loss';
}) {
  const labels = history.map((h) => `E${h.epoch}`);
  const isAcc = metric === 'accuracy';

  const data = {
    labels,
    datasets: [
      {
        label: isAcc ? 'Training Accuracy (%)' : 'Training Loss',
        data: history.map((h) =>
          isAcc ? Number((h.trainAccuracy * 100).toFixed(2)) : h.trainLoss
        ),
        borderColor: '#0284c7',
        backgroundColor: 'rgba(2, 132, 199, 0.08)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.25,
        fill: true,
      },
      {
        label: isAcc ? 'Validation Accuracy (%)' : 'Validation Loss',
        data: history.map((h) =>
          isAcc ? Number((h.valAccuracy * 100).toFixed(2)) : h.valLoss
        ),
        borderColor: '#0b1324',
        borderDash: [5, 4],
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.25,
        fill: false,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'bottom',
        labels: { color: '#334155', font: BASE_FONT, boxWidth: 14, padding: 14 },
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: '#64748b', maxTicksLimit: 8, font: BASE_FONT },
      },
      y: {
        min: isAcc ? 30 : 0,
        max: isAcc ? 102 : undefined,
        grid: { color: '#e2e8f0' },
        ticks: {
          color: '#475569',
          font: BASE_FONT,
          callback: (val) => (isAcc ? `${val}%` : `${val}`),
        },
      },
    },
  };

  return (
    <div className="h-64 w-full">
      <Line data={data} options={options} />
    </div>
  );
}

export function RocCurveChart({
  perClassMetrics,
}: {
  perClassMetrics: ClassMetrics[];
}) {
  const colors: Record<LungClass, string> = {
    Normal: '#0284c7',
    Benign: '#0d9488',
    Malignant: '#dc2626',
  };

  const datasets = [
    ...perClassMetrics.map((m) => ({
      label: `${m.className} (AUC = ${m.auc.toFixed(3)})`,
      data: m.rocPoints.map((pt) => ({ x: pt.fpr, y: pt.tpr })),
      borderColor: colors[m.className],
      backgroundColor: 'transparent',
      borderWidth: 2.2,
      pointRadius: 2,
      tension: 0,
    })),
    {
      label: 'Random Classifier (AUC = 0.500)',
      data: [
        { x: 0, y: 0 },
        { x: 1, y: 1 },
      ],
      borderColor: '#94a3b8',
      borderDash: [6, 4],
      borderWidth: 1.5,
      pointRadius: 0,
    },
  ];

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: { color: '#334155', font: BASE_FONT, boxWidth: 14, padding: 14 },
      },
    },
    scales: {
      x: {
        type: 'linear',
        min: 0,
        max: 1,
        title: {
          display: true,
          text: 'False Positive Rate (1 - Specificity)',
          color: '#475569',
          font: BASE_FONT,
        },
        grid: { color: '#e2e8f0' },
        ticks: { color: '#475569', font: BASE_FONT },
      },
      y: {
        type: 'linear',
        min: 0,
        max: 1.02,
        title: {
          display: true,
          text: 'True Positive Rate (Sensitivity)',
          color: '#475569',
          font: BASE_FONT,
        },
        grid: { color: '#e2e8f0' },
        ticks: { color: '#475569', font: BASE_FONT },
      },
    },
  };

  return (
    <div className="h-72 w-full">
      <Line data={{ datasets }} options={options} />
    </div>
  );
}
