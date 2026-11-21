import { useState } from 'react';
import { Award, Loader2, RefreshCw } from 'lucide-react';
import {
  ModelComparisonChart,
  RocCurveChart,
  TrainingCurveChart,
} from '../components/Charts';
import { CLASS_LABELS, ModelArchitectureId, ModelInfoResponse } from '../types/api';

interface PerformanceDashboardProps {
  modelInfo: ModelInfoResponse | null;
  isLoading: boolean;
  onRerunEvaluation: () => Promise<void>;
}

const ARCH_ORDER: ModelArchitectureId[] = [
  'resnet50',
  'efficientnet_b0',
  'vgg16',
  'custom_cnn',
];

export function PerformanceDashboard({
  modelInfo,
  isLoading,
  onRerunEvaluation,
}: PerformanceDashboardProps) {
  const [selectedArch, setSelectedArch] = useState<ModelArchitectureId>('resnet50');
  const [isReEvaluating, setIsReEvaluating] = useState(false);

  if (isLoading || !modelInfo) {
    return (
      <div className="py-20 text-center text-sm text-slate-500">
        Loading patient-level evaluation metrics...
      </div>
    );
  }

  const activeReport =
    modelInfo.models[selectedArch] || modelInfo.models[modelInfo.bestModelId];
  const split = modelInfo.splitStrategy;

  const handleRerun = async () => {
    setIsReEvaluating(true);
    try {
      await onRerunEvaluation();
    } finally {
      setIsReEvaluating(false);
    }
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Header & Re-Run Trigger */}
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="font-display text-2xl font-semibold text-slate-900">
            Model Performance & Comparative Evaluation
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            Actual held-out 15% patient-level test metrics comparing Custom CNN against
            VGG16, ResNet50, and EfficientNetB0 transfer learning architectures.
          </p>
        </div>
        <button
          type="button"
          disabled={isReEvaluating}
          onClick={() => void handleRerun()}
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-800 bg-white border border-slate-300 hover:bg-slate-100 rounded-lg transition-colors whitespace-nowrap shrink-0 disabled:opacity-50"
        >
          {isReEvaluating ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
              <span>Re-Running Training & Evaluation...</span>
            </>
          ) : (
            <>
              <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Re-Run Test Evaluation</span>
            </>
          )}
        </button>
      </div>

      {/* Patient-Level Split Integrity Banner */}
      <section className="bg-white border border-slate-200 rounded-xl p-5 flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="space-y-1">
          <p className="font-semibold text-slate-900 text-sm">
            Patient-Level Split Protocol (Zero Leakage)
          </p>
          <p className="text-slate-600">{modelInfo.datasetProtocol}</p>
        </div>
        <div className="flex flex-wrap items-center gap-6 font-mono tabular-nums">
          <div>
            <span className="text-slate-500">Train (70%): </span>
            <span className="font-semibold text-slate-900">
              {split.trainPatients} pts / {split.trainSlices} slices
            </span>
          </div>
          <div>
            <span className="text-slate-500">Val (15%): </span>
            <span className="font-semibold text-slate-900">
              {split.valPatients} pts / {split.valSlices} slices
            </span>
          </div>
          <div>
            <span className="text-slate-500">Held-Out Test (15%): </span>
            <span className="font-semibold text-sky-700">
              {split.testPatients} pts / {split.testSlices} slices
            </span>
          </div>
        </div>
      </section>

      {/* Architecture Selector Tabs */}
      <div className="flex flex-wrap items-center gap-2 p-1.5 bg-slate-100 rounded-xl border border-slate-200">
        {ARCH_ORDER.map((id) => {
          const rep = modelInfo.models[id];
          const isActive = selectedArch === id;
          const isBest = modelInfo.bestModelId === id;
          return (
            <button
              key={id}
              type="button"
              onClick={() => setSelectedArch(id)}
              className={`flex-1 min-w-[160px] px-4 py-2.5 rounded-lg text-left transition-colors ${
                isActive
                  ? 'bg-[#0B1324] text-white'
                  : 'text-slate-700 hover:bg-white/70'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold whitespace-nowrap">
                  {rep.displayName.split(' (')[0]}
                </span>
                {isBest && (
                  <Award
                    className={`h-3.5 w-3.5 shrink-0 ${
                      isActive ? 'text-sky-400' : 'text-sky-600'
                    }`}
                    aria-label="Best Model"
                  />
                )}
              </div>
              <p
                className={`mt-0.5 text-[11px] font-mono tabular-nums ${
                  isActive ? 'text-sky-300' : 'text-slate-500'
                }`}
              >
                Acc {(rep.accuracy * 100).toFixed(1)}% · AUC {rep.macroAuc.toFixed(3)}
              </p>
            </button>
          );
        })}
      </div>

      {/* Selected Model Medical KPI Row */}
      <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        {[
          { label: 'Test Accuracy', val: `${(activeReport.accuracy * 100).toFixed(1)}%` },
          {
            label: 'Macro Sensitivity',
            val: `${(activeReport.macroSensitivity * 100).toFixed(1)}%`,
          },
          {
            label: 'Macro Specificity',
            val: `${(activeReport.macroSpecificity * 100).toFixed(1)}%`,
          },
          {
            label: 'Macro Precision',
            val: `${(activeReport.macroPrecision * 100).toFixed(1)}%`,
          },
          {
            label: 'Macro F1-Score',
            val: `${(activeReport.macroF1 * 100).toFixed(1)}%`,
          },
          { label: 'Macro ROC-AUC', val: activeReport.macroAuc.toFixed(4) },
        ].map((kpi) => (
          <div
            key={kpi.label}
            className="bg-white border border-slate-200 rounded-xl p-4"
          >
            <p className="text-xs text-slate-500">{kpi.label}</p>
            <p className="mt-1.5 font-mono tabular-nums text-2xl font-bold text-slate-900">
              {kpi.val}
            </p>
          </div>
        ))}
      </section>

      {/* CNN vs Transfer Learning Comparison + ROC Curve */}
      <section className="grid lg:grid-cols-12 gap-8">
        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-900">
              Custom CNN vs. Transfer Learning Comparison
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Test accuracy, sensitivity, specificity, and F1-score across all 4
              architectures.
            </p>
          </div>
          <ModelComparisonChart models={modelInfo.models} />
        </div>

        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-900">
              Multi-Class ROC Curve ({activeReport.displayName.split(' (')[0]})
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              One-vs-Rest Receiver Operating Characteristic curves on the held-out patient
              test set.
            </p>
          </div>
          <RocCurveChart perClassMetrics={activeReport.perClassMetrics} />
        </div>
      </section>

      {/* Training & Validation Learning Curves */}
      <section className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="font-display text-base font-semibold text-slate-900">
              Training vs. Validation Accuracy
            </h2>
            <span className="text-xs font-mono text-slate-500">
              Best Epoch: {activeReport.bestEpoch} / {activeReport.epochsTrained}
            </span>
          </div>
          <TrainingCurveChart history={activeReport.history} metric="accuracy" />
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="font-display text-base font-semibold text-slate-900">
              Training vs. Validation Loss
            </h2>
            <span className="text-xs font-mono text-slate-500">
              {activeReport.inputNormalization}
            </span>
          </div>
          <TrainingCurveChart history={activeReport.history} metric="loss" />
        </div>
      </section>

      {/* Confusion Matrix & Class-Wise Medical Metrics Report */}
      <section className="grid lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-900">
              Test Set Confusion Matrix
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Rows = Ground Truth · Columns = Predicted Class ({split.testSlices} test
              slices)
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-xs">
              <thead>
                <tr>
                  <th className="p-2 text-left text-slate-500 font-medium">
                    True \ Pred
                  </th>
                  {CLASS_LABELS.map((c) => (
                    <th
                      key={c}
                      className="p-2 text-center font-semibold text-slate-700"
                    >
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {CLASS_LABELS.map((rowCls, rIdx) => (
                  <tr key={rowCls} className="border-t border-slate-200">
                    <td className="p-2.5 font-semibold text-slate-700">{rowCls}</td>
                    {CLASS_LABELS.map((colCls, cIdx) => {
                      const count = activeReport.confusionMatrix[rIdx][cIdx];
                      const isDiag = rIdx === cIdx;
                      return (
                        <td key={colCls} className="p-1.5 text-center">
                          <div
                            className={`py-3 rounded-lg font-mono tabular-nums text-base font-bold ${
                              isDiag
                                ? 'bg-[#0B1324] text-sky-300'
                                : count > 0
                                ? 'bg-rose-50 text-rose-800 border border-rose-200'
                                : 'bg-slate-50 text-slate-400'
                            }`}
                          >
                            {count}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-900">
              Class-Wise Classification Report
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Medical evaluation emphasizes per-class sensitivity and specificity rather
              than accuracy alone.
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2.5 pr-3 font-medium">Category</th>
                  <th className="py-2.5 px-3 text-right font-medium">Precision</th>
                  <th className="py-2.5 px-3 text-right font-medium">
                    Sensitivity (Recall)
                  </th>
                  <th className="py-2.5 px-3 text-right font-medium">Specificity</th>
                  <th className="py-2.5 px-3 text-right font-medium">F1-Score</th>
                  <th className="py-2.5 px-3 text-right font-medium">ROC-AUC</th>
                  <th className="py-2.5 pl-3 text-right font-medium">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                {activeReport.perClassMetrics.map((m) => (
                  <tr key={m.className} className="hover:bg-slate-50">
                    <td className="py-3 pr-3 font-sans font-semibold text-slate-900">
                      {m.className}
                    </td>
                    <td className="py-3 px-3 text-right text-slate-700">
                      {(m.precision * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-right font-semibold text-sky-700">
                      {(m.sensitivity * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-right font-semibold text-slate-900">
                      {(m.specificity * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-right text-slate-700">
                      {(m.f1Score * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-right text-slate-700">
                      {m.auc.toFixed(4)}
                    </td>
                    <td className="py-3 pl-3 text-right text-slate-500">
                      {m.support}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
}
