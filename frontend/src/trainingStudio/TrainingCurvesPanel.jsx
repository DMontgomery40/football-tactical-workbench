import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import {
  expandChartDomain,
  formatCurveAxis,
  formatCurveTooltipLabel,
  formatCurveValue,
  formatLearningRate,
} from './formatters';

export default function TrainingCurvesPanel({ trainingCurves }) {
  const lossData = Array.isArray(trainingCurves?.loss) ? trainingCurves.loss : [];
  const optimizerData = Array.isArray(trainingCurves?.optimizer) ? trainingCurves.optimizer : [];
  const sharedCurveProps = {
    type: 'monotone',
    isAnimationActive: false,
    strokeWidth: 2.8,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
    dot: { r: 2.6, strokeWidth: 1.6, fill: 'var(--surface)' },
    activeDot: { r: 4.4, strokeWidth: 1.8, fill: 'var(--surface)' },
    connectNulls: true,
  };
  const tooltipStyle = {
    borderRadius: 14,
    border: '1px solid var(--line-strong)',
    background: 'color-mix(in srgb, var(--surface) 94%, var(--page-panel))',
    boxShadow: '0 14px 28px rgba(15, 23, 42, 0.18)',
    color: 'var(--text)',
  };

  if (lossData.length === 0 && optimizerData.length === 0) {
    return null;
  }

  return (
    <section className="studio-chart-grid">
      <div className="card studio-chart-card">
        <div className="row-between">
          <div>
            <div className="micro-label">Live loss curves</div>
            <div className="section-title">Detection losses</div>
          </div>
          <div className="muted">{lossData.length} samples</div>
        </div>
        <div className="studio-chart-shell">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lossData} margin={{ top: 12, right: 18, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis
                type="number"
                dataKey="epoch_progress"
                domain={['dataMin', 'dataMax']}
                tickFormatter={formatCurveAxis}
                tickCount={6}
                minTickGap={24}
                stroke="currentColor"
              />
              <YAxis stroke="currentColor" width={58} domain={expandChartDomain} tickFormatter={formatCurveValue} />
              <Tooltip
                formatter={(value) => formatCurveValue(value)}
                labelFormatter={formatCurveTooltipLabel}
                contentStyle={tooltipStyle}
              />
              <Legend iconType="circle" iconSize={8} />
              <Line {...sharedCurveProps} dataKey="box_loss" name="box" stroke="var(--chart-box)" />
              <Line {...sharedCurveProps} dataKey="cls_loss" name="cls" stroke="var(--chart-cls)" />
              <Line {...sharedCurveProps} dataKey="dfl_loss" name="dfl" stroke="var(--chart-dfl)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card studio-chart-card">
        <div className="row-between">
          <div>
            <div className="micro-label">Live optimizer signals</div>
            <div className="section-title">Gradient norm and learning rate</div>
          </div>
          <div className="muted">{optimizerData.length} samples</div>
        </div>
        <div className="studio-chart-shell">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={optimizerData} margin={{ top: 12, right: 18, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis
                type="number"
                dataKey="epoch_progress"
                domain={['dataMin', 'dataMax']}
                tickFormatter={formatCurveAxis}
                tickCount={6}
                minTickGap={24}
                stroke="currentColor"
              />
              <YAxis yAxisId="left" stroke="currentColor" width={58} domain={expandChartDomain} tickFormatter={formatCurveValue} />
              <YAxis yAxisId="right" orientation="right" stroke="currentColor" width={64} domain={expandChartDomain} tickFormatter={formatLearningRate} />
              <Tooltip
                formatter={(value, name) => (
                  name === 'lr' ? formatLearningRate(value) : formatCurveValue(value)
                )}
                labelFormatter={formatCurveTooltipLabel}
                contentStyle={tooltipStyle}
              />
              <Legend iconType="circle" iconSize={8} />
              <Line {...sharedCurveProps} yAxisId="left" dataKey="grad_norm" name="grad norm" stroke="var(--chart-grad)" />
              <Line {...sharedCurveProps} yAxisId="right" dataKey="lr" name="lr" stroke="var(--chart-lr)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}
