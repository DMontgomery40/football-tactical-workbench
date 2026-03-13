import BenchmarkLabShell from './benchmarkLab/BenchmarkLabShell';

interface BenchmarkLabProps {
  apiBase: string;
  helpCatalog?: unknown[];
  activePipeline?: string;
  activeDetector?: string;
}

export default function BenchmarkLab(props: BenchmarkLabProps) {
  return <BenchmarkLabShell {...props} />;
}
