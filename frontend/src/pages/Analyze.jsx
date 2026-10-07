import { useState } from 'react';
import { ArrowUpRight, ScanLine } from 'lucide-react';
import { api, analysisForm } from '../services/api';
import {
  AnalysisCard,
  EmptyState,
  ErrorState,
  PageHeader,
  ProgressIndicator,
  ReportButtons,
  SaveOptions,
  UploadZone,
} from '../components/UI';
import { useToast } from '../components/Providers';
import { useRetention } from '../hooks/useRetention';

export default function Analyze() {
  const [files, setFiles] = useState([]);
  const [save, setSave] = useState(true);
  const [retain, setRetain] = useRetention();
  const [busy, setBusy] = useState(false);
  const [record, setRecord] = useState(null);
  const [error, setError] = useState('');
  const toast = useToast();
  async function analyze() {
    setBusy(true);
    setError('');
    setRecord(null);
    try {
      const next = await api.analyze(analysisForm(files, { save, retain }));
      setRecord(next);
      toast(
        next.saved
          ? 'Analysis complete and saved to history.'
          : 'Analysis complete. This result is session-only.',
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader title="A closer look at your image.">
        Upload a photo and examine its authenticity indicators, with the evidence to match.
      </PageHeader>
      <div className="analysis-workspace">
        <section className="panel upload-panel">
          <div className="section-heading">
            <h2>New image analysis</h2>
            <span className="badge badge-neutral">
              <ScanLine size={13} />
              ELA + FFT
            </span>
          </div>
          <UploadZone
            disabled={busy}
            files={files}
            onChange={(next) => {
              setFiles(next);
              setRecord(null);
              setError('');
            }}
          />
          <SaveOptions {...{ save, setSave, retain, setRetain }} />
          <button
            disabled={!files.length || busy}
            onClick={analyze}
            className="button button-primary button-full"
          >
            {busy ? 'Analyzing image…' : 'Analyze image'}
            <ArrowUpRight size={18} />
          </button>
          <ErrorState message={error} />
          <ProgressIndicator busy={busy} />
        </section>
        <aside className="panel guide-panel">
          <span className="eyebrow">WHAT HAPPENS NEXT</span>
          <h3>
            One image.
            <br />
            Several perspectives.
          </h3>
          <p>
            We measure how your image responds to recompression, inspect its frequency spectrum, and
            collect pixel statistics.
          </p>
          <div className="guide-step">
            <b>01</b>
            <span>An authenticity assessment</span>
          </div>
          <div className="guide-step">
            <b>02</b>
            <span>Visual forensic evidence</span>
          </div>
          <div className="guide-step">
            <b>03</b>
            <span>A report you can revisit</span>
          </div>
          <small>Heuristic indicators today. No invented AI confidence.</small>
        </aside>
      </div>
      {record ? (
        <>
          <AnalysisCard item={record.result.items[0]} />
          <ReportButtons record={record} />
        </>
      ) : (
        !busy && (
          <EmptyState
            title="Your evidence will appear here."
            text="Choose an image above to begin its analysis."
          />
        )
      )}
    </>
  );
}
