import { useEffect, useRef, useState } from 'react';
import { Layers, ArrowUpRight } from 'lucide-react';
import { streamBatch } from '../services/api';
import {
  AnalysisCard,
  ErrorState,
  ImagePreview,
  PageHeader,
  ProgressIndicator,
  ReportButtons,
  SaveOptions,
  UploadZone,
} from '../components/UI';
import { useToast } from '../components/Providers';
import { useRetention } from '../hooks/useRetention';

export function BatchTable({ items, onSelect }) {
  return (
    <div className="batch-grid">
      {items.map((item, index) => (
        <button
          className="panel batch-card"
          key={`${item.filename}-${index}`}
          onClick={() => onSelect(index)}
        >
          <ImagePreview src={item.visualizations?.original} alt={item.filename} />
          <strong title={item.filename}>{item.filename}</strong>
          <span className="badge badge-neutral">{item.status}</span>
          {item.classification ? (
            <>
              <h3>{item.classification.label}</h3>
              <small>
                AI indicators: {item.classification.ai_indicators} · Manipulation: not established
              </small>
              <div className="batch-metrics">
                <span>ELA {item.metrics.ela_mean.toFixed(2)}</span>
                <span>FFT {item.metrics.frequency_mean.toFixed(2)}</span>
              </div>
            </>
          ) : (
            <p>{item.error}</p>
          )}
        </button>
      ))}
    </div>
  );
}
export default function Batch() {
  const [files, setFiles] = useState([]);
  const [save, setSave] = useState(true);
  const [retain, setRetain] = useRetention();
  const [busy, setBusy] = useState(false);
  const [items, setItems] = useState([]);
  const [record, setRecord] = useState(null);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState('');
  const controller = useRef(null);
  const toast = useToast();
  useEffect(() => () => controller.current?.abort(), []);
  async function run() {
    setBusy(true);
    setItems([]);
    setRecord(null);
    setSelected(null);
    setError('');
    controller.current = new AbortController();
    try {
      const next = await streamBatch(
        files,
        { save, retain },
        (progress) => setItems((current) => [...current, progress.item]),
        controller.current.signal,
      );
      setRecord(next);
      toast('Batch finished. Review each image and export the report.');
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader title="A collection, examined together.">
        Compare patterns across your images. Every item gets its own forensic evidence.
      </PageHeader>
      <section className="panel upload-panel">
        <UploadZone
          disabled={busy}
          files={files}
          onChange={(next) => {
            setFiles(next);
            setItems([]);
            setRecord(null);
          }}
          multiple
          label="Bring your image collection"
        />
        <SaveOptions {...{ save, setSave, retain, setRetain }} />
        <button className="button button-primary" disabled={!files.length || busy} onClick={run}>
          <Layers size={17} />
          {busy ? 'Analyzing batch…' : 'Analyze all images'}
          <ArrowUpRight size={17} />
        </button>
        <ErrorState message={error} />
        <ProgressIndicator busy={busy} completed={items.length} total={files.length} />
      </section>
      {!!items.length && (
        <>
          <div className="section-heading">
            <h2>Batch results</h2>
            <span>
              {items.length} / {files.length} processed
            </span>
          </div>
          <BatchTable items={items} onSelect={setSelected} />
        </>
      )}
      {record && <ReportButtons record={record} />}{' '}
      {selected !== null && items[selected] && <AnalysisCard item={items[selected]} />}
    </>
  );
}
