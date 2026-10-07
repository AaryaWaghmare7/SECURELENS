import { useState } from 'react';
import { GitCompareArrows } from 'lucide-react';
import { api } from '../services/api';
import {
  ErrorState,
  ForensicViewer,
  MetricCard,
  PageHeader,
  ProgressIndicator,
  ReportButtons,
  ResultCard,
  SaveOptions,
  UploadZone,
} from '../components/UI';
import { useRetention } from '../hooks/useRetention';

export function ComparePanel({ record }) {
  const [a, b] = record.result.items;
  const comparison = record.result.comparison;
  return (
    <>
      <div className="panel comparison-summary">
        <span className="eyebrow">VISUAL SIMILARITY, NOT AI CONFIDENCE</span>
        <h2>{comparison.similarity.toFixed(1)} / 100</h2>
        <p>{comparison.method}</p>
        <div className="metric-grid">
          {[
            ['Histogram similarity', comparison.histogram_similarity],
            ['Edge similarity', comparison.edge_similarity],
            ['Pixel similarity', comparison.pixel_similarity],
          ].map(([label, value]) => (
            <MetricCard key={label} label={label} value={`${value.toFixed(1)} / 100`} />
          ))}
        </div>
      </div>
      <div className="comparison-grid">
        {[a, b].map((item, i) => (
          <div key={i}>
            <h2>Image {i === 0 ? 'A' : 'B'}</h2>
            <ResultCard item={item} />
            <ForensicViewer item={item} />
          </div>
        ))}
      </div>
      <section className="panel table-scroll">
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Image A</th>
              <th>Image B</th>
            </tr>
          </thead>
          <tbody>
            {[
              'width',
              'height',
              'channels',
              'analysis_width',
              'analysis_height',
              'brightness',
              'ela_mean',
              'ela_std',
              'frequency_mean',
              'frequency_std',
              'ela_raw_mean',
              'format',
              'mode',
              'file_size_bytes',
              'has_exif',
            ].map((key) => (
              <tr key={key}>
                <th>{key.replaceAll('_', ' ')}</th>
                {[a, b].map((item, i) => (
                  <td key={i}>
                    {typeof item.metrics[key] === 'number'
                      ? Number(item.metrics[key]).toFixed(2)
                      : String(item.metrics[key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <h3>JPEG compression comparison</h3>
        <p>
          Mean absolute pixel loss and deviation after recompression, not source compression
          history.
        </p>
        <table>
          <thead>
            <tr>
              <th>JPEG quality</th>
              <th>Image A: mean / deviation</th>
              <th>Image B: mean / deviation</th>
            </tr>
          </thead>
          <tbody>
            {[90, 50, 20].map((quality) => (
              <tr key={quality}>
                <th>{quality}</th>
                {[a, b].map((item, index) => {
                  const loss = item.metrics.compression.find((row) => row.quality === quality);
                  return (
                    <td key={index}>
                      {loss.mean_abs_difference.toFixed(3)} / {loss.std_abs_difference.toFixed(3)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
export default function Compare() {
  const [a, setA] = useState([]);
  const [b, setB] = useState([]);
  const [save, setSave] = useState(true);
  const [retain, setRetain] = useRetention();
  const [record, setRecord] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function run() {
    setBusy(true);
    setRecord(null);
    setError('');
    const data = new FormData();
    data.append('image_a', a[0]);
    data.append('image_b', b[0]);
    data.append('save', String(save));
    data.append('retain_images', String(retain));
    try {
      setRecord(await api.compare(data));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader title="Two images. One clearer picture.">
        Compare image statistics, forensic evidence and measured visual similarity side by side.
      </PageHeader>
      <section className="panel upload-panel">
        <div className="comparison-grid">
          <UploadZone
            disabled={busy}
            files={a}
            onChange={(next) => {
              setA(next);
              setRecord(null);
            }}
            label="Image A"
            compact
          />
          <UploadZone
            disabled={busy}
            files={b}
            onChange={(next) => {
              setB(next);
              setRecord(null);
            }}
            label="Image B"
            compact
          />
        </div>
        <SaveOptions {...{ save, setSave, retain, setRetain }} />
        <button
          className="button button-primary"
          onClick={run}
          disabled={!a.length || !b.length || busy}
        >
          <GitCompareArrows size={18} />
          Compare images
        </button>
        <ErrorState message={error} />
        <ProgressIndicator busy={busy} />
      </section>
      {record && (
        <>
          <ComparePanel record={record} />
          <ReportButtons record={record} />
        </>
      )}
    </>
  );
}
