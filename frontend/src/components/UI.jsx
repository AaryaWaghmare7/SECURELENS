import { useEffect, useId, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  UploadCloud,
  Image as ImageIcon,
  LoaderCircle,
  Download,
  ArrowUpRight,
  X,
  AlertCircle,
  ScanLine,
  FileSearch,
  Check,
} from 'lucide-react';
import { imageUrl, exportReport } from '../services/api';
import { useToast } from './Providers';

export function PageHeader({ eyebrow = 'YOUR SECURELENS WORKSPACE', title, children, action }) {
  return (
    <div className="page-header">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{children}</p>
      </div>
      {action}
    </div>
  );
}
export function ErrorState({ message }) {
  return message ? (
    <div role="alert" className="error-state">
      <AlertCircle size={18} />
      {message}
    </div>
  ) : null;
}
export function EmptyState({ title, text, action }) {
  return (
    <div className="empty-state">
      <div className="icon-tile">
        <FileSearch />
      </div>
      <h3>{title}</h3>
      <p>{text}</p>
      {action}
    </div>
  );
}
export function Skeleton() {
  return (
    <div aria-label="Loading" className="skeleton-grid">
      {[1, 2, 3].map((i) => (
        <div key={i} className="skeleton" />
      ))}
    </div>
  );
}
export function MetricCard({ label, value, detail, icon: Icon }) {
  return (
    <div className="metric-card">
      {Icon && <Icon size={19} />}
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}
export function ImagePreview({ file, src, alt = 'Image preview' }) {
  const [url, setUrl] = useState('');
  useEffect(() => {
    if (!file) {
      setUrl('');
      return;
    }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  return file || src ? (
    <img src={file ? url || undefined : imageUrl(src)} alt={alt} className="image-preview" />
  ) : (
    <div className="preview-missing">
      <ImageIcon />
      <p>Preview not retained</p>
      <small>Enable image retention before saving to reopen visual evidence.</small>
    </div>
  );
}
export function UploadZone({
  files,
  onChange,
  multiple = false,
  label = 'Drop your image here',
  compact = false,
  disabled = false,
}) {
  const id = useId();
  const input = useRef(null);
  const [error, setError] = useState('');
  const [drag, setDrag] = useState(false);
  function select(list) {
    if (disabled) return;
    const next = Array.from(list || []);
    setError('');
    if (next.length > (multiple ? 10 : 1)) {
      setError(multiple ? 'Choose up to 10 images.' : 'Choose one image.');
      return;
    }
    if (next.reduce((bytes, file) => bytes + file.size, 0) > 64 * 1024 * 1024 - 65536) {
      setError('The combined upload must fit within the 64 MB request limit.');
      return;
    }
    if (
      next.some(
        (file) => !['image/jpeg', 'image/png'].includes(file.type) || file.size > 16 * 1024 * 1024,
      )
    ) {
      setError('Choose JPG or PNG images up to 16 MB each.');
      return;
    }
    onChange(next);
  }
  return (
    <div>
      <div
        className={`upload-zone ${drag ? 'dragging' : ''} ${compact ? 'compact' : ''}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDrag(false);
          select(event.dataTransfer.files);
        }}
      >
        <input
          id={id}
          ref={input}
          type="file"
          accept="image/jpeg,image/png"
          multiple={multiple}
          disabled={disabled}
          onChange={(event) => {
            select(event.target.files);
            event.target.value = '';
          }}
          className="sr-only"
        />
        <div className="upload-icon">
          <UploadCloud size={28} />
        </div>
        <h3>{label}</h3>
        <p>or choose from your files</p>
        <label htmlFor={id} className="button button-secondary cursor-pointer">
          {files.length ? 'Choose different images' : 'Browse images'}
          <ArrowUpRight size={16} />
        </label>
        <small>
          JPG, JPEG, PNG · 16 MB per image{multiple ? ' · Up to 10 images / 64 MB total' : ''}
        </small>
      </div>
      <ErrorState message={error} />
      {!!files.length && (
        <div className="selected-files">
          {files.map((file, i) => (
            <div key={`${file.name}-${i}`}>
              <ImagePreview file={file} />
              <span title={file.name}>
                {file.name}
                <small>{(file.size / 1024).toFixed(0)} KB</small>
              </span>
              <button
                disabled={disabled}
                aria-label={`Remove ${file.name}`}
                onClick={() => onChange(files.filter((_, index) => i !== index))}
              >
                <X size={15} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
export function SaveOptions({ save, setSave, retain, setRetain, live = false }) {
  return (
    <div className="save-options">
      <label>
        <input type="checkbox" checked={save} onChange={(event) => setSave(event.target.checked)} />
        {live ? 'Save this captured batch to history' : 'Save metrics and report to my history'}
      </label>
      <label>
        <input
          type="checkbox"
          checked={retain}
          disabled={!save}
          onChange={(event) => setRetain(event.target.checked)}
        />
        Also retain image previews (7 days by default)
      </label>
      <small>Uploads are processed temporarily. Image previews are kept only if you opt in.</small>
    </div>
  );
}
export function ProgressIndicator({ busy, completed, total }) {
  return busy ? (
    <div className="progress-status" role="status">
      <LoaderCircle className="spin" size={20} />
      <div>
        <strong>
          {total ? `${completed} / ${total} images analyzed` : 'Computing authenticity indicators'}
        </strong>
        <p>ELA, frequency analysis and pixel evidence. Please keep this tab open.</p>
      </div>
    </div>
  ) : null;
}
export function ResultCard({ item }) {
  const c = item.classification;
  return (
    <section className={`result-card result-${c.ai_indicators.toLowerCase()}`}>
      <div className="result-top">
        <span className="eyebrow">IMAGE AUTHENTICITY RESULT</span>
        <span className="badge badge-neutral">
          <ScanLine size={13} />
          Heuristic assessment
        </span>
      </div>
      <h2>{c.label}</h2>
      <div className="indicator-row">
        <span className="badge">AI indicators: {c.ai_indicators.toLowerCase()}</span>
        <span
          className="badge badge-neutral"
          title="The current engine has no validated manipulation-specific classifier."
        >
          Manipulation: not established
        </span>
      </div>
      <p>
        {c.ai_indicators === 'HIGH'
          ? 'Both existing forensic rules were triggered. Review the evidence; genuine photos can also trigger these rules.'
          : c.ai_indicators === 'LOW'
            ? 'Neither existing rule was triggered. This is consistent with fewer flagged signals, but does not prove authenticity.'
            : 'One forensic rule was triggered. The evidence is insufficient to resolve AI versus authentic.'}
      </p>
      <small>Forensic heuristics · No trained model probability · Supporting evidence below</small>
    </section>
  );
}
export function ForensicViewer({ item }) {
  return (
    <div className="forensic-grid">
      {[
        [
          'original',
          'Original image',
          'Downscaled preview; metrics use the reported processing dimensions.',
        ],
        ['ela', 'ELA evidence', 'JPEG differences, amplified for inspection.'],
        ['fft', 'Frequency spectrum', 'Log-magnitude spatial frequency patterns.'],
      ].map(([key, title, detail]) => (
        <div className="forensic-card" key={key}>
          <div>
            <span className="eyebrow">{key === 'original' ? 'SOURCE' : 'FORENSIC VIEW'}</span>
            <h3>{title}</h3>
          </div>
          <ImagePreview src={item.visualizations?.[key]} alt={title} />
          <small>{detail}</small>
        </div>
      ))}
    </div>
  );
}
export function AnalysisCard({ item, compact = false }) {
  if (item.status === 'failed') return <ErrorState message={`${item.filename}: ${item.error}`} />;
  const m = item.metrics;
  const values = [
    ['Resolution', `${m.width} × ${m.height}`],
    ['Brightness', `${m.brightness.toFixed(1)} / 255`],
    ['Channels', m.channels],
    ['JPEG Q90 pixel loss', m.ela_raw_mean.toFixed(2)],
  ];
  return (
    <div className="analysis-card">
      <ResultCard item={item} />
      {!compact && (
        <>
          <div className="section-heading">
            <h2>Supporting forensic evidence</h2>
            <span>{item.filename}</span>
          </div>
          <ForensicViewer item={item} />
          <div className="metric-grid">
            {values.map(([label, value]) => (
              <MetricCard label={label} value={value} key={label} />
            ))}
          </div>
          <section className="panel explanation">
            <h3>Why did SecureLens reach this result?</h3>
            {item.classification.explanations.map((text) => (
              <p key={text}>
                <Check size={17} />
                {text}
              </p>
            ))}
            <small>
              Compression, brightness and metadata are context; they do not contribute extra
              classification points.
            </small>
          </section>
          <details className="panel advanced">
            <summary>View Advanced Forensic Analysis</summary>
            <p>
              Measurements describe {m.analysis_width} × {m.analysis_height} RGB pixels.{' '}
              {m.notes.join(' ')}
            </p>
            <div className="advanced-grid">
              {[
                'ela_mean',
                'ela_std',
                'ela_raw_mean',
                'ela_raw_std',
                'frequency_mean',
                'frequency_std',
                'high_frequency_ratio',
                'spectral_centroid',
                'entropy',
                'edge_density',
                'noise',
                'std',
              ].map((key) => (
                <div key={key}>
                  <span>
                    {key === 'noise' ? 'Laplacian variance (sharpness)' : key.replaceAll('_', ' ')}
                  </span>
                  <strong>{Number(m[key]).toFixed(4)}</strong>
                </div>
              ))}
            </div>
            <h4>JPEG compression experiment</h4>
            <p>New pixel loss after recompression, not original compression history.</p>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>JPEG quality</th>
                    <th>Mean absolute error</th>
                    <th>Deviation</th>
                  </tr>
                </thead>
                <tbody>
                  {m.compression.map((row) => (
                    <tr key={row.quality}>
                      <td>{row.quality}</td>
                      <td>{row.mean_abs_difference.toFixed(3)}</td>
                      <td>{row.std_abs_difference.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>
              Source: {m.format} / {m.mode} · {(m.file_size_bytes / 1024).toFixed(1)} KB · EXIF:{' '}
              {m.has_exif ? 'Present' : 'Absent'}
            </p>
            <p>
              Rules: ELA mean &lt; {m.assessment.thresholds.ela_mean_below}: +2; FFT mean &gt;{' '}
              {m.assessment.thresholds.frequency_mean_above}: +1. 0 Low / 1–2 Moderate / 3 High.
            </p>
          </details>
        </>
      )}
      <p className="disclaimer">{item.classification.limitations}</p>
    </div>
  );
}
export function ReportButtons({ record }) {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  async function download(format) {
    setBusy(true);
    try {
      await exportReport(record, format);
    } catch (e) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="report-buttons">
      {[
        'json',
        ...(record.analysis_type === 'batch' ? ['csv'] : []),
        ...(record.id ? ['pdf'] : []),
      ].map((format) => (
        <button
          className="button button-secondary"
          key={format}
          disabled={busy}
          onClick={() => download(format)}
        >
          <Download size={16} />
          {format.toUpperCase()} report
        </button>
      ))}
      <span>{record.saved ? 'Saved to your private history' : 'Session only · not saved'}</span>
    </div>
  );
}
