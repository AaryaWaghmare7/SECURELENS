import { useState } from 'react';
import { Camera, Square, Plus, Save, Layers } from 'lucide-react';
import { useCamera } from '../hooks/useCamera';
import { api, analysisForm, streamBatch } from '../services/api';
import {
  AnalysisCard,
  ErrorState,
  ImagePreview,
  PageHeader,
  ReportButtons,
  SaveOptions,
} from '../components/UI';
import { BatchTable } from './Batch';
import { useToast } from '../components/Providers';

export default function Live() {
  const camera = useCamera();
  const toast = useToast();
  const [saving, setSaving] = useState(false);
  const [retain, setRetain] = useState(false);
  const [save, setSave] = useState(false);
  const [captures, setCaptures] = useState([]);
  const [batch, setBatch] = useState(null);
  const [items, setItems] = useState([]);
  const [selected, setSelected] = useState(null);
  async function saveFrame() {
    if (!camera.lastFrame.current) return;
    setSaving(true);
    try {
      await api.analyze(
        analysisForm([camera.lastFrame.current], { save: true, retain, source: 'webcam' }),
      );
      toast('Captured frame metrics saved.');
    } catch (e) {
      toast(e.message, true);
    } finally {
      setSaving(false);
    }
  }
  async function capture() {
    if (captures.length >= 10) {
      toast('A captured batch can contain up to 10 frames.', true);
      return;
    }
    try {
      const frame = await camera.capture();
      setCaptures((current) => [...current, frame]);
    } catch (e) {
      toast(e.message, true);
    }
  }
  async function batchRun() {
    setSaving(true);
    setItems([]);
    setBatch(null);
    try {
      const record = await streamBatch(captures, { save, retain }, (next) =>
        setItems((current) => [...current, next.item]),
      );
      setBatch(record);
    } catch (e) {
      toast(e.message, true);
    } finally {
      setSaving(false);
    }
  }
  return (
    <>
      <PageHeader eyebrow="LIVE ANALYSIS" title="A live view, with evidence.">
        Camera frames use the same forensic rules as uploads. Being captured live does not force an
        authentic result.
      </PageHeader>
      <div className="live-grid">
        <section className="panel camera-panel">
          <div className="camera-shell">
            <video ref={camera.video} autoPlay muted playsInline />
            {!camera.running && (
              <div className="camera-placeholder">
                <Camera size={42} />
                <h3>Your camera stays yours.</h3>
                <p>Start analysis to enable the live feed.</p>
              </div>
            )}
            <span className={`live-badge ${camera.running ? 'is-live' : ''}`}>
              <i />
              {camera.running ? 'LIVE' : 'CAMERA OFF'}
            </span>
          </div>
          <div className="button-row">
            <button
              className="button button-primary"
              disabled={camera.running}
              onClick={camera.start}
            >
              <Camera size={17} />
              Start analysis
            </button>
            <button
              className="button button-secondary"
              disabled={!camera.running}
              onClick={camera.stop}
            >
              <Square size={16} />
              Stop analysis
            </button>
            <button
              className="button button-secondary"
              disabled={!camera.running || saving}
              onClick={capture}
            >
              <Plus size={16} />
              Capture to batch
            </button>
          </div>
          <ErrorState message={camera.error} />
        </section>
        <aside className="panel guide-panel">
          <span className="eyebrow">PRIVATE UNTIL YOU SAVE</span>
          <h3>{camera.count} frames examined</h3>
          <p>
            One frame is analyzed at a time, with a pause between requests. Live captures are not
            written to storage automatically.
          </p>
          <label className="checkbox-label">
            <input type="checkbox" checked={retain} onChange={(e) => setRetain(e.target.checked)} />
            Retain previews when I save
          </label>
          <button
            className="button button-secondary"
            disabled={!camera.result || saving}
            onClick={saveFrame}
          >
            <Save size={16} />
            Save current frame
          </button>
          <small>
            Saving is your explicit choice. Camera access ends when you stop or leave this page.
          </small>
        </aside>
      </div>
      {camera.result && <AnalysisCard item={camera.result.result.items[0]} />}
      {!!captures.length && (
        <section className="panel captured-batch">
          <h2>Captured batch · {captures.length} frames</h2>
          <div className="capture-strip">
            {captures.map((file) => (
              <ImagePreview key={file.name} file={file} />
            ))}
          </div>
          <SaveOptions {...{ save, setSave, retain, setRetain }} live />
          <div className="button-row">
            <button className="button button-primary" disabled={saving} onClick={batchRun}>
              <Layers size={16} />
              {saving ? `${items.length} / ${captures.length} analyzed` : 'Analyze captured batch'}
            </button>
            <button
              className="button button-secondary"
              disabled={saving}
              onClick={() => {
                setCaptures([]);
                setBatch(null);
                setItems([]);
              }}
            >
              Clear frames
            </button>
          </div>
          {!!items.length && <BatchTable items={items} onSelect={setSelected} />}{' '}
          {batch && <ReportButtons record={batch} />}{' '}
          {selected !== null && items[selected] && <AnalysisCard item={items[selected]} />}
        </section>
      )}
    </>
  );
}
