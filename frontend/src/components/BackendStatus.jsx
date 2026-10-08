import { useSyncExternalStore } from 'react';
import { LoaderCircle, AlertCircle } from 'lucide-react';
import { getBackendStatus, subscribeBackendStatus } from '../services/readiness';

export default function BackendStatus({ retry }) {
  const status = useSyncExternalStore(subscribeBackendStatus, getBackendStatus);
  if (!['waking', 'unavailable'].includes(status)) return null;
  const waking = status === 'waking';
  return (
    <aside
      className="backend-status panel progress-status"
      role="status"
      aria-live="polite"
      aria-busy={waking}
    >
      {waking ? <LoaderCircle className="spin" size={20} /> : <AlertCircle size={20} />}
      <div>
        <strong>{waking ? 'Starting the analysis server' : 'Analysis server unavailable'}</strong>
        <p>
          {waking
            ? 'SecureLens is starting the analysis server. This can take up to a minute on the free hosting tier.'
            : 'SecureLens could not reach the analysis server. Please try again.'}
        </p>
        {!waking && (
          <button className="button button-secondary" onClick={retry}>
            Try again
          </button>
        )}
      </div>
    </aside>
  );
}
