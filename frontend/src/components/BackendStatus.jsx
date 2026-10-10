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
            ? 'The free server may be waking after inactivity. Keep this page open; SecureLens will continue automatically when it is ready. This can take a few minutes.'
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
