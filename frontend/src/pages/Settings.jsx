import { useEffect, useState } from 'react';
import { Cookie, LockKeyhole, Save } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useResource } from '../hooks/useResource';
import { api } from '../services/api';
import { useAuth, useToast } from '../components/Providers';
import { ErrorState, PageHeader, Skeleton } from '../components/UI';

export default function Settings() {
  const { data, error, loading } = useResource(api.settings);
  const { updateName, setRetainImages } = useAuth();
  const toast = useToast();
  const [name, setName] = useState('');
  const [retain, setRetain] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (data) {
      setName(data.name);
      setRetain(data.retain_images);
    }
  }, [data]);
  async function save(event) {
    event.preventDefault();
    setBusy(true);
    try {
      const next = await api.updateSettings({ name, retain_images: retain });
      updateName(next.name);
      setRetainImages(next.retain_images);
      toast('Profile preferences updated.');
    } catch (e) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader title="A workspace on your terms.">
        Profile details, privacy and storage choices.
      </PageHeader>
      <ErrorState message={error} />
      {loading ? (
        <Skeleton />
      ) : (
        data && (
          <div className="settings-grid">
            <form className="panel settings-panel" onSubmit={save}>
              <h2>Profile settings</h2>
              <label>
                Display name
                <input
                  required
                  maxLength={100}
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </label>
              <label>
                Email address
                <input readOnly value={data.email} />
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={retain}
                  onChange={(e) => setRetain(e.target.checked)}
                />
                Prefer retaining image previews
              </label>
              <p>
                This preference fills future upload options. You can override it on each analysis.
                Preview retention defaults to 7 days; reports stay until you delete them.
              </p>
              <button className="button button-primary" disabled={busy}>
                <Save size={16} />
                Save preferences
              </button>
            </form>
            <div className="settings-stack">
              <section className="panel">
                <LockKeyhole />
                <h3>Account security</h3>
                <p>
                  Passwords use Argon2 hashing. Password changes and email recovery will be enabled
                  with verified reset flows later.
                </p>
              </section>
              <section className="panel">
                <Cookie />
                <h3>Privacy settings</h3>
                <button
                  className="button button-secondary"
                  onClick={() => window.dispatchEvent(new Event('securelens:cookies'))}
                >
                  Cookie preferences
                </button>
                <Link className="text-link" to="/history">
                  Manage or delete saved analyses
                </Link>
                <Link className="text-link" to="/privacy">
                  Read privacy information
                </Link>
              </section>
            </div>
          </div>
        )
      )}
    </>
  );
}
