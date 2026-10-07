import { useEffect, useState } from 'react';
import { Cookie, X, ShieldCheck } from 'lucide-react';
import { useAuth, useToast } from './Providers';
import { api } from '../services/api';

const KEY = 'securelens-cookie-preferences';
function savedPreference() {
  try {
    return JSON.parse(localStorage.getItem(KEY));
  } catch {
    return null;
  }
}
export default function CookieBanner() {
  const { user } = useAuth();
  const toast = useToast();
  const [preference, setPreference] = useState(savedPreference);
  const [modal, setModal] = useState(false);
  useEffect(() => {
    function open() {
      setModal(true);
    }
    window.addEventListener('securelens:cookies', open);
    return () => window.removeEventListener('securelens:cookies', open);
  }, []);
  useEffect(() => {
    if (!user) return;
    let live = true;
    api
      .consent()
      .then(async (value) => {
        if (!live) return;
        if (value.recorded) {
          localStorage.setItem(KEY, JSON.stringify(value));
          setPreference(value);
        } else if (preference) await api.saveConsent(preference);
      })
      .catch((e) => toast(e.message, true));
    return () => {
      live = false;
    };
  }, [user?.id]);
  async function choose() {
    const value = { essential: true, authentication: true, analytics: false };
    try {
      if (user) await api.saveConsent(value);
      localStorage.setItem(KEY, JSON.stringify(value));
      setPreference(value);
      setModal(false);
    } catch (e) {
      toast(e.message, true);
    }
  }
  return (
    <>
      {!preference && (
        <aside className="cookie-banner" aria-label="Cookie preferences">
          <div className="cookie-copy">
            <Cookie size={24} />
            <div>
              <strong>A little privacy clarity.</strong>
              <p>
                We use an essential sign-in cookie and local preferences. No analytics or
                advertising trackers are installed.
              </p>
            </div>
          </div>
          <div className="cookie-actions">
            <button onClick={choose} className="button button-primary">
              Accept
            </button>
            <button onClick={choose} className="button button-secondary">
              Reject Non-Essential
            </button>
            <button onClick={() => setModal(true)} className="text-button">
              Manage Preferences
            </button>
          </div>
        </aside>
      )}
      {modal && (
        <div className="modal-backdrop">
          <section
            className="panel modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="cookie-title"
          >
            <button
              className="modal-close"
              aria-label="Close preferences"
              onClick={() => setModal(false)}
            >
              <X />
            </button>
            <ShieldCheck />
            <h2 id="cookie-title">Cookie preferences</h2>
            <p>
              Your session needs essential authentication cookies. There are currently no optional
              trackers.
            </p>
            {[
              ['Essential cookies', 'Required for the sign-in session.'],
              [
                'Authentication cookies',
                'An HttpOnly session cookie; no token in browser local storage.',
              ],
              ['Analytics cookies', 'Not installed. No analytics data is collected.'],
            ].map(([name, detail], i) => (
              <div className="preference-row" key={name}>
                <div>
                  <strong>{name}</strong>
                  <p>{detail}</p>
                </div>
                <span className="badge badge-neutral">{i < 2 ? 'Required' : 'Not used'}</span>
              </div>
            ))}
            <button className="button button-primary" onClick={choose}>
              Save preferences
            </button>
          </section>
        </div>
      )}
    </>
  );
}
