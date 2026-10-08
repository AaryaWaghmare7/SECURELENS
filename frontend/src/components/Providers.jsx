import { createContext, useContext, useEffect, useRef, useState } from 'react';
import { X, CheckCircle2, AlertCircle } from 'lucide-react';
import { api, setCsrfToken } from '../services/api';
import BackendStatus from './BackendStatus';

const AuthContext = createContext(null);
const ToastContext = createContext(null);
export const useAuth = () => useContext(AuthContext);
export const useToast = () => useContext(ToastContext);
export function Providers({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [toast, setToast] = useState(null);
  const [retainImages, setRetainImages] = useState(false);
  const [sessionError, setSessionError] = useState('');
  const [sessionAttempt, setSessionAttempt] = useState(0);
  const generation = useRef(0);
  useEffect(() => {
    function expire() {
      generation.current++;
      setCsrfToken('');
      setUser(null);
      setLoading(false);
    }
    window.addEventListener('securelens:session-expired', expire);
    return () => window.removeEventListener('securelens:session-expired', expire);
  }, []);
  useEffect(() => {
    let live = true;
    const started = generation.current;
    setLoading(true);
    setSessionError('');
    api
      .me()
      .then((data) => {
        if (live && generation.current === started) {
          setCsrfToken(data.csrf_token);
          setUser(data.user);
        }
      })
      .catch((error) => {
        if (live && generation.current === started) {
          if (error.status === 401) {
            setCsrfToken('');
            setUser(null);
          } else setSessionError(error.message);
        }
      })
      .finally(() => {
        if (live && generation.current === started) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, [sessionAttempt]);
  useEffect(() => {
    if (!toast) return;
    const timeout = setTimeout(() => setToast(null), 5000);
    return () => clearTimeout(timeout);
  }, [toast]);
  useEffect(() => {
    let live = true;
    setRetainImages(false);
    if (user?.id)
      api
        .settings()
        .then((data) => {
          if (live) setRetainImages(data.retain_images);
        })
        .catch(() => {});
    return () => {
      live = false;
    };
  }, [user?.id]);
  function accept(data) {
    generation.current++;
    setCsrfToken(data.csrf_token);
    setUser(data.user);
    setLoading(false);
    setSessionError('');
  }
  async function logout() {
    await api.logout();
    generation.current++;
    setCsrfToken('');
    setUser(null);
  }
  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        sessionError,
        retrySession: () => setSessionAttempt((value) => value + 1),
        accept,
        logout,
        retainImages,
        setRetainImages,
        updateName: (name) => setUser((current) => ({ ...current, name })),
      }}
    >
      <ToastContext.Provider value={(message, error = false) => setToast({ message, error })}>
        {children}
        <BackendStatus retry={() => setSessionAttempt((value) => value + 1)} />
        {toast && (
          <div className={`toast ${toast.error ? 'toast-error' : ''}`} role="status">
            {toast.error ? <AlertCircle size={20} /> : <CheckCircle2 size={20} />}
            {toast.message}
            <button onClick={() => setToast(null)} aria-label="Dismiss notification">
              <X size={18} />
            </button>
          </div>
        )}
      </ToastContext.Provider>
    </AuthContext.Provider>
  );
}
