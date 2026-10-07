import { createContext, useContext, useEffect, useState } from 'react';
import { X, CheckCircle2, AlertCircle } from 'lucide-react';
import { api, setCsrfToken } from '../services/api';

const AuthContext = createContext(null);
const ToastContext = createContext(null);
export const useAuth = () => useContext(AuthContext);
export const useToast = () => useContext(ToastContext);
export function Providers({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [toast, setToast] = useState(null);
  const [retainImages, setRetainImages] = useState(false);
  useEffect(() => {
    function expire() {
      setCsrfToken('');
      setUser(null);
    }
    window.addEventListener('securelens:session-expired', expire);
    return () => window.removeEventListener('securelens:session-expired', expire);
  }, []);
  useEffect(() => {
    let live = true;
    api
      .me()
      .then((data) => {
        if (live) {
          setCsrfToken(data.csrf_token);
          setUser(data.user);
        }
      })
      .catch(() => {})
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, []);
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
    setCsrfToken(data.csrf_token);
    setUser(data.user);
  }
  async function logout() {
    await api.logout();
    setCsrfToken('');
    setUser(null);
  }
  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        accept,
        logout,
        retainImages,
        setRetainImages,
        updateName: (name) => setUser((current) => ({ ...current, name })),
      }}
    >
      <ToastContext.Provider value={(message, error = false) => setToast({ message, error })}>
        {children}
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
