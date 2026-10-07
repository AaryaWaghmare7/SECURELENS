import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { ArrowRight, LockKeyhole, Eye, EyeOff, ScanLine } from 'lucide-react';
import { api } from '../services/api';
import { useAuth } from '../components/Providers';
import { ErrorState } from '../components/UI';

export default function AuthForm({ signup = false }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const { accept } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      const data = await (signup
        ? api.register({ name, email, password })
        : api.login({ email, password }));
      accept(data);
      navigate(location.state?.from?.pathname || '/dashboard', { replace: true });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="auth-layout">
      <div className="auth-art">
        <div className="icon-tile">
          <ScanLine size={34} />
        </div>
        <span className="eyebrow">YOUR PRIVATE IMAGE WORKSPACE</span>
        <h1>
          A clearer view
          <br />
          starts <em>here.</em>
        </h1>
        <p>
          Analyze, compare, and return to your evidence.
          <br />
          All in a workspace that's yours.
        </p>
        <div className="auth-privacy">
          <LockKeyhole size={18} />
          Passwords are securely hashed. Image storage is optional.
        </div>
      </div>
      <section className="panel auth-panel">
        <span className="eyebrow">WELCOME TO SECURELENS</span>
        <h2>{signup ? 'Create your account' : 'Welcome back.'}</h2>
        <p>
          {signup
            ? 'A little detail, and you are ready to explore.'
            : 'Sign in to continue your image research.'}
        </p>
        <form onSubmit={submit}>
          {signup && (
            <label>
              Display name
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                maxLength={100}
                autoComplete="name"
                placeholder="Your name"
              />
            </label>
          )}
          <label>
            Email address
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
              placeholder="you@example.com"
            />
          </label>
          <label>
            Password
            <div className="password-field">
              <input
                type={show ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={signup ? 10 : 1}
                maxLength={128}
                autoComplete={signup ? 'new-password' : 'current-password'}
                placeholder={signup ? 'At least 10 characters' : 'Enter your password'}
              />
              <button
                type="button"
                aria-label={show ? 'Hide password' : 'Show password'}
                onClick={() => setShow(!show)}
              >
                {show ? <EyeOff size={17} /> : <Eye size={17} />}
              </button>
            </div>
          </label>
          {!signup && (
            <Link className="auth-forgot" to="/forgot-password">
              Forgot password?
            </Link>
          )}
          <ErrorState message={error} />
          <button disabled={busy} className="button button-primary">
            {busy ? 'Please wait…' : signup ? 'Create account' : 'Log in'}
            <ArrowRight size={17} />
          </button>
        </form>
        <p className="auth-switch">
          {signup ? 'Already have an account?' : 'New to SecureLens?'}{' '}
          <Link to={signup ? '/login' : '/signup'}>{signup ? 'Log in' : 'Create account'}</Link>
        </p>
        <small>
          Read our <Link to="/privacy">privacy & storage information</Link>.
        </small>
      </section>
    </div>
  );
}
export function ForgotPassword() {
  return (
    <section className="panel narrow-page">
      <LockKeyhole />
      <h1>Password recovery</h1>
      <p>
        Email-based recovery has not been connected yet. SecureLens cannot currently send reset
        links. We will add verified, expiring reset tokens and email delivery before enabling this
        feature.
      </p>
      <Link className="button button-secondary" to="/login">
        Back to login
      </Link>
    </section>
  );
}
