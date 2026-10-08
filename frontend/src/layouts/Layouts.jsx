import { useState } from 'react';
import { Link, NavLink, Navigate, Outlet, useLocation, useNavigate } from 'react-router-dom';
import {
  ScanLine,
  LayoutDashboard,
  Image,
  Camera,
  Layers,
  GitCompareArrows,
  History,
  FileText,
  Settings,
  LogOut,
  Menu,
  ArrowUpRight,
  X,
} from 'lucide-react';
import { useAuth, useToast } from '../components/Providers';
import { ErrorState, Skeleton } from '../components/UI';

const routes = [
  ['Overview', '/dashboard', LayoutDashboard],
  ['Analyze', '/analyze', Image],
  ['Live Analysis', '/live-analysis', Camera],
  ['Batch Analysis', '/batch-analysis', Layers],
  ['Compare', '/compare', GitCompareArrows],
  ['History', '/history', History],
  ['Reports', '/reports', FileText],
  ['Settings', '/settings', Settings],
];
export function Brand() {
  return (
    <Link to="/" className="brand">
      <span className="brand-icon">
        <ScanLine size={23} />
      </span>
      <span>
        SecureLens<small>IMAGE AUTHENTICITY STUDIO</small>
      </span>
    </Link>
  );
}
export function Navbar() {
  const { user } = useAuth();
  const [open, setOpen] = useState(false);
  return (
    <header className="navbar">
      <Brand />
      <button className="menu-toggle" aria-label="Toggle navigation" onClick={() => setOpen(!open)}>
        {open ? <X /> : <Menu />}
      </button>
      <nav
        className={open ? 'open' : ''}
        aria-label="Main navigation"
        onClick={() => setOpen(false)}
      >
        {[
          ['Home', '/'],
          ...routes.slice(1, 5).map(([title, path]) => [title, path]),
          ['About', '/about'],
        ].map(([title, path]) => (
          <NavLink key={path} to={path} end>
            {title}
          </NavLink>
        ))}
      </nav>
      <div className="nav-auth">
        {user ? (
          <Link to="/dashboard" className="button button-primary">
            My workspace
            <ArrowUpRight size={15} />
          </Link>
        ) : (
          <>
            <Link to="/login" className="text-button">
              Log in
            </Link>
            <Link to="/signup" className="button button-primary">
              Create account
            </Link>
          </>
        )}
      </div>
    </header>
  );
}
export function Footer() {
  return (
    <footer className="footer">
      <Brand />
      <p>Evidence before certainty.</p>
      <div>
        <Link to="/about">About</Link>
        <Link to="/privacy">Privacy</Link>
        <Link to="/cookie-preferences">Cookie preferences</Link>
      </div>
      <small>© {new Date().getFullYear()} SecureLens · Built for thoughtful image research.</small>
    </footer>
  );
}
export function PublicLayout() {
  return (
    <>
      <Navbar />
      <main>
        <Outlet />
      </main>
      <Footer />
    </>
  );
}
export function WorkspaceLayout() {
  const { user, loading, logout, sessionError, retrySession } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const toast = useToast();
  const [open, setOpen] = useState(false);
  if (loading)
    return (
      <div className="workspace-loading">
        <Skeleton />
      </div>
    );
  if (!user && sessionError)
    return (
      <div className="workspace-loading">
        <ErrorState message={sessionError} />
        <button className="button button-secondary" onClick={retrySession}>
          Retry session check
        </button>
      </div>
    );
  if (!user) return <Navigate to="/login" state={{ from: location }} replace />;
  return (
    <div className="workspace">
      <aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
        <Brand />
        <div className="sidebar-label">YOUR WORKSPACE</div>
        <nav aria-label="Workspace navigation">
          {routes.map(([title, path, Icon]) => (
            <NavLink key={path} to={path} end onClick={() => setOpen(false)}>
              <Icon size={19} />
              {title}
              {path === '/live-analysis' && <span className="nav-dot" />}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-note">
          <ScanLine size={22} />
          <strong>Powered by evidence.</strong>
          <p>
            Forensic heuristics today.
            <br />
            Validated ML is the next step.
          </p>
        </div>
        <div className="profile">
          <span className="avatar">{user.name.charAt(0).toUpperCase()}</span>
          <div>
            <strong>{user.name}</strong>
            <small>Personal workspace</small>
          </div>
          <button
            aria-label="Log out"
            onClick={async () => {
              try {
                await logout();
                navigate('/');
              } catch (e) {
                toast(e.message, true);
              }
            }}
          >
            <LogOut size={18} />
          </button>
        </div>
      </aside>
      {open && (
        <button
          className="sidebar-scrim"
          aria-label="Close navigation"
          onClick={() => setOpen(false)}
        />
      )}
      <div className="workspace-main">
        <header className="workspace-bar">
          <button
            className="menu-toggle"
            aria-label="Open workspace navigation"
            onClick={() => setOpen(!open)}
          >
            <Menu />
          </button>
          <span>
            Workspace <span className="divider">/</span>{' '}
            {routes.find(([, path]) => path === location.pathname)?.[0] || 'Analysis result'}
          </span>
          <span className="engine-status">
            <i />
            Heuristic analysis
          </span>
        </header>
        <main className="workspace-content" key={location.pathname}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
