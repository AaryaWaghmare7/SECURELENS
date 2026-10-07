import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { Providers } from './components/Providers';
import CookieBanner from './components/CookieBanner';
import { PublicLayout, WorkspaceLayout } from './layouts/Layouts';
import Landing from './pages/Landing';
import AuthForm, { ForgotPassword } from './pages/Auth';
import Analyze from './pages/Analyze';
import Batch from './pages/Batch';
import Compare from './pages/Compare';
import Live from './pages/Live';
import History, { SavedResult } from './pages/History';
import Dashboard from './pages/Dashboard';
import Settings from './pages/Settings';
import { About, Privacy, CookiePreferences } from './pages/Information';

export default function App() {
  return (
    <BrowserRouter>
      <Providers>
        <Routes>
          <Route element={<PublicLayout />}>
            <Route index element={<Landing />} />
            <Route path="login" element={<AuthForm />} />
            <Route path="signup" element={<AuthForm signup />} />
            <Route path="forgot-password" element={<ForgotPassword />} />
            <Route path="about" element={<About />} />
            <Route path="privacy" element={<Privacy />} />
            <Route path="cookie-preferences" element={<CookiePreferences />} />
          </Route>
          <Route element={<WorkspaceLayout />}>
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="analyze" element={<Analyze />} />
            <Route path="live-analysis" element={<Live />} />
            <Route path="batch-analysis" element={<Batch />} />
            <Route path="compare" element={<Compare />} />
            <Route path="history" element={<History />} />
            <Route path="history/:id" element={<SavedResult />} />
            <Route path="reports" element={<History reports />} />
            <Route path="settings" element={<Settings />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <CookieBanner />
      </Providers>
    </BrowserRouter>
  );
}
