export function resolveApiBase(env) {
  return env.VITE_API_PROXY === 'true' ? '' : (env.VITE_API_BASE_URL || '').replace(/\/$/, '');
}

export function resolveHealthWakeUrl(env) {
  if (env.VITE_API_PROXY !== 'true' || !env.VITE_API_BASE_URL) return null;
  try {
    const url = new URL(env.VITE_API_BASE_URL);
    if (
      url.protocol !== 'https:' ||
      url.username ||
      url.password ||
      url.pathname !== '/' ||
      url.search ||
      url.hash ||
      ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)
    )
      return null;
    return `${url.origin}/api/health`;
  } catch {
    return null;
  }
}
