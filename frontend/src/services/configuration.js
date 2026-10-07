export function resolveApiBase(env) {
  return env.VITE_API_PROXY === 'true' ? '' : (env.VITE_API_BASE_URL || '').replace(/\/$/, '');
}
