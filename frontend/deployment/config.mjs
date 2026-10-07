export function createVercelConfig(env) {
  const proxy = env.VITE_API_PROXY === 'true';
  const value = env.VITE_API_BASE_URL || '';
  let origin;
  if (value) {
    const url = new URL(value);
    if (
      url.protocol !== 'https:' ||
      url.username ||
      url.password ||
      url.pathname !== '/' ||
      url.search ||
      url.hash ||
      ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)
    )
      throw new Error(
        'VITE_API_BASE_URL must be a public HTTPS origin with no path or credentials.',
      );
    origin = url.origin;
  }
  if ((proxy || env.VERCEL === '1') && !origin) {
    throw new Error('Set VITE_API_BASE_URL to your Render HTTPS origin before deploying.');
  }
  const headers = [
    { key: 'X-Content-Type-Options', value: 'nosniff' },
    { key: 'X-Frame-Options', value: 'DENY' },
    { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
    { key: 'Permissions-Policy', value: 'camera=(self), microphone=(), geolocation=()' },
  ];
  return {
    framework: 'vite',
    buildCommand: 'npm run build',
    outputDirectory: 'dist',
    rewrites: [
      ...(proxy ? [{ source: '/api/:path*', destination: `${origin}/api/:path*` }] : []),
      { source: '/((?!api(?:/|$)|assets(?:/|$)).*)', destination: '/index.html' },
    ],
    headers: [
      { source: '/(.*)', headers },
      { source: '/api/:path*', headers: [{ key: 'Cache-Control', value: 'private, no-store' }] },
    ],
  };
}
