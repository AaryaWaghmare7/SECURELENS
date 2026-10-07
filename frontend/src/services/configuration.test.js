import { expect, test } from 'vitest';
import { resolveApiBase } from './configuration';
import { createVercelConfig } from '../../deployment/config.mjs';

test('local and Vercel proxy requests stay same-origin', () => {
  expect(resolveApiBase({})).toBe('');
  expect(
    resolveApiBase({ VITE_API_BASE_URL: 'https://backend.example', VITE_API_PROXY: 'true' }),
  ).toBe('');
  expect(resolveApiBase({ VITE_API_BASE_URL: 'https://api.example/' })).toBe('https://api.example');
});

test('Vercel forwards API paths before the SPA fallback without caching private data', () => {
  const config = createVercelConfig({
    VITE_API_BASE_URL: 'https://backend.example',
    VITE_API_PROXY: 'true',
  });
  expect(config.rewrites[0]).toEqual({
    source: '/api/:path*',
    destination: 'https://backend.example/api/:path*',
  });
  const fallback = new RegExp(`^${config.rewrites[1].source}$`);
  expect(fallback.test('/history/saved-result')).toBe(true);
  expect(fallback.test('/api/history')).toBe(false);
  expect(fallback.test('/api')).toBe(false);
  expect(fallback.test('/assets/missing.js')).toBe(false);
  expect(config.headers[1].headers[0].value).toBe('private, no-store');
});

test('Vercel deployment fails clearly instead of silently targeting localhost', () => {
  expect(() => createVercelConfig({ VERCEL: '1' })).toThrow('Set VITE_API_BASE_URL');
  for (const origin of [
    'http://api.example',
    'https://api.example/api',
    'https://user:secret@api.example',
    'https://localhost',
  ]) {
    expect(() => createVercelConfig({ VITE_API_BASE_URL: origin })).toThrow('public HTTPS origin');
  }
});
