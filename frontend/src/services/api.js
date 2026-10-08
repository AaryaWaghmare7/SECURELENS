import { resolveApiBase } from './configuration';

export const API_BASE = resolveApiBase(import.meta.env);
const RESPONSE_TIMEOUT_MS = 90_000;
let csrfToken = '';
export function setCsrfToken(token) {
  csrfToken = token || '';
}

async function fetchApi(path, options = {}) {
  const headers = { ...options.headers };
  if (options.body && !(options.body instanceof FormData))
    headers['Content-Type'] = 'application/json';
  if (options.method && !['GET', 'HEAD'].includes(options.method))
    headers['X-CSRF-Token'] = csrfToken;
  const timeout = new AbortController();
  const signal = options.signal
    ? AbortSignal.any([options.signal, timeout.signal])
    : timeout.signal;
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    timeout.abort();
  }, RESPONSE_TIMEOUT_MS);
  let response;
  try {
    response = await fetch(`${API_BASE}/api${path}`, {
      credentials: 'include',
      ...options,
      headers,
      signal,
    });
  } catch (error) {
    if (options.signal?.aborted) throw error;
    if (timedOut)
      throw new Error(
        'SecureLens is taking too long to respond. The server may be waking up; wait a minute and try again.',
      );
    if (error instanceof TypeError)
      throw new Error(
        'Could not connect to SecureLens. Check your connection and try again. If this persists, the API may be unavailable or the browser may be blocking the request (CORS).',
      );
    throw error;
  } finally {
    // Bound the wait for headers, not the lifetime of a batch response stream.
    clearTimeout(timer);
  }
  if (!response.ok) {
    if (response.status === 401 && !['/auth/login', '/auth/register'].includes(path)) {
      window.dispatchEvent(new Event('securelens:session-expired'));
    }
    if ([502, 503, 504].includes(response.status))
      throw new Error(
        'The SecureLens server is temporarily unavailable or waking up. Wait a minute and try again.',
      );
    if (response.status >= 500)
      throw new Error('SecureLens could not complete this request. Please try again later.');
    let message = 'The request could not finish. Please try again.';
    try {
      const error = await response.json();
      message =
        typeof error.detail === 'string'
          ? error.detail
          : 'Please check the information or image you provided.';
    } catch {
      /* Non-JSON proxy failure. */
    }
    throw new Error(message);
  }
  return response;
}

export async function request(path, options) {
  const response = await fetchApi(path, options);
  return response.status === 204 ? null : response.json();
}
export const api = {
  me: () => request('/auth/me'),
  login: (data) => request('/auth/login', { method: 'POST', body: JSON.stringify(data) }),
  register: (data) => request('/auth/register', { method: 'POST', body: JSON.stringify(data) }),
  logout: () => request('/auth/logout', { method: 'POST' }),
  analyze: (data, signal) => request('/analyze', { method: 'POST', body: data, signal }),
  compare: (data) => request('/analyze/compare', { method: 'POST', body: data }),
  history: (offset = 0) => request(`/history?offset=${offset}`),
  detail: (id) => request(`/history/${id}`),
  remove: (id) => request(`/history/${id}`, { method: 'DELETE' }),
  dashboard: () => request('/dashboard'),
  settings: () => request('/settings'),
  updateSettings: (data) => request('/settings', { method: 'PATCH', body: JSON.stringify(data) }),
  consent: () => request('/cookie-consent'),
  saveConsent: (data) => request('/cookie-consent', { method: 'PUT', body: JSON.stringify(data) }),
};

export function analysisForm(files, { save = true, retain = false, source = 'upload' } = {}) {
  const body = new FormData();
  if (files.length === 1) body.append('image', files[0]);
  body.append('save', String(save));
  body.append('retain_images', String(retain));
  body.append('capture_source', source);
  return body;
}
export async function streamBatch(files, options, onProgress, signal) {
  const body = new FormData();
  files.forEach((file) => body.append('images', file));
  body.append('save', String(options.save));
  body.append('retain_images', String(options.retain));
  const response = await fetchApi('/analyze/batch?stream=true', { method: 'POST', body, signal });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let record;
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      let end;
      while ((end = buffer.indexOf('\n\n')) >= 0) {
        const event = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        const data = event
          .split('\n')
          .filter((line) => line.startsWith('data: '))
          .map((line) => line.slice(6))
          .join('\n');
        if (data) {
          const message = JSON.parse(data);
          if (message.error) throw new Error(message.error);
          if (message.record) record = message.record;
          else onProgress(message);
        }
      }
      if (done) break;
    }
  } finally {
    reader.releaseLock();
  }
  if (!record) throw new Error('Batch analysis was interrupted. Please retry.');
  return record;
}
export function imageUrl(url) {
  return url?.startsWith('/api/') ? `${API_BASE}${url}` : url;
}
export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function reportCsv(record) {
  const rows = [
    [
      'filename',
      'status',
      'result',
      'ai_indicators',
      'manipulation_indicators',
      'ela_mean',
      'fft_mean',
      'brightness',
      'width',
      'height',
      'entropy',
      'edge_density',
      'jpeg_q90_mae',
      'decision_source',
    ],
  ];
  for (const item of record.result.items) {
    const m = item.metrics || {};
    const c = item.classification || {};
    rows.push([
      item.filename,
      item.status,
      c.label,
      c.ai_indicators,
      c.manipulation_indicators,
      m.ela_mean,
      m.frequency_mean,
      m.brightness,
      m.width,
      m.height,
      m.entropy,
      m.edge_density,
      m.ela_raw_mean,
      c.source,
    ]);
  }
  return (
    rows
      .map((row) =>
        row
          .map((value) => {
            let text = String(value ?? '');
            if (/^[=+@\-\t\r]/.test(text)) text = "'" + text;
            return '"' + text.replaceAll('"', '""') + '"';
          })
          .join(','),
      )
      .join('\r\n') + '\r\n'
  );
}
export async function exportReport(record, format = 'json') {
  if (record.id) {
    const response = await fetchApi(`/reports/${record.id}?format=${format}`);
    saveBlob(await response.blob(), `securelens-${record.id}.${format}`);
    return;
  }
  if (format === 'csv') {
    saveBlob(
      new Blob([reportCsv(record)], { type: 'text/csv;charset=utf-8' }),
      'securelens-batch.csv',
    );
    return;
  }
  if (format !== 'json') throw new Error('Save this analysis to download its PDF report.');
  const clean = {
    ...record,
    title: 'SecureLens Analysis Report',
    result: {
      ...record.result,
      items: record.result.items.map(({ visualizations, ...item }) => item),
    },
  };
  saveBlob(
    new Blob([JSON.stringify(clean, null, 2)], { type: 'application/json' }),
    'securelens-analysis.json',
  );
}
