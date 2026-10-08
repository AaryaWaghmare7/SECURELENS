import { afterEach, expect, test, vi } from 'vitest';
import { api, exportReport, reportCsv, setCsrfToken, streamBatch } from './api';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
  setCsrfToken('');
});

test('signup uses the registration endpoint, credentials and CSRF header', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ user: {} }) });
  vi.stubGlobal('fetch', fetch);
  setCsrfToken('test-csrf');
  const data = { name: 'Test User', email: 'test@example.com', password: 'test-password' };
  await api.register(data);
  expect(fetch).toHaveBeenCalledWith(
    '/api/auth/register',
    expect.objectContaining({
      method: 'POST',
      credentials: 'include',
      body: JSON.stringify(data),
      headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': 'test-csrf' },
    }),
  );
});

test('network failures explain connectivity and possible CORS without retrying signup', async () => {
  const fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
  vi.stubGlobal('fetch', fetch);
  await expect(api.register({})).rejects.toThrow('Could not connect to SecureLens');
  expect(fetch).toHaveBeenCalledOnce();
});

test('a slow response times out with a wake-up hint and does not retry a POST', async () => {
  vi.useFakeTimers();
  const fetch = vi.fn().mockImplementation(
    (_, { signal }) =>
      new Promise((resolve, reject) => {
        signal.addEventListener('abort', () => reject(signal.reason), { once: true });
      }),
  );
  vi.stubGlobal('fetch', fetch);
  const failure = expect(api.register({})).rejects.toThrow('server may be waking up');
  await vi.advanceTimersByTimeAsync(90_000);
  await failure;
  expect(fetch).toHaveBeenCalledOnce();
  expect(vi.getTimerCount()).toBe(0);
});

test('user cancellation remains an AbortError rather than a connectivity failure', async () => {
  const controller = new AbortController();
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation(
      (_, { signal }) =>
        new Promise((resolve, reject) => {
          signal.addEventListener('abort', () => reject(signal.reason), { once: true });
        }),
    ),
  );
  const failure = expect(api.analyze(new FormData(), controller.signal)).rejects.toMatchObject({
    name: 'AbortError',
  });
  controller.abort();
  await failure;
});

test.each([400, 409])('HTTP %i preserves safe backend validation messages', async (status) => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: false,
      status,
      json: async () => ({ detail: 'An account with this email already exists.' }),
    }),
  );
  await expect(api.register({})).rejects.toThrow('An account with this email already exists.');
});

test('validation arrays do not expose request values', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ detail: [{ input: 'private-value', msg: 'Invalid input' }] }),
    }),
  );
  await expect(api.register({})).rejects.toThrow(
    'Please check the information or image you provided.',
  );
});

test.each([502, 503, 504])(
  'gateway HTTP %i explains temporary server unavailability',
  async (status) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status }));
    await expect(api.register({})).rejects.toThrow('temporarily unavailable or waking up');
  },
);

test('server errors never display internal exception details', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({ detail: 'private database traceback' }),
    }),
  );
  await expect(api.login({})).rejects.toThrow('SecureLens could not complete this request.');
});

test('response timers clear after headers without breaking cancellation of a batch stream', async () => {
  vi.useFakeTimers();
  const controller = new AbortController();
  let requestSignal;
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(stream) {
      stream.enqueue(encoder.encode('data: {"record":{"id":"batch-test"}}\n\n'));
      stream.close();
    },
  });
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation(async (_, { signal }) => {
      requestSignal = signal;
      return { ok: true, body };
    }),
  );
  await streamBatch([], { save: false, retain: false }, vi.fn(), controller.signal);
  expect(vi.getTimerCount()).toBe(0);
  controller.abort();
  expect(requestSignal.aborted).toBe(true);
});

test('saved report downloads use authenticated API requests and the right file type', async () => {
  vi.useFakeTimers();
  const objectUrl = vi.fn().mockReturnValue('blob:test-report');
  const revoke = vi.fn();
  vi.stubGlobal('URL', { createObjectURL: objectUrl, revokeObjectURL: revoke });
  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  const blob = new Blob(['{"title":"SecureLens Analysis Report"}'], { type: 'application/json' });
  const fetch = vi.fn().mockResolvedValue({ ok: true, blob: async () => blob });
  vi.stubGlobal('fetch', fetch);
  await exportReport({ id: 'test-record' }, 'json');
  expect(fetch).toHaveBeenCalledWith(
    '/api/reports/test-record?format=json',
    expect.objectContaining({ credentials: 'include' }),
  );
  expect(objectUrl).toHaveBeenCalledWith(blob);
  expect(click.mock.contexts[0].download).toBe('securelens-test-record.json');
  expect(document.querySelector('a[download]')).toBeNull();
  vi.runAllTimers();
  expect(revoke).toHaveBeenCalledWith('blob:test-report');
});

test('unsaved batch CSV exports locally without creating a saved analysis', async () => {
  vi.useFakeTimers();
  const objectUrl = vi.fn().mockReturnValue('blob:test-csv');
  vi.stubGlobal('URL', { createObjectURL: objectUrl, revokeObjectURL: vi.fn() });
  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  const fetch = vi.fn();
  vi.stubGlobal('fetch', fetch);
  await exportReport(
    {
      id: null,
      analysis_type: 'batch',
      result: { items: [{ filename: 'a.png', status: 'complete' }] },
    },
    'csv',
  );
  expect(fetch).not.toHaveBeenCalled();
  expect(objectUrl.mock.calls[0][0].type).toContain('text/csv');
  expect(click.mock.contexts[0].download).toBe('securelens-batch.csv');
  vi.runAllTimers();
});

test('an expired API session tells the frontend to return to sign-in', async () => {
  const expire = vi.fn();
  window.addEventListener('securelens:session-expired', expire);
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockResolvedValue({ ok: false, status: 401, json: async () => ({ detail: 'Sign in' }) }),
  );
  await expect(api.history()).rejects.toThrow('Sign in');
  expect(expire).toHaveBeenCalledOnce();
  window.removeEventListener('securelens:session-expired', expire);
});

test('session-only batch CSV preserves failures and escapes spreadsheet formulas', () => {
  const csv = reportCsv({ result: { items: [{ filename: '=formula.png', status: 'failed' }] } });
  expect(csv).toContain('"\'=formula.png","failed"');
  expect(csv).toContain('manipulation_indicators');
  expect(csv).toContain('decision_source');
});

test('batch progress handles events split across network chunks', async () => {
  const encoder = new TextEncoder();
  const payload =
    'data: {"completed":1,"total":2,"item":{"filename":"a.png"}}\n\n' +
    'data: {"completed":2,"total":2,"item":{"filename":"b.png"}}\n\n' +
    'data: {"record":{"id":"test-record"}}\n\n';
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(payload.slice(0, 35)));
      controller.enqueue(encoder.encode(payload.slice(35)));
      controller.close();
    },
  });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, body }));
  const progress = vi.fn();
  const record = await streamBatch(
    [new File(['a'], 'a.png')],
    { save: false, retain: false },
    progress,
  );
  expect(progress.mock.calls.map(([event]) => event.completed)).toEqual([1, 2]);
  expect(record.id).toBe('test-record');
});
