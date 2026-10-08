import { afterEach, beforeEach, expect, test, vi } from 'vitest';

let readiness;
let api;
let streamBatch;
const healthy = () => ({ ok: true, status: 200, json: async () => ({ status: 'ok' }) });
const failed = (status) => ({ ok: false, status, json: async () => ({}) });
beforeEach(async () => {
  vi.resetModules();
  vi.useFakeTimers();
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  readiness = await import('./readiness');
  ({ api, streamBatch } = await import('./api'));
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

test('concurrent readiness callers share one health check and a short healthy cache', async () => {
  const fetch = vi.fn().mockResolvedValue(healthy());
  vi.stubGlobal('fetch', fetch);
  await Promise.all([readiness.ensureBackendReady(), readiness.ensureBackendReady()]);
  await readiness.ensureBackendReady();
  expect(fetch).toHaveBeenCalledOnce();
  expect(fetch.mock.calls[0][0]).toBe('/api/health');
  expect(fetch.mock.calls[0][1]).toMatchObject({ credentials: 'include', cache: 'no-store' });
  expect(readiness.getBackendStatus()).toBe('ready');
  await vi.advanceTimersByTimeAsync(10_001);
  await readiness.ensureBackendReady();
  expect(fetch).toHaveBeenCalledTimes(2);
  expect(vi.getTimerCount()).toBe(0);
});

test('502/503 recovery shows waking state, backs off, then submits signup exactly once', async () => {
  let healthCalls = 0;
  const fetch = vi.fn().mockImplementation(async (url) => {
    if (url === '/api/health')
      return ++healthCalls < 3 ? failed(healthCalls === 1 ? 503 : 502) : healthy();
    return { ok: true, json: async () => ({ user: { id: 'test-user' } }) };
  });
  vi.stubGlobal('fetch', fetch);
  const signup = api.register({});
  await vi.advanceTimersByTimeAsync(0);
  expect(readiness.getBackendStatus()).toBe('waking');
  expect(fetch.mock.calls.every(([url]) => url === '/api/health')).toBe(true);
  await vi.advanceTimersByTimeAsync(1_999);
  expect(healthCalls).toBe(1);
  await vi.advanceTimersByTimeAsync(1);
  expect(healthCalls).toBe(2);
  await vi.advanceTimersByTimeAsync(3_000);
  expect((await signup).user.id).toBe('test-user');
  expect(fetch.mock.calls.filter(([url]) => url === '/api/auth/register')).toHaveLength(1);
  expect(readiness.getBackendStatus()).toBe('ready');
});

test.each(['register', 'login', 'analyze', 'compare', 'logout'])(
  '%s does not replay a mutation after a gateway failure',
  async (method) => {
    const fetch = vi
      .fn()
      .mockImplementation(async (url) => (url === '/api/health' ? healthy() : failed(504)));
    vi.stubGlobal('fetch', fetch);
    await expect(api[method](new FormData())).rejects.toMatchObject({
      code: 'GATEWAY_TIMEOUT',
      status: 504,
    });
    expect(fetch.mock.calls.filter(([url]) => url !== '/api/health')).toHaveLength(1);
  },
);

test('batch also waits for health and never replays a failed POST', async () => {
  const fetch = vi
    .fn()
    .mockImplementation(async (url) => (url === '/api/health' ? healthy() : failed(502)));
  vi.stubGlobal('fetch', fetch);
  await expect(streamBatch([], { save: true, retain: false }, vi.fn())).rejects.toMatchObject({
    status: 502,
  });
  expect(fetch.mock.calls.map(([url]) => url)).toEqual([
    '/api/health',
    '/api/analyze/batch?stream=true',
  ]);
});

test('unavailable health exhausts the bounded window without sending a POST', async () => {
  const fetch = vi.fn().mockResolvedValue(failed(503));
  vi.stubGlobal('fetch', fetch);
  const result = expect(api.register({})).rejects.toMatchObject({ code: 'READINESS_TIMEOUT' });
  await vi.advanceTimersByTimeAsync(90_000);
  await result;
  expect(fetch.mock.calls.every(([url]) => url === '/api/health')).toBe(true);
  expect(fetch.mock.calls.length).toBeLessThanOrEqual(12);
  expect(readiness.getBackendStatus()).toBe('unavailable');
  expect(vi.getTimerCount()).toBe(0);
});

test('health network errors can recover without exposing raw browser failures', async () => {
  const fetch = vi
    .fn()
    .mockRejectedValueOnce(new TypeError('Failed to fetch'))
    .mockResolvedValue(healthy());
  vi.stubGlobal('fetch', fetch);
  const result = readiness.ensureBackendReady();
  await vi.advanceTimersByTimeAsync(2_000);
  await result;
  expect(fetch).toHaveBeenCalledTimes(2);
});

test('slow health attempts time out and recover within the total window', async () => {
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockImplementationOnce(
        (_, { signal }) =>
          new Promise((resolve, reject) => {
            signal.addEventListener('abort', () => reject(signal.reason), { once: true });
          }),
      )
      .mockResolvedValue(healthy()),
  );
  const result = readiness.ensureBackendReady();
  await vi.advanceTimersByTimeAsync(1_500);
  expect(readiness.getBackendStatus()).toBe('waking');
  await vi.advanceTimersByTimeAsync(12_500);
  await result;
  expect(readiness.getBackendStatus()).toBe('ready');
  expect(vi.getTimerCount()).toBe(0);
});

test.each([400, 401, 403, 404, 409, 422, 429, 500])(
  'health HTTP %i fails immediately instead of being mislabeled as a cold start',
  async (status) => {
    const fetch = vi.fn().mockResolvedValue(failed(status));
    vi.stubGlobal('fetch', fetch);
    await expect(readiness.ensureBackendReady()).rejects.toMatchObject({
      status,
      retryable: false,
    });
    expect(fetch).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
  },
);

test('one cancelled caller does not cancel a shared health check or submit its POST', async () => {
  const controller = new AbortController();
  vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(failed(503)).mockResolvedValue(healthy()));
  const cancelled = expect(api.analyze(new FormData(), controller.signal)).rejects.toMatchObject({
    name: 'AbortError',
  });
  const other = readiness.ensureBackendReady();
  controller.abort();
  await cancelled;
  await vi.advanceTimersByTimeAsync(2_000);
  await other;
  expect(fetch.mock.calls.every(([url]) => url === '/api/health')).toBe(true);
});

test('safe GET recovery rechecks health and retries the read only once', async () => {
  let calls = 0;
  const fetch = vi.fn().mockImplementation(async (url) => {
    if (url === '/api/health') return healthy();
    return ++calls === 1 ? failed(502) : { ok: true, json: async () => ({ total: 1 }) };
  });
  vi.stubGlobal('fetch', fetch);
  expect(await api.dashboard()).toEqual({ total: 1 });
  expect(fetch.mock.calls.map(([url]) => url)).toEqual([
    '/api/health',
    '/api/dashboard',
    '/api/health',
    '/api/dashboard',
  ]);
});

test('invalid health JSON never permits a mutation', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'error' }) });
  vi.stubGlobal('fetch', fetch);
  await expect(api.register({})).rejects.toMatchObject({ code: 'HEALTH_RESPONSE' });
  expect(fetch).toHaveBeenCalledOnce();
});

test('malformed health content does not leak raw gateway markup', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => {
        throw new SyntaxError('private gateway markup');
      },
    }),
  );
  await expect(api.register({})).rejects.toMatchObject({
    code: 'HEALTH_RESPONSE',
    message: expect.not.stringContaining('private'),
  });
});

test('initial unsigned session probes do not expire a concurrently accepted signup', async () => {
  const expire = vi.fn();
  window.addEventListener('securelens:session-expired', expire);
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation(async (url) => (url === '/api/health' ? healthy() : failed(401))),
  );
  await expect(api.me()).rejects.toMatchObject({ status: 401 });
  expect(expire).not.toHaveBeenCalled();
  window.removeEventListener('securelens:session-expired', expire);
});
