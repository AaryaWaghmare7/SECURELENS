import { afterEach, expect, test, vi } from 'vitest';
import { api, exportReport, reportCsv, streamBatch } from './api';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
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
