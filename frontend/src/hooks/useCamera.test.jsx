import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { useCamera } from './useCamera';
import { api } from '../services/api';

vi.mock('../services/api', async (importOriginal) => ({
  ...(await importOriginal()),
  api: { analyze: vi.fn() },
}));
let stopTrack;
let getUserMedia;
beforeEach(() => {
  stopTrack = vi.fn();
  getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [{ stop: stopTrack }] });
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia } });
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage: vi.fn() });
  vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((callback) =>
    callback(new Blob(['frame'], { type: 'image/png' })),
  );
  api.analyze.mockResolvedValue({
    result: { items: [{ classification: { label: 'Inconclusive' } }] },
  });
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

function setup() {
  const hook = renderHook(() => useCamera());
  hook.result.current.video.current = {
    videoWidth: 640,
    videoHeight: 480,
    play: vi.fn().mockResolvedValue(),
    srcObject: null,
  };
  return hook;
}

test('live frames are session-only and stopping closes camera tracks', async () => {
  const hook = setup();
  await act(() => hook.result.current.start());
  await waitFor(() => expect(hook.result.current.count).toBe(1));
  const form = api.analyze.mock.calls[0][0];
  expect(form.get('save')).toBe('false');
  expect(form.get('capture_source')).toBe('webcam');
  expect(form.get('retain_images')).toBe('false');
  expect(form.get('image').type).toBe('image/png');
  act(() => hook.result.current.stop());
  expect(stopTrack).toHaveBeenCalledOnce();
  expect(hook.result.current.running).toBe(false);
  hook.unmount();
});

test('permission errors show an actionable message and do not analyze', async () => {
  getUserMedia.mockRejectedValue(new Error('Denied'));
  const hook = setup();
  await act(() => hook.result.current.start());
  expect(hook.result.current.error).toMatch(/denied or is unavailable/);
  expect(api.analyze).not.toHaveBeenCalled();
  expect(hook.result.current.running).toBe(false);
  hook.unmount();
});

test('a camera granted after stopping is immediately closed', async () => {
  let resolve;
  getUserMedia.mockImplementation(
    () =>
      new Promise((done) => {
        resolve = done;
      }),
  );
  const hook = setup();
  let starting;
  act(() => {
    starting = hook.result.current.start();
  });
  act(() => hook.result.current.stop());
  await act(async () => {
    resolve({ getTracks: () => [{ stop: stopTrack }] });
    await starting;
  });
  expect(stopTrack).toHaveBeenCalledOnce();
  expect(api.analyze).not.toHaveBeenCalled();
  hook.unmount();
});
