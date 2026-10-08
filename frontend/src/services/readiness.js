import { ApiError, diagnose, fetchResponse } from './transport';

const WINDOW_MS = 90_000;
const ATTEMPT_MS = 12_000;
const READY_TTL_MS = 10_000;
const BACKOFF_MS = [2_000, 3_000, 5_000, 8_000, 10_000];
const listeners = new Set();
let status = 'idle';
let readyUntil = 0;
let flight = null;

export const getBackendStatus = () => status;
export function subscribeBackendStatus(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
function publish(next) {
  status = next;
  listeners.forEach((listener) => listener());
}
export function invalidateBackendReady() {
  readyUntil = 0;
}

async function poll() {
  const deadline = Date.now() + WINDOW_MS;
  const slow = setTimeout(() => publish('waking'), 1_500);
  publish('checking');
  let attempt = 0;
  try {
    while (Date.now() < deadline) {
      const controller = new AbortController();
      const timer = setTimeout(
        () => controller.abort(),
        Math.min(ATTEMPT_MS, deadline - Date.now()),
      );
      try {
        const response = await fetchResponse(
          '/health',
          {
            signal: controller.signal,
            cache: 'no-store',
          },
          ATTEMPT_MS,
        );
        let body;
        try {
          body = await response.json();
        } catch (error) {
          if (controller.signal.aborted) throw error;
          throw new ApiError(
            'SecureLens returned an invalid health response. Please contact support.',
            'HEALTH_RESPONSE',
          );
        }
        if (body.status !== 'ok')
          throw new ApiError(
            'SecureLens returned an invalid health response. Please contact support.',
            'HEALTH_RESPONSE',
          );
        readyUntil = Date.now() + READY_TTL_MS;
        publish('ready');
        return;
      } catch (error) {
        if (controller.signal.aborted)
          error = new ApiError('Health check timed out.', 'TIMEOUT', null, true);
        diagnose('/health', 'GET', error);
        if (!error.retryable) throw error;
      } finally {
        clearTimeout(timer);
      }
      publish('waking');
      const delay = Math.min(
        BACKOFF_MS[Math.min(attempt++, BACKOFF_MS.length - 1)],
        deadline - Date.now(),
      );
      if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay));
    }
    throw new ApiError(
      'SecureLens could not reach the analysis server. Please try again.',
      'READINESS_TIMEOUT',
    );
  } catch (error) {
    readyUntil = 0;
    publish('unavailable');
    throw error;
  } finally {
    clearTimeout(slow);
  }
}

// Each caller can cancel its wait without cancelling another caller's health check.
function waitForFlight(promise, signal) {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason);
    signal.addEventListener('abort', abort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort));
  });
}

export function ensureBackendReady({ signal, force = false } = {}) {
  if (signal?.aborted) return Promise.reject(signal.reason);
  if (!force && Date.now() < readyUntil) return Promise.resolve();
  if (!flight) {
    flight = poll().finally(() => {
      flight = null;
    });
  }
  return waitForFlight(flight, signal);
}
