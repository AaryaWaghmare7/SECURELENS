import { resolveApiBase } from './configuration';

export const API_BASE = resolveApiBase(import.meta.env);
export class ApiError extends Error {
  constructor(message, code, status = null, retryable = false) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.retryable = retryable;
  }
}

const HTTP_ERRORS = {
  401: ['Sign in to access your SecureLens workspace.', 'AUTHENTICATION'],
  403: ['The security check failed. Sign in again or contact support.', 'FORBIDDEN'],
  409: ['An account with that email already exists.', 'CONFLICT'],
  422: ['Please check the information or image you provided.', 'VALIDATION'],
  429: ['Too many requests. Please wait a minute before trying again.', 'RATE_LIMIT'],
  500: ['SecureLens could not complete this request. Please try again later.', 'BACKEND_ERROR'],
  502: ['The gateway could not reach the SecureLens server. Please try again.', 'BAD_GATEWAY'],
  503: ['The SecureLens server is currently unavailable. Please try again.', 'UNAVAILABLE'],
  504: ['The gateway timed out waiting for SecureLens. Please try again.', 'GATEWAY_TIMEOUT'],
};

export async function fetchResponse(path, options = {}, timeoutMs = 90_000) {
  const timeout = new AbortController();
  const signal = options.signal
    ? AbortSignal.any([options.signal, timeout.signal])
    : timeout.signal;
  const timer = setTimeout(() => timeout.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE}/api${path}`, {
      credentials: 'include',
      ...options,
      signal,
    });
    if (response.ok) return response;
    let [message, code] = HTTP_ERRORS[response.status] || [
      'The request could not finish. Please try again.',
      'HTTP_ERROR',
    ];
    if (response.status < 500) {
      try {
        const error = await response.json();
        if (typeof error.detail === 'string') message = error.detail;
      } catch {
        // Gateway HTML and validation arrays must not be shown verbatim.
      }
    }
    throw new ApiError(message, code, response.status, [502, 503, 504].includes(response.status));
  } catch (error) {
    if (options.signal?.aborted) throw error;
    if (timeout.signal.aborted)
      throw new ApiError(
        'SecureLens took too long to respond. The request was not retried; check your history or account before submitting again.',
        'TIMEOUT',
        null,
        true,
      );
    if (error instanceof TypeError)
      throw new ApiError(
        'Could not connect to SecureLens. Check your connection. The API may be unavailable or the browser may be blocking the request (CORS).',
        'NETWORK',
        null,
        true,
      );
    throw error;
  } finally {
    // Keep caller cancellation connected while a batch response is being read.
    clearTimeout(timer);
  }
}

export function diagnose(path, method, error) {
  if (error.name === 'AbortError' || (path === '/auth/me' && error.status === 401)) return;
  console.warn('SecureLens API request failed', {
    method,
    endpoint: path.split('?')[0],
    code: error.code || 'UNKNOWN',
    status: error.status || null,
  });
}
