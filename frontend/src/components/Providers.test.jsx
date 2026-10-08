import { afterEach, expect, test, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Providers, useAuth } from './Providers';
import { api } from '../services/api';

vi.mock('../services/api', () => ({
  api: { me: vi.fn(), settings: vi.fn().mockResolvedValue({ retain_images: false }) },
  setCsrfToken: vi.fn(),
}));
vi.mock('./BackendStatus', () => ({ default: () => null }));
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function Probe() {
  const auth = useAuth();
  return (
    <>
      <div>{auth.user?.name || (auth.loading ? 'Loading' : 'Signed out')}</div>
      {auth.sessionError && <div role="alert">{auth.sessionError}</div>}
      <button
        onClick={() => auth.accept({ user: { id: 'new', name: 'New signup' }, csrf_token: 'test' })}
      >
        Accept signup
      </button>
      <button onClick={auth.retrySession}>Retry session</button>
    </>
  );
}
test('a late anonymous session response cannot overwrite accepted signup', async () => {
  let reject;
  api.me.mockImplementationOnce(
    () =>
      new Promise((_, fail) => {
        reject = fail;
      }),
  );
  render(
    <Providers>
      <Probe />
    </Providers>,
  );
  fireEvent.click(screen.getByText('Accept signup'));
  await act(async () => reject({ status: 401 }));
  expect(screen.getByText('New signup')).toBeInTheDocument();
});
test('temporary session unavailability is shown and can recover without a forced logout', async () => {
  api.me
    .mockRejectedValueOnce({ status: 503, message: 'Server unavailable' })
    .mockResolvedValueOnce({ user: { id: 'existing', name: 'Existing user' }, csrf_token: 'test' });
  render(
    <Providers>
      <Probe />
    </Providers>,
  );
  expect(await screen.findByRole('alert')).toHaveTextContent('Server unavailable');
  fireEvent.click(screen.getByText('Retry session'));
  await waitFor(() => expect(screen.getByText('Existing user')).toBeInTheDocument());
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});
