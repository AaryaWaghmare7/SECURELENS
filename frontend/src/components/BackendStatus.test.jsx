import { afterEach, expect, test, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import BackendStatus from './BackendStatus';

const state = vi.hoisted(() => ({ status: 'idle', listener: null }));
vi.mock('../services/readiness', () => ({
  getBackendStatus: () => state.status,
  subscribeBackendStatus: (listener) => {
    state.listener = listener;
    return () => {
      state.listener = null;
    };
  },
}));
afterEach(() => {
  cleanup();
  state.status = 'idle';
});

test('waking status is real loading, disappears on recovery, and allows retry on failure', () => {
  const retry = vi.fn();
  render(<BackendStatus retry={retry} />);
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
  act(() => {
    state.status = 'waking';
    state.listener();
  });
  expect(screen.getByRole('status')).toHaveAttribute('aria-busy', 'true');
  expect(screen.getByText(/continue automatically/)).toBeInTheDocument();
  expect(screen.queryByText(/\d+%/)).not.toBeInTheDocument();
  act(() => {
    state.status = 'ready';
    state.listener();
  });
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
  act(() => {
    state.status = 'unavailable';
    state.listener();
  });
  expect(screen.getByText(/could not reach/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
  expect(retry).toHaveBeenCalledOnce();
});
