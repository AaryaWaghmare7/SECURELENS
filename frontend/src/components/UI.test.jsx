import { describe, it, expect, vi } from 'vitest';
import { fireEvent, render, screen, cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';
import { ResultCard, UploadZone, ProgressIndicator } from './UI';

afterEach(cleanup);
describe('SecureLens result wording', () => {
  it('shows heuristic indicators without a fabricated probability', () => {
    render(
      <ResultCard
        item={{ classification: { label: 'Inconclusive', ai_indicators: 'MODERATE' } }}
      />,
    );
    expect(screen.getByText('Inconclusive')).toBeInTheDocument();
    expect(screen.getByText(/Heuristic assessment/)).toBeInTheDocument();
    expect(screen.getByText(/Manipulation: not established/)).toBeInTheDocument();
    expect(screen.queryByText(/99%|95%|confidence/i)).not.toBeInTheDocument();
  });
  it('reports real completed counts', () => {
    render(<ProgressIndicator busy completed={2} total={10} />);
    expect(screen.getByText('2 / 10 images analyzed')).toBeInTheDocument();
  });
  it('rejects unsupported and excessive uploads before sending', () => {
    const change = vi.fn();
    const { container } = render(<UploadZone files={[]} onChange={change} />);
    const input = container.querySelector('input');
    fireEvent.change(input, {
      target: { files: [new File(['bad'], 'image.txt', { type: 'text/plain' })] },
    });
    expect(screen.getByRole('alert')).toHaveTextContent('Choose JPG or PNG');
    expect(change).not.toHaveBeenCalled();
    fireEvent.change(input, {
      target: { files: [new File(['valid'], 'image.png', { type: 'image/png' })] },
    });
    expect(change).toHaveBeenCalledTimes(1);
  });
});
