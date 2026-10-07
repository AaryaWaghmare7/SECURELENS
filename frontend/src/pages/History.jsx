import { Link, useParams } from 'react-router-dom';
import { useState } from 'react';
import { Eye, Download, Trash2, ArrowUpRight } from 'lucide-react';
import { api, exportReport } from '../services/api';
import { useResource } from '../hooks/useResource';
import { useToast } from '../components/Providers';
import {
  AnalysisCard,
  EmptyState,
  ErrorState,
  PageHeader,
  ReportButtons,
  Skeleton,
} from '../components/UI';
import { ComparePanel } from './Compare';
import { BatchTable } from './Batch';

export function HistoryTable({ rows, refresh, reports = false }) {
  const toast = useToast();
  const [busy, setBusy] = useState(null);
  async function remove(row) {
    if (!window.confirm('Delete this saved analysis, its report and any retained previews?'))
      return;
    setBusy(row.id);
    try {
      await api.remove(row.id);
      refresh();
      toast('Analysis and stored files deleted.');
    } catch (e) {
      toast(e.message, true);
    } finally {
      setBusy(null);
    }
  }
  return (
    <div className="panel table-scroll">
      <table className="history-table">
        <thead>
          <tr>
            <th>Image / collection</th>
            <th>Type</th>
            <th>Assessment</th>
            <th>Saved</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>
                <strong>{row.filename}</strong>
                <small>
                  {row.item_count} image{row.item_count !== 1 ? 's' : ''}
                </small>
              </td>
              <td>
                <span className="badge badge-neutral">{row.analysis_type}</span>
              </td>
              <td>{row.label}</td>
              <td>{new Date(row.created_at).toLocaleDateString()}</td>
              <td>
                <div className="table-actions">
                  <Link
                    title="View result"
                    aria-label={`View ${row.filename}`}
                    to={`/history/${row.id}`}
                  >
                    <Eye size={17} />
                  </Link>
                  <button
                    title="Download JSON report"
                    aria-label={`Download ${row.filename}`}
                    onClick={async () => {
                      try {
                        await exportReport({ id: row.id });
                      } catch (e) {
                        toast(e.message, true);
                      }
                    }}
                  >
                    <Download size={17} />
                  </button>
                  {!reports && (
                    <button
                      disabled={busy === row.id}
                      title="Delete analysis"
                      aria-label={`Delete ${row.filename}`}
                      onClick={() => remove(row)}
                    >
                      <Trash2 size={16} />
                    </button>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export default function History({ reports = false }) {
  const [offset, setOffset] = useState(0);
  const { data, error, loading, refresh } = useResource(() => api.history(offset), [offset]);
  return (
    <>
      <PageHeader
        title={reports ? 'Your research, ready to share.' : 'Return to your evidence.'}
        action={
          <Link className="button button-primary" to="/analyze">
            New analysis
            <ArrowUpRight size={17} />
          </Link>
        }
      >
        {reports
          ? 'Download saved JSON, CSV batch and PDF reports from each result.'
          : 'Saved metrics and reports are private to your account. Retained images have a limited lifetime.'}
      </PageHeader>
      <ErrorState message={error} />
      {loading ? (
        <Skeleton />
      ) : data?.length ? (
        <HistoryTable rows={data} refresh={refresh} reports={reports} />
      ) : (
        <EmptyState
          title="A fresh workspace."
          text="Save an analysis to start building your private history."
        />
      )}
      <div className="pagination">
        <button
          className="button button-secondary"
          disabled={offset === 0 || loading}
          onClick={() => setOffset(Math.max(0, offset - 30))}
        >
          Previous
        </button>
        <span>Page {offset / 30 + 1}</span>
        <button
          className="button button-secondary"
          disabled={!data || data.length < 30 || loading}
          onClick={() => setOffset(offset + 30)}
        >
          Next
        </button>
      </div>
    </>
  );
}
export function SavedResult() {
  const { id } = useParams();
  const [selected, setSelected] = useState(null);
  const { data, error, loading } = useResource(() => api.detail(id), [id]);
  return (
    <>
      <PageHeader title="Your saved analysis.">
        Revisit the assessment and its supporting measurements.
      </PageHeader>
      <ErrorState message={error} />
      {loading ? (
        <Skeleton />
      ) : (
        data && (
          <>
            {data.analysis_type === 'compare' ? (
              <ComparePanel record={data} />
            ) : data.analysis_type === 'batch' ? (
              <>
                <BatchTable items={data.result.items} onSelect={setSelected} />
                {selected !== null && data.result.items[selected] && (
                  <AnalysisCard item={data.result.items[selected]} />
                )}
              </>
            ) : (
              data.result.items.map((item, i) => <AnalysisCard key={i} item={item} />)
            )}
            <ReportButtons record={data} />
          </>
        )
      )}
    </>
  );
}
