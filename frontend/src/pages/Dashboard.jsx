import { Link } from 'react-router-dom';
import { ArrowUpRight, Image, Layers, GitCompareArrows, Activity, Camera } from 'lucide-react';
import { useAuth } from '../components/Providers';
import { useResource } from '../hooks/useResource';
import { api } from '../services/api';
import { EmptyState, ErrorState, MetricCard, PageHeader, Skeleton } from '../components/UI';

export default function Dashboard() {
  const { user } = useAuth();
  const { data, error, loading } = useResource(api.dashboard);
  return (
    <>
      <PageHeader
        title={`Welcome back, ${user.name.split(' ')[0]}.`}
        action={
          <Link to="/analyze" className="button button-primary">
            New analysis
            <ArrowUpRight size={17} />
          </Link>
        }
      >
        Your images, your evidence, your next discovery.
      </PageHeader>
      <ErrorState message={error} />
      {loading ? (
        <Skeleton />
      ) : (
        data && (
          <>
            <div className="metric-grid">
              {[
                ['Total analyses', data.total, Activity],
                ['Single analyses', data.single, Image],
                ['Batch collections', data.batch, Layers],
                ['Comparisons', data.compare, GitCompareArrows],
              ].map(([label, value, Icon]) => (
                <MetricCard key={label} label={label} value={value} icon={Icon} />
              ))}
            </div>
            <section className="dashboard-highlight">
              <div>
                <span className="eyebrow">LOOK BEYOND THE FIRST IMPRESSION</span>
                <h2>
                  A little evidence.
                  <br />A clearer perspective.
                </h2>
                <p>
                  Start with a single image, then explore the ELA and frequency signals behind its
                  assessment.
                </p>
                <Link to="/analyze" className="button button-primary">
                  Explore an image
                  <ArrowUpRight size={17} />
                </Link>
              </div>
              <div className="highlight-lens">
                <Image size={55} />
                <span className="lens-ring" />
              </div>
            </section>
            <div className="quick-actions">
              {[
                [Camera, 'Live analysis', '/live-analysis', `${data.live} saved captures`],
                [Layers, 'Batch workspace', '/batch-analysis', 'Up to 10 images at a time'],
                [
                  GitCompareArrows,
                  'Compare images',
                  '/compare',
                  'Find similarities and differences',
                ],
              ].map(([Icon, title, path, detail]) => (
                <Link className="panel" key={path} to={path}>
                  <Icon />
                  <div>
                    <h3>{title}</h3>
                    <p>{detail}</p>
                  </div>
                  <ArrowUpRight size={20} />
                </Link>
              ))}
            </div>
            <section className="panel recent-activity">
              <div className="section-heading">
                <h2>Recent activity</h2>
                <Link className="text-link" to="/history">
                  View all
                  <ArrowUpRight size={16} />
                </Link>
              </div>
              {data.recent.length ? (
                data.recent.map((row) => (
                  <Link className="activity-row" key={row.id} to={`/history/${row.id}`}>
                    <span className="icon-tile">
                      <Image size={19} />
                    </span>
                    <div>
                      <strong>{row.filename}</strong>
                      <small>
                        {row.analysis_type} · {new Date(row.created_at).toLocaleString()}
                      </small>
                    </div>
                    <span className="badge badge-neutral">{row.status}</span>
                    <ArrowUpRight size={16} />
                  </Link>
                ))
              ) : (
                <EmptyState
                  title="Your first discovery starts here."
                  text="Save an analysis and it will appear in your activity."
                />
              )}
            </section>
          </>
        )
      )}
    </>
  );
}
