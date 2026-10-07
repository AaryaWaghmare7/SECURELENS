import { Link } from 'react-router-dom';
import {
  ArrowUpRight,
  ArrowRight,
  ScanLine,
  Camera,
  Layers,
  GitCompareArrows,
  ShieldCheck,
  Activity,
  Image,
  Fingerprint,
  LockKeyhole,
  Sparkles,
  Waves,
  Check,
} from 'lucide-react';

const features = [
  [
    Image,
    'Single image analysis',
    'One image. Multiple perspectives. ELA, frequencies and pixel evidence.',
  ],
  [
    Camera,
    'A live point of view',
    'Analyze camera frames without automatically saving your captures.',
  ],
  [Layers, 'Research at scale', 'Review image batches, follow progress and export your findings.'],
  [
    GitCompareArrows,
    'Side-by-side clarity',
    'Compare image signals, compression and visual similarity.',
  ],
];
export default function Landing() {
  return (
    <div className="landing">
      <section className="landing-hero">
        <div className="hero-copy">
          <span className="pill">
            <Sparkles size={13} /> IMAGE INTELLIGENCE, WITH INTENTION
          </span>
          <h1>
            SecureLens
            <br />
            See beyond the <em>pixels.</em>
          </h1>
          <h2>AI Image Authenticity Detection</h2>
          <p>Analyze images for potential AI generation, manipulation and forensic anomalies.</p>
          <small>Forensic indicators today. Not a validated AI classifier.</small>
          <div className="hero-buttons">
            <Link to="/analyze" className="button button-primary">
              Analyze an image
              <ArrowUpRight size={18} />
            </Link>
            <a href="#how-it-works" className="button button-secondary">
              Explore SecureLens
              <ArrowRight size={17} />
            </a>
          </div>
          <div className="hero-proof">
            <span>
              <Check size={15} />
              Evidence-led results
            </span>
            <span>
              <LockKeyhole size={15} />
              Private by default
            </span>
            <span>
              <Waves size={15} />
              ELA + FFT
            </span>
          </div>
        </div>
        <div className="hero-art" aria-label="Illustration of the SecureLens analysis workflow">
          <div className="art-orbit orbit-a" />
          <div className="art-orbit orbit-b" />
          <div className="art-dot dot-a" />
          <div className="art-dot dot-b" />
          <div className="art-small">
            <ScanLine size={16} /> SIGNAL ANALYSIS
            <span className="pulse-dot" />
          </div>
          <div className="art-main">
            <div className="art-top">
              <span>SecureLens / evidence studio</span>
              <span>01</span>
            </div>
            <div className="sample-scene">
              <div className="scene-sun" />
              <div className="scene-hill hill-one" />
              <div className="scene-hill hill-two" />
              <div className="scene-grid" />
              <span className="scan-corner corner-a" />
              <span className="scan-corner corner-b" />
              <div className="scan-sweep" />
              <span className="sample-tag">ILLUSTRATIVE PREVIEW</span>
            </div>
            <div className="art-caption">
              <span className="art-shield">
                <Fingerprint size={25} />
              </span>
              <div>
                <strong>Every image has a story.</strong>
                <small>Let's examine the evidence.</small>
              </div>
              <ArrowUpRight size={18} />
            </div>
          </div>
          <div className="art-evidence">
            <div className="icon-tile">
              <Activity size={21} />
            </div>
            <div>
              <strong>Behind the pixels</strong>
              <span>ELA · Frequency · Compression</span>
            </div>
            <div className="mini-bars">
              {[20, 36, 26, 48, 34, 54, 31].map((n, i) => (
                <i key={i} style={{ height: n }} />
              ))}
            </div>
          </div>
        </div>
      </section>
      <div className="method-strip">
        <span>A closer look, from every angle.</span>
        <strong>ERROR LEVEL ANALYSIS</strong>
        <i />
        <strong>FREQUENCY EVIDENCE</strong>
        <i />
        <strong>PIXEL STATISTICS</strong>
        <i />
        <strong>COMPRESSION SIGNALS</strong>
      </div>
      <section id="how-it-works" className="landing-section">
        <div className="section-intro">
          <span className="eyebrow">A LITTLE CURIOSITY. A LOT OF EVIDENCE.</span>
          <h2>From upload to understanding.</h2>
          <p>One thoughtful workflow, with the technical detail always within reach.</p>
        </div>
        <div className="steps-grid">
          {[
            ['01', 'Bring an image', 'Upload a photo, collect a camera frame, or gather a batch.'],
            [
              '02',
              'Explore its signals',
              'SecureLens measures recompression, frequency and pixel patterns.',
            ],
            [
              '03',
              'Make an informed call',
              'Review a cautious assessment and the evidence that shaped it.',
            ],
          ].map(([n, title, text]) => (
            <div key={n}>
              <span className="step-number">{n}</span>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
          ))}
        </div>
      </section>
      <section className="landing-section feature-section">
        <div className="section-intro">
          <span className="eyebrow">YOUR COMPLETE IMAGE WORKSPACE</span>
          <h2>Curiosity deserves better tools.</h2>
        </div>
        <div className="features-grid">
          {features.map(([Icon, title, text], i) => (
            <Link
              to={['/analyze', '/live-analysis', '/batch-analysis', '/compare'][i]}
              className="feature-card"
              key={title}
            >
              <div className="icon-tile">
                <Icon />
              </div>
              <h3>{title}</h3>
              <p>{text}</p>
              <ArrowUpRight className="feature-arrow" size={20} />
            </Link>
          ))}
        </div>
      </section>
      <section className="landing-section" id="analysis-methods">
        <div className="section-intro">
          <span className="eyebrow">SUPPORTING EVIDENCE, NOT A BLACK BOX</span>
          <h2>The methods behind the result.</h2>
          <p>Measured signals help explain the assessment. None proves an image is AI-generated.</p>
        </div>
        <div className="features-grid">
          {[
            [
              ScanLine,
              'Error level analysis',
              'JPEG recompression differences, with raw and enhanced error values.',
            ],
            [
              Waves,
              'Frequency analysis',
              'FFT spectrum, high-frequency energy and spectral centroid.',
            ],
            [
              Activity,
              'Pixel statistics',
              'Brightness, deviation, entropy, edges and RGB channel statistics.',
            ],
            [
              Layers,
              'Compression & metadata',
              'Resolution, channels, EXIF presence and measured JPEG quality experiments.',
            ],
          ].map(([Icon, title, text]) => (
            <article className="feature-card" key={title}>
              <div className="icon-tile">
                <Icon />
              </div>
              <h3>{title}</h3>
              <p>{text}</p>
            </article>
          ))}
        </div>
      </section>
      <section className="landing-section why-grid">
        <div>
          <span className="eyebrow">WHY SECURELENS</span>
          <h2>
            Transparency is
            <br />
            the real advantage.
          </h2>
          <p>
            We show the signals and their limitations. No made-up confidence, no hidden certainty. A
            workspace for examining an image, not guessing at a number.
          </p>
          <Link to="/about" className="text-link">
            Inside the analysis engine
            <ArrowRight size={18} />
          </Link>
        </div>
        <div className="panel privacy-card">
          <div className="icon-tile">
            <ShieldCheck />
          </div>
          <h3>Your images. Your choices.</h3>
          <p>
            Results stay in your account only when you choose to save them. Image previews are
            optional and have a retention window.
          </p>
          <div className="privacy-points">
            <span>
              <LockKeyhole size={17} />
              Protected account history
            </span>
            <span>
              <Fingerprint size={17} />
              No analytics trackers
            </span>
            <span>
              <ScanLine size={17} />
              Clear, inspectable methods
            </span>
          </div>
          <Link to="/privacy" className="text-link">
            Privacy & storage
            <ArrowUpRight size={17} />
          </Link>
        </div>
      </section>
      <section className="landing-cta">
        <span className="eyebrow">LOOK A LITTLE CLOSER</span>
        <h2>
          Your next image deserves
          <br />a second perspective.
        </h2>
        <Link to="/signup" className="button button-primary">
          Create your workspace
          <ArrowUpRight size={18} />
        </Link>
      </section>
    </div>
  );
}
