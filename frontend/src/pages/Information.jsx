import { Link } from 'react-router-dom';
import { PageHeader } from '../components/UI';

export function About() {
  return (
    <div className="information-page">
      <PageHeader eyebrow="BEHIND SECURELENS" title="Evidence before certainty.">
        AI Image Authenticity & Manipulation Detection, designed for careful image research.
      </PageHeader>
      <section className="panel">
        <h2>How the engine works</h2>
        <p>
          SecureLens measures JPEG recompression differences with ELA, spatial frequency patterns
          with FFT, and image statistics such as brightness, entropy, edges and compression loss.
        </p>
        <p>
          The current assessment reuses two existing heuristic rules: gain-20 ELA mean below 8 adds
          two points, and FFT mean above 120 adds one. Zero is Low, one or two is Moderate, and
          three is High.
        </p>
        <p>
          These are unvalidated indicators. They are sensitive to scene content and resolution.
          Genuine photos can trigger both rules. There is no validated AI classifier or
          manipulation-localization classifier in the primary app, so no model probabilities are
          shown.
        </p>
        <h3>Why multiple workflows?</h3>
        <p>
          Single uploads, live frames, batches and image comparisons provide different ways to
          examine the same evidence. Comparison similarity describes pixels and histograms, not
          whether an image is real.
        </p>
        <h3>Next: validated machine learning</h3>
        <p>
          A representative labeled dataset, independent held-out testing, documented preprocessing,
          correct class mapping and calibrated probabilities are required before a trained
          classifier can produce reliable confidence.
        </p>
        <Link className="button button-primary" to="/analyze">
          Try the workspace
        </Link>
      </section>
    </div>
  );
}
export function Privacy() {
  return (
    <div className="information-page">
      <PageHeader eyebrow="PRIVACY & STORAGE" title="Your images. Your choices.">
        Plain-language information about this development application; formal legal review is still
        needed before public launch.
      </PageHeader>
      <section className="panel prose">
        <h2>What you provide</h2>
        <p>
          Your display name, email and password are used to create an account. Passwords are stored
          as Argon2 hashes. Image uploads are sent to the analysis API for processing.
        </p>
        <h2>What is saved</h2>
        <p>
          When you choose Save to history, analysis records and JSON reports are saved in PostgreSQL
          and private report storage. Local development uses local files; a deployment must use a
          persistent private disk or a configured private cloud adapter. Records include sanitized
          filenames, timestamps, summary metrics and heuristic results. The original Django database
          remains separate.
        </p>
        <h2>Image retention</h2>
        <p>
          Uploads are processed temporarily and may use temporary disk buffering. These upload files
          are closed after processing. If you explicitly enable preview retention, downscaled
          original/ELA/FFT previews are saved privately for the configured window (7 days by
          default). Camera frames use session-only analysis unless you choose Save current frame or
          save a captured batch.
        </p>
        <p>
          Expired previews are removed on history requests and by the API's periodic cleanup worker
          while it is running (hourly by default). Cleanup also runs when the API starts. If the
          service is stopped, removal resumes when it starts again. A maintenance command is
          available too. Reports remain until you delete the analysis.
        </p>
        <h2>Cookies and local preferences</h2>
        <p>
          An essential HttpOnly authentication cookie keeps your session. Cookie preferences use
          local browser storage and are synced to your account when signed in. No advertising or
          analytics services are installed.
        </p>
        <p>Fonts may be requested from Google Fonts; they are not an analytics integration.</p>
        <h2>Your controls</h2>
        <p>
          You can delete individual analyses, their reports and retained previews from History.
          Downloaded files on your device are outside the app's control. Account deletion and
          automated email recovery are not implemented yet.
        </p>
        <p>
          Do not upload sensitive images to a public deployment before its storage, retention,
          access controls and privacy notice have been reviewed.
        </p>
      </section>
    </div>
  );
}

export function CookiePreferences() {
  return (
    <div className="information-page">
      <PageHeader eyebrow="COOKIE PREFERENCES" title="Only what your workspace needs.">
        Essential sign-in cookies keep your session secure. No optional trackers are installed.
      </PageHeader>
      <section className="panel">
        <h2>Essential and authentication</h2>
        <p>
          SecureLens uses one HttpOnly session cookie. It is required for account features and
          expires after the configured session window. Your sign-in token is not stored in browser
          local storage.
        </p>
        <h2>Analytics</h2>
        <p>
          Not used. Accepting and rejecting non-essential cookies currently have the same effect
          because there are no optional analytics or advertising cookies.
        </p>
        <h2>Remembering your choice</h2>
        <p>
          Cookie preferences are stored in this browser and synced to your account when signed in.
        </p>
        <button
          className="button button-primary"
          onClick={() => window.dispatchEvent(new Event('securelens:cookies'))}
        >
          Manage preferences
        </button>
        <p>
          <Link to="/privacy" className="text-link">
            Read the privacy notice
          </Link>
        </p>
      </section>
    </div>
  );
}
