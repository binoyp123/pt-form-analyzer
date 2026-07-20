import { Link } from "react-router-dom";

export default function Landing() {
  return (
    <div className="app-shell">
      <header className="app-header" />
      <h1>PT Form Analyzer</h1>
      <p className="lead">
        Analyze physical therapy form from video — or get live cues from your
        webcam — using MediaPipe pose estimation and rule-based biomechanics
        checks.
      </p>

      <div className="card">
        <h2>How it works</h2>
        <ol style={{ margin: 0, paddingLeft: "1.25rem", color: "var(--muted)" }}>
          <li style={{ marginBottom: "0.5rem" }}>
            Pick an exercise (bird dog, glute bridge, or cat-cow)
          </li>
          <li style={{ marginBottom: "0.5rem" }}>
            Upload a short clip for a scored report with skeleton overlay — or
            open live coaching for real-time cues
          </li>
          <li>
            Get joint-angle feedback you can act on (not a black-box score)
          </li>
        </ol>
      </div>

      <div className="actions-row">
        <Link to="/exercises" className="btn btn-primary">
          Analyze my form
        </Link>
        <Link to="/live/bird_dog" className="btn btn-secondary">
          Try live coaching
        </Link>
      </div>
    </div>
  );
}
