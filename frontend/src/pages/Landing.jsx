import { Link } from "react-router-dom";

export default function Landing() {
  return (
    <div className="app-shell">
      <header className="app-header" />
      <h1>PT Form Analyzer</h1>
      <p className="lead">
        Upload a short video of your physical therapy exercise and get instant
        feedback on your form — powered by pose detection, not guesswork.
      </p>

      <div className="card">
        <h2>How it works</h2>
        <ol style={{ margin: 0, paddingLeft: "1.25rem", color: "var(--muted)" }}>
          <li style={{ marginBottom: "0.5rem" }}>Pick an exercise (bird dog, bridge, or cat-cow)</li>
          <li style={{ marginBottom: "0.5rem" }}>Record yourself from the side, full body in frame</li>
          <li>Upload the video and receive a score with specific cues</li>
        </ol>
      </div>

      <div className="actions-row">
        <Link to="/exercises" className="btn btn-primary">
          Analyze my form
        </Link>
      </div>
    </div>
  );
}
