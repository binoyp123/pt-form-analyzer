import { Link } from "react-router-dom";

export default function Landing() {
  return (
    <div className="landing">
      <section className="hero">
        <p className="hero__eyebrow">Home PT form feedback</p>
        <h1 className="hero__brand">PT Form Analyzer</h1>
        <p className="hero__tag">
          Upload a short clip — or open your webcam — and get clear joint-angle
          cues on bird dog, glute bridge, and cat-cow.
        </p>
        <div className="hero__actions">
          <Link to="/exercises" className="btn btn-primary">
            Analyze my form
          </Link>
          <Link to="/live/bird_dog" className="btn btn-secondary">
            Try live coaching
          </Link>
        </div>
      </section>

      <section className="story" aria-labelledby="story-heading">
        <h2 id="story-heading">Why I built this</h2>
        <p>
          After a herniated disc, a lot of my recovery was exercises I was
          supposed to do at home. I wanted something that could glance at my
          form the way a session in clinic does — not replace my PT, just help
          me practice with a little more confidence between visits.
        </p>
        <p>
          That became this project: MediaPipe pose estimation, rule-based
          biomechanics checks you can actually understand, and feedback you can
          act on — from a video upload or live on camera.
        </p>
      </section>

      <section className="how" aria-labelledby="how-heading">
        <h2 id="how-heading">How it works</h2>
        <ol className="how__list">
          <li>
            <strong>Pick an exercise</strong>
            <span>Bird dog, glute bridge, or cat-cow</span>
          </li>
          <li>
            <strong>Upload or go live</strong>
            <span>Scored report with skeleton overlay, or real-time cues</span>
          </li>
          <li>
            <strong>Adjust with intent</strong>
            <span>Specific joint-angle feedback — not a black-box score</span>
          </li>
        </ol>
      </section>
    </div>
  );
}
