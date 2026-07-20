import { Link } from "react-router-dom";

export default function Landing() {
  return (
    <div className="landing">
      <section className="hero">
        <p className="hero__eyebrow">Home PT form feedback</p>
        <h1 className="hero__brand">PT Form Analyzer</h1>
        <p className="hero__tag">
          Upload a short clip or open your webcam and get clear joint-angle cues
          on bird dog, glute bridge, and cat-cow.
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
          About three years ago I hurt my lower back. Imaging later showed a
          mild herniated disc at L5-S1, and I still do PT-style exercises at
          home. In-person physical therapy helps a lot, but it gets expensive
          without good insurance coverage, and I cannot always get into the
          clinic as often as I would like.
        </p>
        <p>
          Online videos show you what the movement should look like, but they
          cannot tell you whether your form is actually right or wrong in the
          moment. I built this so I (and anyone else practicing at home) could
          get simple, specific feedback on form from an uploaded clip or a live
          camera, without pretending to replace a real PT.
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
            <span>Specific joint-angle feedback, not a black-box score</span>
          </li>
        </ol>
      </section>
    </div>
  );
}
