import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchExercises } from "../api.js";
import ExerciseStepViewer from "../components/ExerciseStepViewer.jsx";
import { getGuide } from "../data/exerciseGuides.js";

export default function ExerciseSelect() {
  const [exercises, setExercises] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchExercises()
      .then(setExercises)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <Link to="/">← Home</Link>
      </header>
      <h1>Choose an exercise</h1>
      <p className="lead">
        Tap an exercise to see step-by-step form, upload a video, or open live
        coaching.
      </p>

      {error && <div className="error-banner">{error}</div>}

      {loading && <p className="meta">Loading exercises…</p>}

      {!loading && !error && (
        <ul className="exercise-list">
          {exercises.map((ex) => {
            const guide = getGuide(ex.id);
            return (
              <li key={ex.id}>
                <Link to={`/exercise/${ex.id}`} className="exercise-card">
                  <div className="exercise-card__visual">
                    <ExerciseStepViewer
                      exerciseId={ex.id}
                      size="compact"
                      initialStep={1}
                      showCaption={false}
                      showControls={false}
                    />
                  </div>
                  <div className="exercise-card__body">
                    <strong>{ex.name}</strong>
                    <span className="exercise-card__desc">{ex.description}</span>
                    {guide?.phases && (
                      <span className="exercise-card__hint">
                        Upload · live coaching · 3 form steps
                      </span>
                    )}
                  </div>
                  <span className="exercise-card__arrow" aria-hidden="true">
                    →
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
