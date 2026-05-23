import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchExercises } from "../api.js";

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
      <p className="lead">Select the movement you want analyzed.</p>

      {error && <div className="error-banner">{error}</div>}

      {loading && <p className="meta">Loading exercises…</p>}

      {!loading && !error && (
        <ul className="exercise-list">
          {exercises.map((ex) => (
            <li key={ex.id}>
              <Link to={`/exercise/${ex.id}`} className="exercise-link">
                <strong>{ex.name}</strong>
                <span>{ex.description}</span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
