import ExerciseStepViewer from "./ExerciseStepViewer.jsx";
import { getGuide } from "../data/exerciseGuides.js";

export default function FilmingGuide({ exerciseId }) {
  const guide = getGuide(exerciseId);
  if (!guide) return null;

  return (
    <section className="filming-guide" aria-label="Exercise and filming guide">
      <div className="filming-guide__hero card">
        <ExerciseStepViewer exerciseId={exerciseId} size="large" />
      </div>

      <div className="guide-grid">
        <GuideBlock icon="📷" {...guide.film} variant="film" />
        <GuideBlock icon="✓" {...guide.movement} variant="move" />
        <GuideBlock icon="!" {...guide.avoid} variant="avoid" />
      </div>
    </section>
  );
}

function GuideBlock({ icon, title, items, variant }) {
  return (
    <div className={`guide-block guide-block--${variant} card`}>
      <h3>
        <span className="guide-block__icon" aria-hidden="true">
          {icon}
        </span>
        {title}
      </h3>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
