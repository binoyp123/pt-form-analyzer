import { useState } from "react";
import { getGuide } from "../data/exerciseGuides.js";

/**
 * Step-through exercise images — click a phase or use arrows to change step.
 */
export default function ExerciseStepViewer({
  exerciseId,
  size = "large",
  initialStep = 0,
  showCaption = true,
  showControls = true,
}) {
  const guide = getGuide(exerciseId);
  if (!guide?.steps?.length) return null;

  const [step, setStep] = useState(initialStep);
  const total = guide.steps.length;
  const current = guide.steps[step];
  const phaseLabel = guide.phases[step] ?? `Step ${step + 1}`;

  function go(delta) {
    setStep((s) => (s + delta + total) % total);
  }

  const compact = size === "compact";

  return (
    <div
      className={`step-viewer step-viewer--${size}`}
      aria-label={`${exerciseId.replace(/_/g, " ")} demonstration`}
    >
      <div className="step-viewer__frame">
        <img
          key={current.image}
          src={current.image}
          alt={`${phaseLabel}: ${current.caption}`}
          className="step-viewer__img"
        />
        {showControls && !compact && (
          <>
            <button
              type="button"
              className="step-viewer__nav step-viewer__nav--prev"
              onClick={() => go(-1)}
              aria-label="Previous step"
            >
              ‹
            </button>
            <button
              type="button"
              className="step-viewer__nav step-viewer__nav--next"
              onClick={() => go(1)}
              aria-label="Next step"
            >
              ›
            </button>
          </>
        )}
      </div>

      {showCaption && !compact && (
        <p className="step-viewer__caption">{current.caption}</p>
      )}

      {showControls && (
        <div className="phase-pills" role="tablist" aria-label="Exercise steps">
          {guide.phases.map((phase, i) => (
            <button
              key={phase}
              type="button"
              role="tab"
              aria-selected={i === step}
              className={`phase-pill ${i === step ? "phase-pill--active" : ""}`}
              onClick={() => setStep(i)}
            >
              <span className="phase-pill__num">{i + 1}</span>
              {phase}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
