const ICONS = { good: "✓", warning: "!", error: "✕" };

export default function FeedbackCard({
  status,
  message,
  problemFrames = [],
  onSeekFrame,
}) {
  const seekable =
    typeof onSeekFrame === "function" &&
    Array.isArray(problemFrames) &&
    problemFrames.length > 0;

  function handleClick() {
    if (!seekable) return;
    onSeekFrame(problemFrames[0]);
  }

  function handleKeyDown(e) {
    if (!seekable) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleClick();
    }
  }

  return (
    <div
      className={`feedback-item ${status}${seekable ? " seekable" : ""}`}
      role={seekable ? "button" : undefined}
      tabIndex={seekable ? 0 : undefined}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      title={seekable ? "Jump to first problem frame" : undefined}
    >
      <span className="feedback-icon" aria-hidden="true">
        {ICONS[status] || "•"}
      </span>
      <span className="feedback-message">{message}</span>
      {seekable && (
        <span className="feedback-seek-hint" aria-hidden="true">
          Jump →
        </span>
      )}
    </div>
  );
}
