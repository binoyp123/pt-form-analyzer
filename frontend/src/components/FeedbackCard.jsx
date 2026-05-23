const ICONS = { good: "✓", warning: "!", error: "✕" };

export default function FeedbackCard({ status, message }) {
  return (
    <div className={`feedback-item ${status}`}>
      <span className="feedback-icon" aria-hidden="true">
        {ICONS[status] || "•"}
      </span>
      <span className="feedback-message">{message}</span>
    </div>
  );
}
