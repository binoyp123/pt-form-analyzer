export default function FormTimeline({
  timeline,
  durationSec = 0,
  currentTimeMs = 0,
  onSeek,
}) {
  if (!timeline?.length) return null;

  const endMs =
    timeline[timeline.length - 1]?.timestamp_ms || 1;
  const spanMs = Math.max(endMs, durationSec * 1000, 1);
  const playheadPct = Math.min(100, (currentTimeMs / spanMs) * 100);

  const counts = timeline.reduce(
    (acc, t) => {
      acc[t.status] = (acc[t.status] || 0) + 1;
      return acc;
    },
    {}
  );

  return (
    <div className="form-timeline card">
      <div className="form-timeline__header">
        <h2>Form timeline</h2>
        <p className="form-timeline__hint">Click a segment to jump in the video</p>
      </div>

      <div className="form-timeline__track-wrap">
        <div
          className="form-timeline__playhead"
          style={{ left: `${playheadPct}%` }}
          aria-hidden="true"
        />
        <div className="form-timeline__track" role="list">
          {timeline.map((seg) => (
            <button
              key={seg.frame_num}
              type="button"
              role="listitem"
              className={`form-timeline__seg form-timeline__seg--${seg.status}`}
              title={`${(seg.timestamp_ms / 1000).toFixed(1)}s — ${seg.status}`}
              onClick={() => onSeek(seg.timestamp_ms / 1000)}
            />
          ))}
        </div>
      </div>

      <div className="form-timeline__legend">
        <span className="legend-item legend-item--good">
          Good ({counts.good ?? 0})
        </span>
        <span className="legend-item legend-item--issue">
          Issue ({counts.issue ?? 0})
        </span>
        <span className="legend-item legend-item--neutral">
          Other ({counts.neutral ?? 0})
        </span>
      </div>
    </div>
  );
}
