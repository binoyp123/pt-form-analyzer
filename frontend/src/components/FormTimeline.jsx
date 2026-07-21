export default function FormTimeline({
  timeline,
  durationSec = 0,
  currentTimeMs = 0,
  onSeek,
}) {
  if (!timeline?.length) return null;

  const lastTs = timeline[timeline.length - 1]?.timestamp_ms || 0;
  const spanMs = Math.max(durationSec * 1000, lastTs, 1);
  const playheadPct = Math.min(100, Math.max(0, (currentTimeMs / spanMs) * 100));

  const counts = timeline.reduce(
    (acc, t) => {
      acc[t.status] = (acc[t.status] || 0) + 1;
      return acc;
    },
    {}
  );

  // Position each segment by timestamp so the playhead lines up with the video.
  const segments = timeline.map((seg, i) => {
    const start = seg.timestamp_ms;
    const end =
      i < timeline.length - 1 ? timeline[i + 1].timestamp_ms : spanMs;
    const left = (start / spanMs) * 100;
    const width = Math.max(0.35, ((end - start) / spanMs) * 100);
    return { seg, left, width };
  });

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
          {segments.map(({ seg, left, width }) => (
            <button
              key={seg.frame_num}
              type="button"
              role="listitem"
              className={`form-timeline__seg form-timeline__seg--${seg.status}`}
              style={{ left: `${left}%`, width: `${width}%` }}
              title={`${(seg.timestamp_ms / 1000).toFixed(1)}s · ${seg.status}`}
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
