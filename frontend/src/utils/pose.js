export function findPoseAtTime(poseFrames, timeMs) {
  if (!poseFrames?.length) return null;
  let best = poseFrames[0];
  let bestDiff = Math.abs(best.timestamp_ms - timeMs);
  for (let i = 1; i < poseFrames.length; i++) {
    const d = Math.abs(poseFrames[i].timestamp_ms - timeMs);
    if (d < bestDiff) {
      best = poseFrames[i];
      bestDiff = d;
    }
  }
  return best;
}

export function findTimelineEntry(timeline, frameNum) {
  return timeline?.find((t) => t.frame_num === frameNum) ?? null;
}
