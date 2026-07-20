/** Shared 2D geometry helpers for live coaching (mirrors Python pose_extractor). */

export const LANDMARKS = {
  nose: 0,
  left_shoulder: 11,
  right_shoulder: 12,
  left_elbow: 13,
  right_elbow: 14,
  left_wrist: 15,
  right_wrist: 16,
  left_hip: 23,
  right_hip: 24,
  left_knee: 25,
  right_knee: 26,
  left_ankle: 27,
  right_ankle: 28,
};

export function lm(landmarks, name) {
  const idx = LANDMARKS[name];
  if (idx == null || !landmarks?.[idx]) return null;
  const p = landmarks[idx];
  return {
    x: p.x,
    y: p.y,
    z: p.z ?? 0,
    visibility: p.visibility ?? 1,
  };
}

export function mid(a, b) {
  if (!a || !b) return null;
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: ((a.z || 0) + (b.z || 0)) / 2,
  };
}

export function calcAngle(p1, p2, p3) {
  if (!p1 || !p2 || !p3) return 0;
  const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
  const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
  const mag1 = Math.hypot(v1.x, v1.y);
  const mag2 = Math.hypot(v2.x, v2.y);
  if (mag1 * mag2 === 0) return 0;
  const cos = Math.max(-1, Math.min(1, (v1.x * v2.x + v1.y * v2.y) / (mag1 * mag2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

export function visible(p, threshold = 0.5) {
  return p != null && (p.visibility ?? 0) >= threshold;
}
