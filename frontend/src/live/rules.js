/**
 * Client-side biomechanical checks mirrored from the Python evaluators.
 * Used for live webcam coaching (video upload still uses the FastAPI evaluators).
 */

import { calcAngle, lm, mid, visible } from "./geometry.js";
import { evaluateBridgeCoach, resetBridgeCoach } from "./bridgeCoach.js";

function birdDogHold(landmarks) {
  const lS = lm(landmarks, "left_shoulder");
  const rS = lm(landmarks, "right_shoulder");
  const lW = lm(landmarks, "left_wrist");
  const rW = lm(landmarks, "right_wrist");
  const lH = lm(landmarks, "left_hip");
  const rH = lm(landmarks, "right_hip");
  const lA = lm(landmarks, "left_ankle");
  const rA = lm(landmarks, "right_ankle");
  if (![lS, rS, lW, rW, lH, rH, lA, rA].every(Boolean)) return false;

  const lArmExt =
    Math.abs(lW.y - lS.y) < 0.2 || Math.abs(lW.x - lS.x) > 0.15;
  const rArmExt =
    Math.abs(rW.y - rS.y) < 0.2 || Math.abs(rW.x - rS.x) > 0.15;
  const lLegExt =
    Math.abs(lA.y - lH.y) < 0.25 || Math.abs(lA.x - lH.x) > 0.15;
  const rLegExt =
    Math.abs(rA.y - rH.y) < 0.25 || Math.abs(rA.x - rH.x) > 0.15;

  const lArmReach = Math.abs(lW.x - lS.x) > 0.1 || Math.abs(lW.y - lS.y) < 0.15;
  const rArmReach = Math.abs(rW.x - rS.x) > 0.1 || Math.abs(rW.y - rS.y) < 0.15;
  const lLegReach = Math.abs(lA.x - lH.x) > 0.1 || Math.abs(lA.y - lH.y) < 0.2;
  const rLegReach = Math.abs(rA.x - rH.x) > 0.1 || Math.abs(rA.y - rH.y) < 0.2;

  return (
    (lArmExt && lArmReach && rLegExt && rLegReach) ||
    (rArmExt && rArmReach && lLegExt && lLegReach)
  );
}

/** Rough check: body flattened on the floor instead of hands-and-knees. */
function looksLyingFlat(landmarks) {
  const points = [
    lm(landmarks, "left_shoulder"),
    lm(landmarks, "right_shoulder"),
    lm(landmarks, "left_hip"),
    lm(landmarks, "right_hip"),
    lm(landmarks, "left_knee"),
    lm(landmarks, "right_knee"),
    lm(landmarks, "left_ankle"),
    lm(landmarks, "right_ankle"),
  ].filter(Boolean);
  if (points.length < 6) return false;
  const ys = points.map((p) => p.y);
  return Math.max(...ys) - Math.min(...ys) < 0.14;
}

function birdDogReadyCues(landmarks) {
  if (looksLyingFlat(landmarks)) {
    return [
      "You look flat on the floor. Get onto hands and knees first.",
      "Then reach one arm forward and kick the opposite leg straight back.",
    ];
  }
  return [
    "From hands and knees, reach one arm forward.",
    "At the same time, kick the opposite leg straight back and hold.",
  ];
}

function checkBirdDog(landmarks) {
  const issues = [];
  const cues = [];
  const lS = lm(landmarks, "left_shoulder");
  const rS = lm(landmarks, "right_shoulder");
  const lH = lm(landmarks, "left_hip");
  const rH = lm(landmarks, "right_hip");
  const lK = lm(landmarks, "left_knee");
  const rK = lm(landmarks, "right_knee");
  const lA = lm(landmarks, "left_ankle");
  const rA = lm(landmarks, "right_ankle");
  const lW = lm(landmarks, "left_wrist");
  const rW = lm(landmarks, "right_wrist");

  const midS = mid(lS, rS);
  const midH = mid(lH, rH);
  if (midS && midH) {
    const dy = Math.abs(midS.y - midH.y);
    const dx = Math.abs(midS.x - midH.x);
    if (dx > 0.01 && dy / dx > 0.5) {
      issues.push("back_arch");
      cues.push("Flatten your back. Pull your belly in so your hips stay level.");
    }
  }

  const lDiff = lW && lS ? Math.abs(lW.y - lS.y) : 99;
  const rDiff = rW && rS ? Math.abs(rW.y - rS.y) : 99;
  const leftArm = lDiff <= rDiff;
  const armSide = leftArm ? "left" : "right";
  const legSide = leftArm ? "right" : "left";
  const shoulder = leftArm ? lS : rS;
  const wrist = leftArm ? lW : rW;
  const hip = leftArm ? rH : lH;
  const knee = leftArm ? rK : lK;
  const ankle = leftArm ? rA : lA;

  if (shoulder && wrist && Math.abs(shoulder.y - wrist.y) >= 0.35) {
    issues.push("arm_not_parallel");
    cues.push(
      `Raise or lower your ${armSide} arm until it lines up parallel with the floor.`
    );
  }
  if (hip && ankle && Math.abs(hip.y - ankle.y) >= 0.35) {
    issues.push("leg_not_parallel");
    cues.push(
      `Lift your ${legSide} leg until it is parallel with the floor, heel pointing back.`
    );
  }
  if (hip && knee && ankle && calcAngle(hip, knee, ankle) <= 120) {
    issues.push("leg_bent");
    cues.push(`Straighten your ${legSide} knee so the leg reaches long.`);
  }

  return { issues, cues };
}

function spineMetric(landmarks) {
  const lS = lm(landmarks, "left_shoulder");
  const rS = lm(landmarks, "right_shoulder");
  const lH = lm(landmarks, "left_hip");
  const rH = lm(landmarks, "right_hip");
  if (![lS, rS, lH, rH].every((p) => visible(p))) return null;
  const midS = mid(lS, rS);
  const midH = mid(lH, rH);
  if (!midS || !midH) return null;
  return midH.y - midS.y;
}

function checkCatCow(landmarks, ctx) {
  const issues = [];
  const cues = [];
  const m = spineMetric(landmarks);
  if (m == null) {
    return {
      status: "ready",
      issues,
      cues: [
        "Move so I can see your shoulders and hips from the side.",
        "Stay on hands and knees.",
      ],
      inHold: false,
    };
  }

  ctx.spine = ctx.spine || [];
  ctx.spine.push(m);
  if (ctx.spine.length > 60) ctx.spine.shift();

  if (ctx.spine.length < 12) {
    return {
      status: "ready",
      issues,
      cues: [
        "Round your back up for cat, then drop your belly for cow.",
        "Move slowly between those two shapes.",
      ],
      inHold: true,
    };
  }

  const span = Math.max(...ctx.spine) - Math.min(...ctx.spine);
  const recent = ctx.spine.slice(-8);
  const delta = Math.abs(recent[recent.length - 1] - recent[0]);

  if (span < 0.02) {
    issues.push("amplitude");
    cues.push("Make the motion bigger. Round up more, then drop your chest more.");
  } else if (delta < 0.008) {
    issues.push("rhythm");
    cues.push("Keep alternating. Do not freeze in the middle.");
  } else {
    cues.push("Good. Keep slow cat and cow rounds.");
  }

  return {
    status: issues.length ? "issue" : "hold",
    issues,
    cues,
    inHold: true,
  };
}

/**
 * @returns {{ status: 'ready'|'hold'|'issue', cues: string[], issues: string[], inHold: boolean, stage?: string, stageLabel?: string, stageChanged?: boolean }}
 */
export function evaluateLiveFrame(exerciseId, landmarks, ctx = {}) {
  if (!landmarks?.length) {
    return {
      status: "ready",
      cues: ["Step back so your full body is in the camera frame."],
      issues: [],
      inHold: false,
    };
  }

  if (exerciseId === "bird_dog") {
    const inHold = birdDogHold(landmarks);
    if (!inHold) {
      return {
        status: "ready",
        cues: birdDogReadyCues(landmarks),
        issues: [],
        inHold: false,
      };
    }
    const { issues, cues } = checkBirdDog(landmarks);
    return {
      status: issues.length ? "issue" : "hold",
      cues: cues.length ? cues.slice(0, 2) : ["Hold it. Your bird dog form looks solid."],
      issues,
      inHold: true,
    };
  }

  if (exerciseId === "bridge") {
    return evaluateBridgeCoach(landmarks, ctx);
  }

  if (exerciseId === "cat_cow") {
    return checkCatCow(landmarks, ctx);
  }

  return {
    status: "ready",
    cues: ["Pick bird dog, bridge, or cat-cow."],
    issues: [],
    inHold: false,
  };
}

export function createLiveContext() {
  const ctx = { kneeSamples: [], spine: [] };
  resetBridgeCoach(ctx);
  return ctx;
}
