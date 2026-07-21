/**
 * Guided glute-bridge live coach: setup → lift → hold → lower → repeat.
 * Form corrections only run once hips are clearly lifted (avoids false
 * "alignment" cues while still lying flat).
 */

import { calcAngle, lm, mid } from "./geometry.js";

const LIFT_HOLD = 0.04; // clear hip elevation vs shoulders (normalized)
const LIFT_SETUP_MAX = 0.02; // still "down" on the floor
const SETUP_FRAMES = 12;
const HOLD_GOOD_FRAMES = 40; // ~1–2s of solid hold before cueing lower
const DOWN_FRAMES = 10;

function metrics(landmarks) {
  const lS = lm(landmarks, "left_shoulder");
  const rS = lm(landmarks, "right_shoulder");
  const lH = lm(landmarks, "left_hip");
  const rH = lm(landmarks, "right_hip");
  const lK = lm(landmarks, "left_knee");
  const rK = lm(landmarks, "right_knee");
  const lA = lm(landmarks, "left_ankle");
  const rA = lm(landmarks, "right_ankle");
  if (![lS, rS, lH, rH, lK, rK, lA, rA].every(Boolean)) return null;

  const midS = mid(lS, rS);
  const midH = mid(lH, rH);
  const midK = mid(lK, rK);
  if (!midS || !midH || !midK) return null;

  const lKnee = calcAngle(lH, lK, lA);
  const rKnee = calcAngle(rH, rK, rA);
  const avgKnee = (lKnee + rKnee) / 2;
  const lift = midS.y - midH.y;

  return {
    lS,
    rS,
    lH,
    rH,
    midS,
    midH,
    midK,
    avgKnee,
    lift,
    shoulderDiff: Math.abs(lS.y - rS.y),
    xAlign: Math.abs(midS.x - midH.x),
  };
}

/** Lying on back, knees bent, hips still down. */
function isSetup(m) {
  if (!m) return false;
  if (m.lift > LIFT_SETUP_MAX) return false;
  if (!(m.avgKnee >= 35 && m.avgKnee <= 130)) return false;
  // Knees should sit higher in the frame than hips (bent, not legs flat).
  if (m.midK.y > m.midH.y + 0.02) return false;
  return true;
}

/** Clear bridge top position. */
function isLifted(m) {
  if (!m) return false;
  if (m.lift < LIFT_HOLD) return false;
  if (m.midK.y > m.midH.y) return false;
  if (!(m.avgKnee >= 25 && m.avgKnee <= 140)) return false;
  return true;
}

function formCorrections(m, ctx) {
  const issues = [];
  const cues = [];
  if (!m || m.lift < LIFT_HOLD) return { issues, cues };

  if (m.lift < LIFT_HOLD + 0.015) {
    issues.push("hip_height");
    cues.push("Lift your hips a little higher. Squeeze your glutes at the top.");
  }

  ctx.kneeSamples = ctx.kneeSamples || [];
  ctx.kneeSamples.push(m.avgKnee);
  if (ctx.kneeSamples.length > 45) ctx.kneeSamples.shift();
  if (ctx.kneeSamples.length >= 8) {
    const sorted = [...ctx.kneeSamples].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    if (Math.abs(m.avgKnee - median) > 22) {
      issues.push("knee_angle");
      cues.push("Keep your knee bend steady. Do not let the knees fall in or out.");
    }
  }

  if (m.shoulderDiff > 0.1) {
    issues.push("shoulder_level");
    cues.push("Press both shoulders evenly into the floor.");
  }

  // Only after a clear lift; starting position often looks "misaligned" in 2D.
  if (m.lift >= LIFT_HOLD + 0.01 && m.xAlign > 0.25) {
    issues.push("alignment");
    cues.push("Keep your hips centered. Do not drift to one side.");
  }

  return { issues, cues };
}

function result({ status, cues, issues = [], stage, stageChanged = false }) {
  return {
    status,
    cues,
    issues,
    inHold: status === "hold" || status === "issue",
    stage,
    stageChanged,
    stageLabel: STAGE_LABEL[stage] || stage,
  };
}

const STAGE_LABEL = {
  find_setup: "Step 1 · Setup",
  ready_to_lift: "Step 2 · Lift",
  holding: "Step 3 · Hold",
  ready_to_lower: "Step 4 · Lower",
};

/**
 * @param {object} ctx mutable live context (persists across frames)
 */
export function evaluateBridgeCoach(landmarks, ctx) {
  if (!ctx.bridgeStage) ctx.bridgeStage = "find_setup";
  ctx.setupCount = ctx.setupCount || 0;
  ctx.holdGoodCount = ctx.holdGoodCount || 0;
  ctx.downCount = ctx.downCount || 0;

  const m = metrics(landmarks);
  const setup = isSetup(m);
  const lifted = isLifted(m);
  const prev = ctx.bridgeStage;

  if (!m) {
    return result({
      status: "ready",
      stage: ctx.bridgeStage,
      cues: [
        "I need a clearer view. Lie on your back with your full body in frame.",
      ],
      stageChanged: false,
    });
  }

  let stageChanged = false;

  if (ctx.bridgeStage === "find_setup") {
    if (setup && !lifted) {
      ctx.setupCount += 1;
    } else {
      ctx.setupCount = 0;
    }

    if (ctx.setupCount >= SETUP_FRAMES) {
      ctx.bridgeStage = "ready_to_lift";
      stageChanged = true;
      return result({
        status: "ready",
        stage: "ready_to_lift",
        stageChanged: true,
        cues: [
          "Good. Back flat, knees bent.",
          "Now squeeze your glutes and lift your hips up.",
        ],
      });
    }

    return result({
      status: "ready",
      stage: "find_setup",
      stageChanged: false,
      cues: [
        "Start on your back. Knees bent, feet flat on the floor.",
        "Keep your hips down until I say to lift.",
      ],
    });
  }

  if (ctx.bridgeStage === "ready_to_lift") {
    if (lifted) {
      ctx.bridgeStage = "holding";
      ctx.holdGoodCount = 0;
      stageChanged = true;
      return result({
        status: "hold",
        stage: "holding",
        stageChanged: true,
        cues: [
          "Hips are up. Hold here and keep squeezing your glutes.",
        ],
      });
    }

    if (!setup && !lifted) {
      ctx.bridgeStage = "find_setup";
      ctx.setupCount = 0;
      return result({
        status: "ready",
        stage: "find_setup",
        stageChanged: true,
        cues: [
          "Find the start again. Back flat, knees bent, feet on the floor.",
        ],
      });
    }

    return result({
      status: "ready",
      stage: "ready_to_lift",
      stageChanged: false,
      cues: [
        "Squeeze your glutes and press your hips toward the ceiling.",
        "Drive through your heels as you lift.",
      ],
    });
  }

  if (ctx.bridgeStage === "holding") {
    if (!lifted) {
      // Dropped early → go back toward setup / lift
      ctx.downCount += 1;
      if (ctx.downCount >= DOWN_FRAMES) {
        ctx.bridgeStage = setup ? "ready_to_lift" : "find_setup";
        ctx.holdGoodCount = 0;
        ctx.downCount = 0;
        return result({
          status: "ready",
          stage: ctx.bridgeStage,
          stageChanged: true,
          cues: setup
            ? ["Hips are down. Squeeze and lift again when ready."]
            : ["Reset. Back flat, knees bent, then lift again."],
        });
      }
      return result({
        status: "ready",
        stage: "holding",
        stageChanged: false,
        cues: ["Hips dropped. Lift back up or reset on the floor."],
      });
    }

    ctx.downCount = 0;
    const { issues, cues } = formCorrections(m, ctx);
    if (issues.length) {
      ctx.holdGoodCount = Math.max(0, ctx.holdGoodCount - 2);
      return result({
        status: "issue",
        stage: "holding",
        stageChanged: false,
        issues,
        cues: cues.slice(0, 2),
      });
    }

    ctx.holdGoodCount += 1;
    if (ctx.holdGoodCount >= HOLD_GOOD_FRAMES) {
      ctx.bridgeStage = "ready_to_lower";
      return result({
        status: "hold",
        stage: "ready_to_lower",
        stageChanged: true,
        cues: [
          "Nice hold. Now lower your hips slowly back down with control.",
        ],
      });
    }

    return result({
      status: "hold",
      stage: "holding",
      stageChanged: false,
      cues: ["Hold. Keep hips high and level."],
    });
  }

  if (ctx.bridgeStage === "ready_to_lower") {
    if (!lifted) {
      ctx.downCount += 1;
      if (ctx.downCount >= DOWN_FRAMES) {
        ctx.bridgeStage = setup ? "ready_to_lift" : "find_setup";
        ctx.holdGoodCount = 0;
        ctx.downCount = 0;
        return result({
          status: "ready",
          stage: ctx.bridgeStage,
          stageChanged: true,
          cues: [
            "Good. That was one rep.",
            "When ready, squeeze and lift again.",
          ],
        });
      }
    } else {
      ctx.downCount = 0;
      const { issues, cues } = formCorrections(m, ctx);
      if (issues.length) {
        return result({
          status: "issue",
          stage: "ready_to_lower",
          stageChanged: false,
          issues,
          cues: cues.slice(0, 2),
        });
      }
    }

    return result({
      status: lifted ? "hold" : "ready",
      stage: "ready_to_lower",
      stageChanged: prev !== "ready_to_lower" ? stageChanged : false,
      cues: [
        "Lower your hips slowly until your back is flat again.",
      ],
    });
  }

  ctx.bridgeStage = "find_setup";
  return result({
    status: "ready",
    stage: "find_setup",
    stageChanged: true,
    cues: ["Lie on your back with knees bent to begin."],
  });
}

export function resetBridgeCoach(ctx) {
  ctx.bridgeStage = "find_setup";
  ctx.setupCount = 0;
  ctx.holdGoodCount = 0;
  ctx.downCount = 0;
  ctx.kneeSamples = [];
}
