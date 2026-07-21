/**
 * Guided glute-bridge live coach: setup → lift → hold → lower → repeat.
 *
 * Hold quality is scored against an explicit template:
 *   both feet planted, both knees bent similarly, hips clearly lifted,
 *   shoulders level, hips centered.
 */

import { calcAngle, lm, mid } from "./geometry.js";

const LIFT_HOLD = 0.035;
const LIFT_SETUP_MAX = 0.02;
const SETUP_FRAMES = 12;
const HOLD_GOOD_FRAMES = 36;
const DOWN_FRAMES = 10;

/** Ideal hold template (normalized / degrees). */
const HOLD_TEMPLATE = {
  minLift: LIFT_HOLD,
  kneeMin: 35,
  kneeMax: 125,
  maxKneeAsymmetry: 22,
  straightLeg: 145, // above this ≈ leg kicked straight up
  maxAnkleYDiff: 0.09,
  maxShoulderDiff: 0.1,
  maxXAlign: 0.24,
};

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
    lK,
    rK,
    lA,
    rA,
    midS,
    midH,
    midK,
    lKnee,
    rKnee,
    avgKnee,
    lift,
    ankleYDiff: Math.abs(lA.y - rA.y),
    kneeAsym: Math.abs(lKnee - rKnee),
    shoulderDiff: Math.abs(lS.y - rS.y),
    xAlign: Math.abs(midS.x - midH.x),
  };
}

function isSetup(m) {
  if (!m) return false;
  if (m.lift > LIFT_SETUP_MAX) return false;
  if (!(m.avgKnee >= 35 && m.avgKnee <= 130)) return false;
  if (m.midK.y > m.midH.y + 0.02) return false;
  // Both feet should look planted in setup too.
  if (m.lKnee > HOLD_TEMPLATE.straightLeg || m.rKnee > HOLD_TEMPLATE.straightLeg) {
    return false;
  }
  return true;
}

function isLifted(m) {
  if (!m) return false;
  if (m.lift < LIFT_HOLD) return false;
  if (m.midK.y > m.midH.y) return false;
  if (!(m.avgKnee >= 25 && m.avgKnee <= 150)) return false;
  return true;
}

/**
 * Compare current pose to the hold template.
 * Returns score 0–100 plus ranked cues (worst first).
 */
function matchHoldTemplate(m) {
  const issues = [];
  const cues = [];
  let score = 100;
  if (!m) return { score: 0, issues, cues };

  const T = HOLD_TEMPLATE;

  // Critical: one leg kicked straight up / off the floor
  if (m.lKnee >= T.straightLeg || m.rKnee >= T.straightLeg) {
    const side = m.lKnee >= m.rKnee ? "left" : "right";
    issues.push("leg_raised");
    cues.push(
      `Plant your ${side} foot back on the floor. Both feet stay down in a bridge.`
    );
    score -= 45;
  } else if (m.ankleYDiff > T.maxAnkleYDiff) {
    const side = m.lA.y < m.rA.y ? "left" : "right";
    issues.push("foot_lifted");
    cues.push(
      `Your ${side} foot looks lifted. Keep both feet flat on the floor.`
    );
    score -= 35;
  }

  if (m.kneeAsym > T.maxKneeAsymmetry && !issues.includes("leg_raised")) {
    issues.push("knee_asym");
    cues.push(
      "Bend both knees about the same. One leg should not be straighter than the other."
    );
    score -= 20;
  }

  if (m.lift < T.minLift + 0.02) {
    issues.push("hip_height");
    cues.push("Lift your hips higher and squeeze your glutes at the top.");
    score -= 25;
  } else if (m.lift < T.minLift + 0.035) {
    score -= 8;
  }

  if (m.avgKnee < T.kneeMin || m.avgKnee > T.kneeMax) {
    if (!issues.includes("leg_raised")) {
      issues.push("knee_range");
      cues.push(
        m.avgKnee > T.kneeMax
          ? "Bend your knees more. Feet should stay under your knees."
          : "Let your knees open a bit more into a comfortable bridge bend."
      );
      score -= 15;
    }
  }

  if (m.shoulderDiff > T.maxShoulderDiff) {
    issues.push("shoulder_level");
    cues.push("Press both shoulders evenly into the floor.");
    score -= 12;
  }

  if (m.lift >= T.minLift + 0.015 && m.xAlign > T.maxXAlign) {
    issues.push("alignment");
    cues.push("Keep your hips centered. Do not drift sideways.");
    score -= 15;
  }

  score = Math.max(0, Math.min(100, Math.round(score)));

  if (issues.length === 0) {
    cues.push("Good form. Both feet planted, hips high. Keep holding.");
  }

  return { score, issues, cues: cues.slice(0, 2) };
}

function result({
  status,
  cues,
  issues = [],
  stage,
  stageChanged = false,
  formMatch = null,
}) {
  return {
    status,
    cues,
    issues,
    inHold: status === "hold" || status === "issue",
    stage,
    stageChanged,
    stageLabel: STAGE_LABEL[stage] || stage,
    formMatch,
  };
}

const STAGE_LABEL = {
  find_setup: "Step 1 · Setup",
  ready_to_lift: "Step 2 · Lift",
  holding: "Step 3 · Hold",
  ready_to_lower: "Step 4 · Lower",
};

export function evaluateBridgeCoach(landmarks, ctx) {
  if (!ctx.bridgeStage) ctx.bridgeStage = "find_setup";
  ctx.setupCount = ctx.setupCount || 0;
  ctx.holdGoodCount = ctx.holdGoodCount || 0;
  ctx.downCount = ctx.downCount || 0;
  ctx.lastGoodSpeakAt = ctx.lastGoodSpeakAt || 0;

  const m = metrics(landmarks);
  const setup = isSetup(m);
  const lifted = isLifted(m);

  if (!m) {
    return result({
      status: "ready",
      stage: ctx.bridgeStage,
      cues: [
        "I need a clearer view. Lie on your back with your full body in frame.",
      ],
      formMatch: 0,
    });
  }

  if (ctx.bridgeStage === "find_setup") {
    if (setup && !lifted) ctx.setupCount += 1;
    else ctx.setupCount = 0;

    if (ctx.setupCount >= SETUP_FRAMES) {
      ctx.bridgeStage = "ready_to_lift";
      return result({
        status: "ready",
        stage: "ready_to_lift",
        stageChanged: true,
        formMatch: 70,
        cues: [
          "Good start. Back flat, knees bent, feet planted.",
          "Now squeeze your glutes and lift your hips up.",
        ],
      });
    }

    return result({
      status: "ready",
      stage: "find_setup",
      formMatch: setup ? 55 : 25,
      cues: [
        "Start on your back. Knees bent, both feet flat on the floor.",
        "Keep your hips down until I say to lift.",
      ],
    });
  }

  if (ctx.bridgeStage === "ready_to_lift") {
    if (lifted) {
      ctx.bridgeStage = "holding";
      ctx.holdGoodCount = 0;
      const match = matchHoldTemplate(m);
      return result({
        status: match.issues.length ? "issue" : "hold",
        stage: "holding",
        stageChanged: true,
        formMatch: match.score,
        issues: match.issues,
        cues: match.issues.length
          ? match.cues
          : ["Hips are up. Hold and keep squeezing your glutes."],
      });
    }

    if (!setup && !lifted) {
      ctx.bridgeStage = "find_setup";
      ctx.setupCount = 0;
      return result({
        status: "ready",
        stage: "find_setup",
        stageChanged: true,
        formMatch: 20,
        cues: [
          "Find the start again. Back flat, knees bent, both feet on the floor.",
        ],
      });
    }

    return result({
      status: "ready",
      stage: "ready_to_lift",
      formMatch: 60,
      cues: [
        "Squeeze your glutes and press your hips toward the ceiling.",
        "Keep both feet flat as you lift.",
      ],
    });
  }

  if (ctx.bridgeStage === "holding") {
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
          formMatch: setup ? 65 : 30,
          cues: setup
            ? ["Hips are down. Squeeze and lift again when ready."]
            : ["Reset. Back flat, knees bent, then lift again."],
        });
      }
      return result({
        status: "ready",
        stage: "holding",
        formMatch: 40,
        cues: ["Hips dropped. Lift back up, or reset with both feet on the floor."],
      });
    }

    ctx.downCount = 0;
    const match = matchHoldTemplate(m);

    if (match.issues.length) {
      ctx.holdGoodCount = Math.max(0, ctx.holdGoodCount - 3);
      return result({
        status: "issue",
        stage: "holding",
        formMatch: match.score,
        issues: match.issues,
        cues: match.cues,
      });
    }

    ctx.holdGoodCount += 1;
    if (ctx.holdGoodCount >= HOLD_GOOD_FRAMES) {
      ctx.bridgeStage = "ready_to_lower";
      return result({
        status: "hold",
        stage: "ready_to_lower",
        stageChanged: true,
        formMatch: match.score,
        cues: [
          "Great form on that hold. Now lower your hips slowly with control.",
        ],
      });
    }

    return result({
      status: "hold",
      stage: "holding",
      formMatch: match.score,
      cues: match.cues,
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
          formMatch: 80,
          cues: [
            "Good rep. When ready, squeeze and lift again.",
          ],
        });
      }
    } else {
      ctx.downCount = 0;
      const match = matchHoldTemplate(m);
      if (match.issues.length) {
        return result({
          status: "issue",
          stage: "ready_to_lower",
          formMatch: match.score,
          issues: match.issues,
          cues: match.cues,
        });
      }
    }

    return result({
      status: lifted ? "hold" : "ready",
      stage: "ready_to_lower",
      formMatch: lifted ? 85 : 70,
      cues: ["Lower your hips slowly until your back is flat again."],
    });
  }

  ctx.bridgeStage = "find_setup";
  return result({
    status: "ready",
    stage: "find_setup",
    stageChanged: true,
    formMatch: 0,
    cues: ["Lie on your back with knees bent to begin."],
  });
}

export function resetBridgeCoach(ctx) {
  ctx.bridgeStage = "find_setup";
  ctx.setupCount = 0;
  ctx.holdGoodCount = 0;
  ctx.downCount = 0;
  ctx.kneeSamples = [];
  ctx.lastGoodSpeakAt = 0;
}
