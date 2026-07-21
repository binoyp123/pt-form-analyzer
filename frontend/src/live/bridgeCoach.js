/**
 * Guided glute-bridge live coach: setup → lift → hold → lower → repeat.
 *
 * Tuned for webcam 2D pose (small apparent hip lift) with hysteresis so
 * stages do not flicker ("find start again" loops).
 */

import { calcAngle, lm, mid } from "./geometry.js";

// Lift is often tiny in side/angled phone video — keep recognition loose.
const LIFT_ENTER = 0.012; // become "up"
const LIFT_EXIT = 0.006; // drop back "down" (hysteresis)
const LIFT_SETUP_MAX = 0.018; // still counting as hips-down setup

const SETUP_FRAMES = 8;
const HOLD_GOOD_FRAMES = 28;
const DOWN_FRAMES = 14;

const HOLD_TEMPLATE = {
  minLift: LIFT_ENTER,
  kneeMin: 30,
  kneeMax: 140,
  maxKneeAsymmetry: 30,
  straightLeg: 150,
  maxAnkleYDiff: 0.12,
  maxShoulderDiff: 0.12,
  maxXAlign: 0.28,
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

  return {
    lA,
    rA,
    midS,
    midH,
    midK,
    lKnee,
    rKnee,
    avgKnee: (lKnee + rKnee) / 2,
    lift: midS.y - midH.y,
    ankleYDiff: Math.abs(lA.y - rA.y),
    kneeAsym: Math.abs(lKnee - rKnee),
    shoulderDiff: Math.abs(lS.y - rS.y),
    xAlign: Math.abs(midS.x - midH.x),
  };
}

/** Forgiving start: on back-ish, knees bent, hips not clearly up. */
function isSetup(m) {
  if (!m) return false;
  if (m.lift > LIFT_SETUP_MAX) return false;
  // Wide knee band for camera angle differences
  if (!(m.avgKnee >= 25 && m.avgKnee <= 145)) return false;
  // Don't require perfect foot plant in setup
  if (m.lKnee > 160 && m.rKnee > 160) return false;
  return true;
}

function isLifted(m, wasLifted) {
  if (!m) return false;
  const need = wasLifted ? LIFT_EXIT : LIFT_ENTER;
  if (m.lift < need) return false;
  // Knees usually above hips in a bridge; allow slack for weird angles
  if (m.midK.y > m.midH.y + 0.06) return false;
  if (!(m.avgKnee >= 20 && m.avgKnee <= 155)) return false;
  return true;
}

function matchHoldTemplate(m) {
  const issues = [];
  const cues = [];
  let score = 100;
  if (!m) return { score: 0, issues, cues };

  const T = HOLD_TEMPLATE;

  if (m.lKnee >= T.straightLeg || m.rKnee >= T.straightLeg) {
    const side = m.lKnee >= m.rKnee ? "left" : "right";
    issues.push("leg_raised");
    cues.push(
      `Plant your ${side} foot back on the floor. Both feet stay down.`
    );
    score -= 40;
  } else if (m.ankleYDiff > T.maxAnkleYDiff) {
    const side = m.lA.y < m.rA.y ? "left" : "right";
    issues.push("foot_lifted");
    cues.push(`Keep your ${side} foot flat on the floor.`);
    score -= 28;
  }

  if (m.kneeAsym > T.maxKneeAsymmetry && !issues.includes("leg_raised")) {
    issues.push("knee_asym");
    cues.push("Bend both knees about the same amount.");
    score -= 15;
  }

  if (m.lift < T.minLift + 0.01) {
    issues.push("hip_height");
    cues.push("Lift your hips a bit higher and squeeze your glutes.");
    score -= 18;
  }

  if (m.shoulderDiff > T.maxShoulderDiff) {
    issues.push("shoulder_level");
    cues.push("Keep both shoulders pressed evenly into the floor.");
    score -= 10;
  }

  if (m.lift >= T.minLift && m.xAlign > T.maxXAlign) {
    issues.push("alignment");
    cues.push("Keep your hips centered. Avoid drifting sideways.");
    score -= 12;
  }

  score = Math.max(0, Math.min(100, Math.round(score)));
  if (!issues.length) {
    cues.push("Good form. Keep holding.");
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
  ctx.lostSetupCount = ctx.lostSetupCount || 0;
  ctx.wasLifted = Boolean(ctx.wasLifted);

  const m = metrics(landmarks);
  const setup = isSetup(m);
  const lifted = isLifted(m, ctx.wasLifted);
  if (lifted) ctx.wasLifted = true;
  if (!lifted && m && m.lift < LIFT_EXIT) ctx.wasLifted = false;

  if (!m) {
    return result({
      status: "ready",
      stage: ctx.bridgeStage,
      cues: ["Step back so your full body is visible on camera."],
      formMatch: 0,
    });
  }

  // —— Step 1: lock a stable start ——
  if (ctx.bridgeStage === "find_setup") {
    if (setup) ctx.setupCount += 1;
    else ctx.setupCount = Math.max(0, ctx.setupCount - 1);

    if (ctx.setupCount >= SETUP_FRAMES) {
      ctx.bridgeStage = "ready_to_lift";
      ctx.lostSetupCount = 0;
      return result({
        status: "ready",
        stage: "ready_to_lift",
        stageChanged: true,
        formMatch: 75,
        cues: [
          "Good start. Now squeeze your glutes and lift your hips.",
        ],
      });
    }

    // If they already lifted past setup, jump straight into hold.
    if (lifted) {
      ctx.bridgeStage = "holding";
      ctx.holdGoodCount = 0;
      ctx.downCount = 0;
      const match = matchHoldTemplate(m);
      return result({
        status: match.issues.length ? "issue" : "hold",
        stage: "holding",
        stageChanged: true,
        formMatch: match.score,
        issues: match.issues,
        cues: match.issues.length
          ? match.cues
          : ["Hips are up. Hold and keep squeezing."],
      });
    }

    return result({
      status: "ready",
      stage: "find_setup",
      formMatch: setup ? 60 : 30,
      cues: [
        "Lie on your back with knees bent and both feet on the floor.",
      ],
    });
  }

  // —— Step 2: wait for lift (do NOT bounce back to setup easily) ——
  if (ctx.bridgeStage === "ready_to_lift") {
    if (lifted) {
      ctx.bridgeStage = "holding";
      ctx.holdGoodCount = 0;
      ctx.downCount = 0;
      ctx.lostSetupCount = 0;
      const match = matchHoldTemplate(m);
      return result({
        status: match.issues.length ? "issue" : "hold",
        stage: "holding",
        stageChanged: true,
        formMatch: Math.max(match.score, 70),
        issues: match.issues,
        cues: match.issues.length
          ? match.cues
          : ["Nice. Hips are up. Hold here."],
      });
    }

    // Only reset if clearly gone for many frames (not a flicker).
    if (!setup && !lifted) {
      ctx.lostSetupCount += 1;
      if (ctx.lostSetupCount >= 45) {
        ctx.bridgeStage = "find_setup";
        ctx.setupCount = 0;
        ctx.lostSetupCount = 0;
        return result({
          status: "ready",
          stage: "find_setup",
          stageChanged: true,
          formMatch: 25,
          cues: ["I lost your start position. Knees bent, feet flat again."],
        });
      }
    } else {
      ctx.lostSetupCount = 0;
    }

    return result({
      status: "ready",
      stage: "ready_to_lift",
      formMatch: 65,
      cues: [
        "Lift your hips toward the ceiling. Keep both feet on the floor.",
      ],
    });
  }

  // —— Step 3: hold + template corrections ——
  if (ctx.bridgeStage === "holding") {
    if (!lifted) {
      ctx.downCount += 1;
      if (ctx.downCount >= DOWN_FRAMES) {
        ctx.bridgeStage = "ready_to_lift";
        ctx.holdGoodCount = 0;
        ctx.downCount = 0;
        return result({
          status: "ready",
          stage: "ready_to_lift",
          stageChanged: true,
          formMatch: 55,
          cues: ["Hips are down. Lift again when you are ready."],
        });
      }
      return result({
        status: "ready",
        stage: "holding",
        formMatch: 45,
        cues: ["Keep those hips up."],
      });
    }

    ctx.downCount = 0;
    const match = matchHoldTemplate(m);

    // During hold, only treat major mistakes as red (leg up / foot lifted).
    const hardIssues = match.issues.filter((i) =>
      ["leg_raised", "foot_lifted"].includes(i)
    );
    if (hardIssues.length) {
      ctx.holdGoodCount = Math.max(0, ctx.holdGoodCount - 2);
      return result({
        status: "issue",
        stage: "holding",
        formMatch: match.score,
        issues: hardIssues,
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
        formMatch: Math.max(match.score, 85),
        cues: ["Good rep hold. Now lower your hips slowly."],
      });
    }

    // Soft tips (hip a bit low, etc.) stay green so we don't nag.
    const softTip =
      match.issues.includes("hip_height") && match.cues[0]
        ? match.cues[0]
        : "Good form. Keep holding.";

    return result({
      status: "hold",
      stage: "holding",
      formMatch: Math.max(match.score, 75),
      cues: [softTip],
    });
  }

  // —— Step 4: lower ——
  if (ctx.bridgeStage === "ready_to_lower") {
    if (!lifted) {
      ctx.downCount += 1;
      if (ctx.downCount >= DOWN_FRAMES) {
        ctx.bridgeStage = "ready_to_lift";
        ctx.holdGoodCount = 0;
        ctx.downCount = 0;
        return result({
          status: "ready",
          stage: "ready_to_lift",
          stageChanged: true,
          formMatch: 85,
          cues: ["Nice. That counted. Lift again when ready."],
        });
      }
    } else {
      ctx.downCount = 0;
      const match = matchHoldTemplate(m);
      const hard = match.issues.filter((i) =>
        ["leg_raised", "foot_lifted"].includes(i)
      );
      if (hard.length) {
        return result({
          status: "issue",
          stage: "ready_to_lower",
          formMatch: match.score,
          issues: hard,
          cues: match.cues,
        });
      }
    }

    return result({
      status: lifted ? "hold" : "ready",
      stage: "ready_to_lower",
      formMatch: lifted ? 80 : 70,
      cues: ["Lower slowly until your back is flat."],
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
  ctx.lostSetupCount = 0;
  ctx.wasLifted = false;
  ctx.kneeSamples = [];
}
