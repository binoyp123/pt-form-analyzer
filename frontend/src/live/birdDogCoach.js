/**
 * Guided bird-dog coach: tabletop → extend → hold → return → repeat.
 * Forgiving hysteresis so stages do not flicker.
 */

import { calcAngle, lm, mid } from "./geometry.js";

const SETUP_FRAMES = 8;
const HOLD_GOOD_FRAMES = 28;
const LOST_FRAMES = 40;
const EXIT_HOLD_FRAMES = 12;

const STAGE_LABEL = {
  find_setup: "Step 1 · Tabletop",
  ready_to_extend: "Step 2 · Extend",
  holding: "Step 3 · Hold",
  ready_to_return: "Step 4 · Return",
};

function pts(landmarks) {
  const lS = lm(landmarks, "left_shoulder");
  const rS = lm(landmarks, "right_shoulder");
  const lW = lm(landmarks, "left_wrist");
  const rW = lm(landmarks, "right_wrist");
  const lH = lm(landmarks, "left_hip");
  const rH = lm(landmarks, "right_hip");
  const lK = lm(landmarks, "left_knee");
  const rK = lm(landmarks, "right_knee");
  const lA = lm(landmarks, "left_ankle");
  const rA = lm(landmarks, "right_ankle");
  if (![lS, rS, lW, rW, lH, rH, lK, rK, lA, rA].every(Boolean)) return null;
  return { lS, rS, lW, rW, lH, rH, lK, rK, lA, rA };
}

function looksFlat(p) {
  const ys = [p.lS, p.rS, p.lH, p.rH, p.lK, p.rK, p.lA, p.rA].map((x) => x.y);
  return Math.max(...ys) - Math.min(...ys) < 0.14;
}

/** Hands-and-knees-ish, not fully extended yet. */
function isTabletop(p) {
  if (!p || looksFlat(p)) return false;
  // Wrists roughly near shoulder height or below (supporting) — forgiving
  const midS = mid(p.lS, p.rS);
  const midH = mid(p.lH, p.rH);
  if (!midS || !midH) return false;
  // Not a clear opposite-limb hold yet
  return !isHold(p, false);
}

function reachScores(p) {
  const lArm =
    Math.max(Math.abs(p.lW.x - p.lS.x), 0.01) +
    (Math.abs(p.lW.y - p.lS.y) < 0.22 ? 0.12 : 0);
  const rArm =
    Math.max(Math.abs(p.rW.x - p.rS.x), 0.01) +
    (Math.abs(p.rW.y - p.rS.y) < 0.22 ? 0.12 : 0);
  const lLeg =
    Math.max(Math.abs(p.lA.x - p.lH.x), 0.01) +
    (Math.abs(p.lA.y - p.lH.y) < 0.28 ? 0.1 : 0);
  const rLeg =
    Math.max(Math.abs(p.rA.x - p.rH.x), 0.01) +
    (Math.abs(p.rA.y - p.rH.y) < 0.28 ? 0.1 : 0);
  return { lArm, rArm, lLeg, rLeg };
}

function isHold(p, wasHold) {
  if (!p) return false;
  const { lArm, rArm, lLeg, rLeg } = reachScores(p);
  const armNeed = wasHold ? 0.14 : 0.18;
  const legNeed = wasHold ? 0.14 : 0.18;
  const leftArmRightLeg = lArm > armNeed && rLeg > legNeed;
  const rightArmLeftLeg = rArm > armNeed && lLeg > legNeed;
  return leftArmRightLeg || rightArmLeftLeg;
}

function whichSide(p) {
  const { lArm, rArm, lLeg, rLeg } = reachScores(p);
  const leftArmRightLeg = lArm + rLeg;
  const rightArmLeftLeg = rArm + lLeg;
  return leftArmRightLeg >= rightArmLeftLeg
    ? { arm: "left", leg: "right", leftArm: true }
    : { arm: "right", leg: "left", leftArm: false };
}

function matchHoldTemplate(p) {
  const issues = [];
  const cues = [];
  let score = 100;
  if (!p) return { score: 0, issues, cues };

  const side = whichSide(p);
  const shoulder = side.leftArm ? p.lS : p.rS;
  const wrist = side.leftArm ? p.lW : p.rW;
  const hip = side.leftArm ? p.rH : p.lH;
  const knee = side.leftArm ? p.rK : p.lK;
  const ankle = side.leftArm ? p.rA : p.lA;

  const midS = mid(p.lS, p.rS);
  const midH = mid(p.lH, p.rH);
  if (midS && midH) {
    const dy = Math.abs(midS.y - midH.y);
    const dx = Math.abs(midS.x - midH.x);
    if (dx > 0.01 && dy / dx > 0.55) {
      issues.push("back_arch");
      cues.push("Keep your back flatter. Pull your belly in a little.");
      score -= 25;
    }
  }

  if (Math.abs(shoulder.y - wrist.y) >= 0.38) {
    issues.push("arm_not_parallel");
    cues.push(
      `Line your ${side.arm} arm up parallel with the floor.`
    );
    score -= 22;
  }

  if (Math.abs(hip.y - ankle.y) >= 0.38) {
    issues.push("leg_not_parallel");
    cues.push(
      `Lift your ${side.leg} leg until it is closer to parallel with the floor.`
    );
    score -= 22;
  }

  const legAngle = calcAngle(hip, knee, ankle);
  if (legAngle <= 115) {
    issues.push("leg_bent");
    cues.push(`Straighten your ${side.leg} knee so the leg reaches long.`);
    score -= 25;
  }

  score = Math.max(0, Math.min(100, Math.round(score)));
  // Soft tips only for red on major issues
  const hard = issues.filter((i) =>
    ["leg_bent", "back_arch", "arm_not_parallel", "leg_not_parallel"].includes(i)
  );
  if (!hard.length) {
    cues.length = 0;
    cues.push("Good form. Hold steady.");
  }

  return { score, issues: hard, cues: cues.slice(0, 2) };
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

export function evaluateBirdDogCoach(landmarks, ctx) {
  if (!ctx.birdStage) ctx.birdStage = "find_setup";
  ctx.birdSetupCount = ctx.birdSetupCount || 0;
  ctx.birdHoldGood = ctx.birdHoldGood || 0;
  ctx.birdLost = ctx.birdLost || 0;
  ctx.birdExitHold = ctx.birdExitHold || 0;
  ctx.birdWasHold = Boolean(ctx.birdWasHold);

  const p = pts(landmarks);
  const tabletop = isTabletop(p);
  const hold = isHold(p, ctx.birdWasHold);
  if (hold) ctx.birdWasHold = true;
  if (!hold) ctx.birdWasHold = false;

  if (!p) {
    return result({
      status: "ready",
      stage: ctx.birdStage,
      formMatch: 0,
      cues: ["Step into frame so your full body is visible."],
    });
  }

  if (looksFlat(p) && ctx.birdStage === "find_setup") {
    return result({
      status: "ready",
      stage: "find_setup",
      formMatch: 20,
      cues: ["Get onto your hands and knees first."],
    });
  }

  if (ctx.birdStage === "find_setup") {
    if (hold) {
      ctx.birdStage = "holding";
      ctx.birdHoldGood = 0;
      const match = matchHoldTemplate(p);
      return result({
        status: match.issues.length ? "issue" : "hold",
        stage: "holding",
        stageChanged: true,
        formMatch: match.score,
        issues: match.issues,
        cues: match.issues.length
          ? match.cues
          : ["Nice extension. Hold it steady."],
      });
    }

    if (tabletop) ctx.birdSetupCount += 1;
    else ctx.birdSetupCount = Math.max(0, ctx.birdSetupCount - 1);

    if (ctx.birdSetupCount >= SETUP_FRAMES) {
      ctx.birdStage = "ready_to_extend";
      ctx.birdLost = 0;
      return result({
        status: "ready",
        stage: "ready_to_extend",
        stageChanged: true,
        formMatch: 70,
        cues: [
          "Good tabletop. Reach one arm forward and the opposite leg back.",
        ],
      });
    }

    return result({
      status: "ready",
      stage: "find_setup",
      formMatch: tabletop ? 55 : 30,
      cues: ["Start on hands and knees with a flat back."],
    });
  }

  if (ctx.birdStage === "ready_to_extend") {
    if (hold) {
      ctx.birdStage = "holding";
      ctx.birdHoldGood = 0;
      ctx.birdLost = 0;
      const match = matchHoldTemplate(p);
      return result({
        status: match.issues.length ? "issue" : "hold",
        stage: "holding",
        stageChanged: true,
        formMatch: Math.max(match.score, 70),
        issues: match.issues,
        cues: match.issues.length
          ? match.cues
          : ["Good. You are in the hold. Keep balancing."],
      });
    }

    if (!tabletop && !hold) {
      ctx.birdLost += 1;
      if (ctx.birdLost >= LOST_FRAMES) {
        ctx.birdStage = "find_setup";
        ctx.birdSetupCount = 0;
        ctx.birdLost = 0;
        return result({
          status: "ready",
          stage: "find_setup",
          stageChanged: true,
          formMatch: 25,
          cues: ["Find hands and knees again, then extend."],
        });
      }
    } else {
      ctx.birdLost = 0;
    }

    return result({
      status: "ready",
      stage: "ready_to_extend",
      formMatch: 65,
      cues: [
        "Extend opposite arm and leg. Reach long, then hold.",
      ],
    });
  }

  if (ctx.birdStage === "holding") {
    if (!hold) {
      ctx.birdExitHold += 1;
      if (ctx.birdExitHold >= EXIT_HOLD_FRAMES) {
        ctx.birdStage = "ready_to_return";
        ctx.birdExitHold = 0;
        ctx.birdHoldGood = 0;
        return result({
          status: "ready",
          stage: "ready_to_return",
          stageChanged: true,
          formMatch: 60,
          cues: [
            "Bring the arm and leg back to tabletop, then switch sides.",
          ],
        });
      }
      return result({
        status: "ready",
        stage: "holding",
        formMatch: 50,
        cues: ["Keep the arm and leg extended, or return to tabletop to switch."],
      });
    }

    ctx.birdExitHold = 0;
    const match = matchHoldTemplate(p);
    if (match.issues.length) {
      ctx.birdHoldGood = Math.max(0, ctx.birdHoldGood - 2);
      return result({
        status: "issue",
        stage: "holding",
        formMatch: match.score,
        issues: match.issues,
        cues: match.cues,
      });
    }

    ctx.birdHoldGood += 1;
    if (ctx.birdHoldGood >= HOLD_GOOD_FRAMES) {
      ctx.birdStage = "ready_to_return";
      return result({
        status: "hold",
        stage: "ready_to_return",
        stageChanged: true,
        formMatch: Math.max(match.score, 85),
        cues: [
          "Solid hold. Return to tabletop, then switch to the other side.",
        ],
      });
    }

    return result({
      status: "hold",
      stage: "holding",
      formMatch: Math.max(match.score, 75),
      cues: match.cues,
    });
  }

  if (ctx.birdStage === "ready_to_return") {
    if (hold) {
      // Still extended or started the other side early
      const match = matchHoldTemplate(p);
      if (match.issues.length) {
        return result({
          status: "issue",
          stage: "ready_to_return",
          formMatch: match.score,
          issues: match.issues,
          cues: match.cues,
        });
      }
      ctx.birdStage = "holding";
      ctx.birdHoldGood = 0;
      return result({
        status: "hold",
        stage: "holding",
        stageChanged: true,
        formMatch: match.score,
        cues: ["Holding again. Keep your back steady."],
      });
    }

    if (tabletop) {
      ctx.birdStage = "ready_to_extend";
      ctx.birdLost = 0;
      return result({
        status: "ready",
        stage: "ready_to_extend",
        stageChanged: true,
        formMatch: 80,
        cues: [
          "Good. Now extend the other arm and opposite leg.",
        ],
      });
    }

    return result({
      status: "ready",
      stage: "ready_to_return",
      formMatch: 65,
      cues: ["Come back to hands and knees, then switch sides."],
    });
  }

  ctx.birdStage = "find_setup";
  return result({
    status: "ready",
    stage: "find_setup",
    stageChanged: true,
    formMatch: 0,
    cues: ["Start on hands and knees."],
  });
}

export function resetBirdDogCoach(ctx) {
  ctx.birdStage = "find_setup";
  ctx.birdSetupCount = 0;
  ctx.birdHoldGood = 0;
  ctx.birdLost = 0;
  ctx.birdExitHold = 0;
  ctx.birdWasHold = false;
}
