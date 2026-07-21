/**
 * Guided cat-cow coach: tabletop → cat → cow → keep flowing.
 */

import { lm, mid, visible } from "./geometry.js";

const SETUP_FRAMES = 8;
const PHASE_FRAMES = 14;
const LOST_FRAMES = 40;

const STAGE_LABEL = {
  find_setup: "Step 1 · Tabletop",
  ready_cat: "Step 2 · Cat",
  ready_cow: "Step 3 · Cow",
  flowing: "Step 4 · Keep flowing",
};

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

function isTabletop(landmarks, metric) {
  if (metric == null) return false;
  const lK = lm(landmarks, "left_knee");
  const rK = lm(landmarks, "right_knee");
  const lW = lm(landmarks, "left_wrist");
  const rW = lm(landmarks, "right_wrist");
  // Need supporting limbs roughly visible
  return Boolean(lK && rK && (lW || rW));
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

export function evaluateCatCowCoach(landmarks, ctx) {
  if (!ctx.catStage) ctx.catStage = "find_setup";
  ctx.catSetupCount = ctx.catSetupCount || 0;
  ctx.catLost = ctx.catLost || 0;
  ctx.catPhaseCount = ctx.catPhaseCount || 0;
  ctx.catBaseline = ctx.catBaseline ?? null;
  ctx.catSeenCat = Boolean(ctx.catSeenCat);
  ctx.catSeenCow = Boolean(ctx.catSeenCow);
  ctx.catSeries = ctx.catSeries || [];

  const m = spineMetric(landmarks);
  const tabletop = isTabletop(landmarks, m);

  if (m != null) {
    ctx.catSeries.push(m);
    if (ctx.catSeries.length > 50) ctx.catSeries.shift();
  }

  if (m == null) {
    return result({
      status: "ready",
      stage: ctx.catStage,
      formMatch: 0,
      cues: ["Move so I can see your shoulders and hips from the side."],
    });
  }

  if (ctx.catBaseline == null && tabletop) {
    ctx.catBaseline = m;
  }
  const base = ctx.catBaseline ?? m;
  const delta = m - base;
  // Cat: round up (hips higher relative / metric depends on camera).
  // Using relative change over recent window is more reliable.
  const recent = ctx.catSeries.slice(-12);
  const span =
    recent.length >= 4 ? Math.max(...recent) - Math.min(...recent) : 0;
  const rising =
    recent.length >= 4 &&
    recent[recent.length - 1] > recent[0] + 0.008;
  const falling =
    recent.length >= 4 &&
    recent[recent.length - 1] < recent[0] - 0.008;

  if (ctx.catStage === "find_setup") {
    if (tabletop) ctx.catSetupCount += 1;
    else ctx.catSetupCount = Math.max(0, ctx.catSetupCount - 1);

    if (ctx.catSetupCount >= SETUP_FRAMES) {
      ctx.catStage = "ready_cat";
      ctx.catPhaseCount = 0;
      ctx.catBaseline = m;
      return result({
        status: "ready",
        stage: "ready_cat",
        stageChanged: true,
        formMatch: 70,
        cues: [
          "Good. Now round your back up into cat. Tuck your chin gently.",
        ],
      });
    }

    return result({
      status: "ready",
      stage: "find_setup",
      formMatch: tabletop ? 55 : 25,
      cues: ["Start on hands and knees in a neutral spine."],
    });
  }

  // Shared: do not bounce to setup unless clearly lost for a while
  function maybeLost() {
    if (!tabletop && m == null) {
      ctx.catLost += 1;
    } else if (!tabletop) {
      ctx.catLost += 1;
    } else {
      ctx.catLost = 0;
    }
    if (ctx.catLost >= LOST_FRAMES) {
      ctx.catStage = "find_setup";
      ctx.catSetupCount = 0;
      ctx.catLost = 0;
      ctx.catSeenCat = false;
      ctx.catSeenCow = false;
      return result({
        status: "ready",
        stage: "find_setup",
        stageChanged: true,
        formMatch: 20,
        cues: ["I lost your position. Get back on hands and knees."],
      });
    }
    return null;
  }

  if (ctx.catStage === "ready_cat") {
    const lost = maybeLost();
    if (lost) return lost;

    // Detect cat-ish motion: enough span + direction, or metric moved from baseline
    const catLike = span > 0.012 && (falling || delta < -0.01 || rising);
    // Prefer absolute deviation from baseline
    const moved = Math.abs(delta) > 0.012 || span > 0.015;
    if (moved) {
      ctx.catPhaseCount += 1;
    } else {
      ctx.catPhaseCount = Math.max(0, ctx.catPhaseCount - 1);
    }

    if (ctx.catPhaseCount >= PHASE_FRAMES || (catLike && span > 0.02)) {
      ctx.catSeenCat = true;
      ctx.catStage = "ready_cow";
      ctx.catPhaseCount = 0;
      return result({
        status: "hold",
        stage: "ready_cow",
        stageChanged: true,
        formMatch: 80,
        cues: [
          "Nice cat. Now drop your belly and lift your chest into cow.",
        ],
      });
    }

    return result({
      status: "ready",
      stage: "ready_cat",
      formMatch: 65,
      cues: ["Round your spine up. Push the floor away and tuck your chin."],
    });
  }

  if (ctx.catStage === "ready_cow") {
    const lost = maybeLost();
    if (lost) return lost;

    const moved = Math.abs(delta) > 0.01 || span > 0.015;
    if (moved) ctx.catPhaseCount += 1;
    else ctx.catPhaseCount = Math.max(0, ctx.catPhaseCount - 1);

    if (ctx.catPhaseCount >= PHASE_FRAMES || span > 0.02) {
      ctx.catSeenCow = true;
      ctx.catStage = "flowing";
      ctx.catPhaseCount = 0;
      return result({
        status: "hold",
        stage: "flowing",
        stageChanged: true,
        formMatch: 85,
        cues: [
          "Good. Keep alternating slow cat and cow rounds.",
        ],
      });
    }

    return result({
      status: "ready",
      stage: "ready_cow",
      formMatch: 70,
      cues: ["Drop your belly, lift your chest slightly. Look forward gently."],
    });
  }

  if (ctx.catStage === "flowing") {
    const lost = maybeLost();
    if (lost) return lost;

    if (span < 0.01) {
      return result({
        status: "issue",
        stage: "flowing",
        formMatch: 45,
        issues: ["amplitude"],
        cues: [
          "Make the motion bigger. Round up more, then drop your chest more.",
        ],
      });
    }

    const recentDelta = Math.abs(
      recent[recent.length - 1] - recent[Math.max(0, recent.length - 6)]
    );
    if (recentDelta < 0.004 && recent.length >= 8) {
      return result({
        status: "issue",
        stage: "flowing",
        formMatch: 55,
        issues: ["rhythm"],
        cues: ["Keep moving. Do not freeze in the middle."],
      });
    }

    return result({
      status: "hold",
      stage: "flowing",
      formMatch: 88,
      cues: ["Good rhythm. Keep slow cat and cow rounds."],
    });
  }

  ctx.catStage = "find_setup";
  return result({
    status: "ready",
    stage: "find_setup",
    stageChanged: true,
    formMatch: 0,
    cues: ["Start on hands and knees."],
  });
}

export function resetCatCowCoach(ctx) {
  ctx.catStage = "find_setup";
  ctx.catSetupCount = 0;
  ctx.catLost = 0;
  ctx.catPhaseCount = 0;
  ctx.catBaseline = null;
  ctx.catSeenCat = false;
  ctx.catSeenCow = false;
  ctx.catSeries = [];
}
