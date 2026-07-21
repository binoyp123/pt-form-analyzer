/**
 * Live coaching entrypoint. Each exercise uses a stage-guided coach.
 * Video upload analysis still uses the Python FastAPI evaluators.
 */

import { evaluateBridgeCoach, resetBridgeCoach } from "./bridgeCoach.js";
import { evaluateBirdDogCoach, resetBirdDogCoach } from "./birdDogCoach.js";
import { evaluateCatCowCoach, resetCatCowCoach } from "./catCowCoach.js";

/**
 * @returns {{
 *   status: 'ready'|'hold'|'issue',
 *   cues: string[],
 *   issues: string[],
 *   inHold: boolean,
 *   stage?: string,
 *   stageLabel?: string,
 *   stageChanged?: boolean,
 *   formMatch?: number|null
 * }}
 */
export function evaluateLiveFrame(exerciseId, landmarks, ctx = {}) {
  if (!landmarks?.length) {
    return {
      status: "ready",
      cues: ["Step back so your full body is in the camera frame."],
      issues: [],
      inHold: false,
      formMatch: 0,
    };
  }

  if (exerciseId === "bird_dog") return evaluateBirdDogCoach(landmarks, ctx);
  if (exerciseId === "bridge") return evaluateBridgeCoach(landmarks, ctx);
  if (exerciseId === "cat_cow") return evaluateCatCowCoach(landmarks, ctx);

  return {
    status: "ready",
    cues: ["Pick bird dog, bridge, or cat-cow."],
    issues: [],
    inHold: false,
    formMatch: 0,
  };
}

export function createLiveContext() {
  const ctx = {};
  resetBridgeCoach(ctx);
  resetBirdDogCoach(ctx);
  resetCatCowCoach(ctx);
  return ctx;
}
