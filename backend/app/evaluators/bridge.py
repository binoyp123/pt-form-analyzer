"""
Glute bridge exercise evaluator.

Heuristics (tune with real videos):
- Hips elevated above shoulder line (MediaPipe Y increases downward)
- Knee flexion near ~90° at hip–knee–ankle
- Shoulders level and staying grounded (stable Y, small L/R difference)
- Hips stacked under shoulders laterally (mid_x alignment)
"""

from __future__ import annotations

from pose_extractor import PoseFrame, PoseExtractor, calc_angle

from .common_types import EvaluationResult, FeedbackItem

# Degrees
KNEE_ANGLE_MIN = 70
KNEE_ANGLE_MAX = 115
HIP_LIFT_MIN = 0.02  # normalized; min (shoulder_y - hip_y) when lifted
SHOULDER_LEVEL_MAX = 0.09  # |L_shoulder.y - R_shoulder.y|
HIP_SHOULDER_X_ALIGN = 0.18  # |mid_shoulder.x - mid_hip.x|


def _mid(lm1: dict, lm2: dict) -> dict | None:
    if not lm1 or not lm2:
        return None
    return {
        "x": (lm1["x"] + lm2["x"]) / 2,
        "y": (lm1["y"] + lm2["y"]) / 2,
        "z": (lm1["z"] + lm2["z"]) / 2,
    }


def is_in_bridge_hold(frame: PoseFrame, ext: PoseExtractor) -> bool:
    """True when pose looks like an active glute bridge (hips up, knees bent)."""
    l_s = ext.get_landmark(frame, "left_shoulder")
    r_s = ext.get_landmark(frame, "right_shoulder")
    l_h = ext.get_landmark(frame, "left_hip")
    r_h = ext.get_landmark(frame, "right_hip")
    l_k = ext.get_landmark(frame, "left_knee")
    r_k = ext.get_landmark(frame, "right_knee")
    l_a = ext.get_landmark(frame, "left_ankle")
    r_a = ext.get_landmark(frame, "right_ankle")

    if not all([l_s, r_s, l_h, r_h, l_k, r_k, l_a, r_a]):
        return False

    mid_s = _mid(l_s, r_s)
    mid_h = _mid(l_h, r_h)
    if not mid_s or not mid_h:
        return False

    # Hips must read as lifted (smaller Y than shoulders)
    if mid_h["y"] > mid_s["y"] - HIP_LIFT_MIN:
        return False

    l_ang = calc_angle(l_h, l_k, l_a)
    r_ang = calc_angle(r_h, r_k, r_a)
    if not (KNEE_ANGLE_MIN <= l_ang <= KNEE_ANGLE_MAX):
        return False
    if not (KNEE_ANGLE_MIN <= r_ang <= KNEE_ANGLE_MAX):
        return False

    return True


def _check_frame(frame: PoseFrame, ext: PoseExtractor) -> list[str]:
    issues: list[str] = []
    l_s = ext.get_landmark(frame, "left_shoulder")
    r_s = ext.get_landmark(frame, "right_shoulder")
    l_h = ext.get_landmark(frame, "left_hip")
    r_h = ext.get_landmark(frame, "right_hip")
    l_k = ext.get_landmark(frame, "left_knee")
    r_k = ext.get_landmark(frame, "right_knee")
    l_a = ext.get_landmark(frame, "left_ankle")
    r_a = ext.get_landmark(frame, "right_ankle")

    if not all([l_s, r_s, l_h, r_h, l_k, r_k, l_a, r_a]):
        return ["visibility"]

    mid_s = _mid(l_s, r_s)
    mid_h = _mid(l_h, r_h)
    if not mid_s or not mid_h:
        return ["visibility"]

    lift = mid_s["y"] - mid_h["y"]
    if lift < HIP_LIFT_MIN:
        issues.append("hip_height")

    l_ang = calc_angle(l_h, l_k, l_a)
    r_ang = calc_angle(r_h, r_k, r_a)
    if not (KNEE_ANGLE_MIN <= l_ang <= KNEE_ANGLE_MAX):
        issues.append("knee_angle")
    if not (KNEE_ANGLE_MIN <= r_ang <= KNEE_ANGLE_MAX):
        issues.append("knee_angle")

    if abs(l_s["y"] - r_s["y"]) > SHOULDER_LEVEL_MAX:
        issues.append("shoulder_level")

    if abs(mid_s["x"] - mid_h["x"]) > HIP_SHOULDER_X_ALIGN:
        issues.append("alignment")

    return issues


def _calc_score(issues: dict[str, list[int]], total: int) -> int:
    if total == 0:
        return 0
    score = 100.0
    weights = {
        "hip_height": 30,
        "knee_angle": 25,
        "shoulder_level": 20,
        "alignment": 25,
        "visibility": 15,
    }
    for key, frames in issues.items():
        if frames:
            pct = len(frames) / total
            score -= weights.get(key, 10) * pct
    return max(0, min(100, int(score)))


def evaluate(frames: list[PoseFrame], extractor: PoseExtractor) -> EvaluationResult:
    if not frames:
        return EvaluationResult(
            0,
            [FeedbackItem("error", "No pose data found", [])],
            0,
            "bridge",
        )

    hold_frames = [f for f in frames if is_in_bridge_hold(f, extractor)]
    if not hold_frames:
        return EvaluationResult(
            0,
            [
                FeedbackItem(
                    "warning",
                    "No glute bridge hold detected (hips up, knees ~90°)",
                    [],
                )
            ],
            len(frames),
            "bridge",
        )

    issues: dict[str, list[int]] = {
        "hip_height": [],
        "knee_angle": [],
        "shoulder_level": [],
        "alignment": [],
        "visibility": [],
    }
    good = 0

    for frame in hold_frames:
        probs = _check_frame(frame, extractor)
        if probs == ["visibility"]:
            issues["visibility"].append(frame.frame_num)
            continue
        if not probs:
            good += 1
            continue
        for p in probs:
            issues[p].append(frame.frame_num)

    total = len(hold_frames)
    score = _calc_score(issues, total)

    feedback: list[FeedbackItem] = [
        FeedbackItem(
            "good",
            f"Scored {total} bridge hold frames out of {len(frames)} total",
            [],
        )
    ]

    if good > total * 0.7:
        feedback.append(
            FeedbackItem("good", f"Solid form on {good}/{total} hold frames", [])
        )

    labels = {
        "hip_height": "Hips not high enough vs shoulders",
        "knee_angle": "Knee angle not near ~90° (check foot placement)",
        "shoulder_level": "Shoulders not level — roll or twist on mat",
        "alignment": "Hips drifting sideways vs shoulders",
        "visibility": "Landmarks not visible — stay in frame",
    }

    for key, msg in labels.items():
        frs = issues[key]
        if frs:
            pct = len(frs) / total * 100
            st = "warning" if pct < 30 else "error"
            feedback.append(FeedbackItem(st, f"{msg} ({pct:.0f}% of hold frames)", frs[:5]))

    if score >= 85 and not any(len(issues[k]) > total * 0.15 for k in issues if k != "visibility"):
        feedback.append(FeedbackItem("good", "Strong bridge mechanics overall", []))

    return EvaluationResult(score, feedback, total, "bridge")


if __name__ == "__main__":
    import sys

    from pose_extractor import PoseExtractor

    if len(sys.argv) < 2:
        print("Usage: python bridge.py <video_path>")
        sys.exit(1)

    with PoseExtractor() as ex:
        print(f"Extracting poses from {sys.argv[1]}...")
        frames = ex.extract_from_video(sys.argv[1])
        print(f"Got {len(frames)} frames with pose")
        r = evaluate(frames, ex)
        print(f"\nScore: {r.score}/100  (frames analyzed: {r.frames_analyzed})")
        for fb in r.feedback:
            print(f"  [{fb.status}] {fb.message}")
