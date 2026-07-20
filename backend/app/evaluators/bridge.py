"""
Glute bridge exercise evaluator.

Hold detection strategy (camera-angle robust):
- Person is supine: shoulders and hips are both in the lower portion of
  the frame (high normalized Y), while knees are above them (lower Y).
- Hips are elevated: mid_hip.y < mid_shoulder.y (closer to top of frame).
- Knees are bent: knees are above hips in Y, and the hip-knee-ankle angle
  is NOT near-straight (i.e. < 160°). We deliberately use a very wide
  acceptable range (25-140°) because 2D projection compresses the angle
  dramatically from a side view (~40°) vs front view (~90°).

Quality checks on hold frames:
- Hip height relative to shoulder-ankle line
- Knee angle consistency (relative to the video's own median angle)
- Shoulders level (small L/R Y difference)
- Lateral alignment (mid_hip.x near mid_shoulder.x)
"""

from __future__ import annotations

from pose_extractor import PoseFrame, PoseExtractor, calc_angle

try:
    from .common_types import EvaluationResult, FeedbackItem
except ImportError:
    from common_types import EvaluationResult, FeedbackItem

# Hold detection — intentionally wide to handle varied camera angles
KNEE_ANGLE_MIN = 25   # side view can compress to ~40°
KNEE_ANGLE_MAX = 140  # front view shows ~90°, allow headroom
HIP_LIFT_MIN = 0.005  # very small; side-view lift is often only 0.01

# Quality thresholds
SHOULDER_LEVEL_MAX = 0.09   # |L_shoulder.y - R_shoulder.y|
HIP_SHOULDER_X_ALIGN = 0.22  # |mid_shoulder.x - mid_hip.x|
KNEE_ANGLE_DEVIATION = 20    # degrees away from the video's own median


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
    mid_k = _mid(l_k, r_k)
    if not mid_s or not mid_h or not mid_k:
        return False

    # 1. Hips must be above shoulders in frame (lower Y = higher in image)
    if mid_h["y"] > mid_s["y"] - HIP_LIFT_MIN:
        return False

    # 2. Knees should be above hips (bent, not lying flat)
    if mid_k["y"] > mid_h["y"]:
        return False

    # 3. Knee angle must be bent (not standing straight ~180°)
    l_ang = calc_angle(l_h, l_k, l_a)
    r_ang = calc_angle(r_h, r_k, r_a)
    avg_ang = (l_ang + r_ang) / 2
    if not (KNEE_ANGLE_MIN <= avg_ang <= KNEE_ANGLE_MAX):
        return False

    return True


def _get_frame_metrics(frame: PoseFrame, ext: PoseExtractor) -> dict | None:
    """Extract all relevant measurements from a single frame."""
    l_s = ext.get_landmark(frame, "left_shoulder")
    r_s = ext.get_landmark(frame, "right_shoulder")
    l_h = ext.get_landmark(frame, "left_hip")
    r_h = ext.get_landmark(frame, "right_hip")
    l_k = ext.get_landmark(frame, "left_knee")
    r_k = ext.get_landmark(frame, "right_knee")
    l_a = ext.get_landmark(frame, "left_ankle")
    r_a = ext.get_landmark(frame, "right_ankle")

    if not all([l_s, r_s, l_h, r_h, l_k, r_k, l_a, r_a]):
        return None

    mid_s = _mid(l_s, r_s)
    mid_h = _mid(l_h, r_h)
    if not mid_s or not mid_h:
        return None

    return {
        "lift": mid_s["y"] - mid_h["y"],
        "l_knee": calc_angle(l_h, l_k, l_a),
        "r_knee": calc_angle(r_h, r_k, r_a),
        "shoulder_diff": abs(l_s["y"] - r_s["y"]),
        "x_align": abs(mid_s["x"] - mid_h["x"]),
    }


def _check_frame(metrics: dict, median_knee: float) -> list[str]:
    """Check a single frame for quality issues."""
    issues: list[str] = []

    if metrics["lift"] < HIP_LIFT_MIN:
        issues.append("hip_height")

    avg_knee = (metrics["l_knee"] + metrics["r_knee"]) / 2
    if abs(avg_knee - median_knee) > KNEE_ANGLE_DEVIATION:
        issues.append("knee_angle")

    if metrics["shoulder_diff"] > SHOULDER_LEVEL_MAX:
        issues.append("shoulder_level")

    if metrics["x_align"] > HIP_SHOULDER_X_ALIGN:
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
                    "No glute bridge hold detected — make sure hips are lifted "
                    "and knees are bent, with full body visible",
                    [],
                )
            ],
            len(frames),
            "bridge",
        )

    # Pre-compute metrics for all hold frames
    frame_metrics: list[tuple[PoseFrame, dict]] = []
    for f in hold_frames:
        m = _get_frame_metrics(f, extractor)
        if m is not None:
            frame_metrics.append((f, m))

    if not frame_metrics:
        return EvaluationResult(
            0,
            [FeedbackItem("warning", "Landmarks not visible in hold frames", [])],
            len(frames),
            "bridge",
        )

    # Compute median knee angle for this video (camera-relative baseline)
    all_knee_avgs = sorted(
        (m["l_knee"] + m["r_knee"]) / 2 for _, m in frame_metrics
    )
    median_knee = all_knee_avgs[len(all_knee_avgs) // 2]

    issues: dict[str, list[int]] = {
        "hip_height": [],
        "knee_angle": [],
        "shoulder_level": [],
        "alignment": [],
    }
    good = 0

    for f, m in frame_metrics:
        probs = _check_frame(m, median_knee)
        if not probs:
            good += 1
        else:
            for p in probs:
                issues[p].append(f.frame_num)

    total = len(frame_metrics)
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
        "hip_height": "Hips not high enough — drive through heels",
        "knee_angle": "Knee angle inconsistent — keep feet planted evenly",
        "shoulder_level": "Shoulders not level — keep both on the mat",
        "alignment": "Hips drifting sideways — squeeze glutes evenly",
    }

    for key, msg in labels.items():
        frs = issues[key]
        if frs:
            pct = len(frs) / total * 100
            st = "warning" if pct < 30 else "error"
            feedback.append(FeedbackItem(st, f"{msg} ({pct:.0f}% of hold frames)", frs[:5]))

    if score >= 85 and not any(len(v) > total * 0.15 for v in issues.values()):
        feedback.append(FeedbackItem("good", "Strong bridge mechanics overall", []))

    issue_frames = sorted({n for nums in issues.values() for n in nums})

    return EvaluationResult(
        score,
        feedback,
        total,
        "bridge",
        scored_frames=[f.frame_num for f, _ in frame_metrics],
        issue_frames=issue_frames,
    )


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
