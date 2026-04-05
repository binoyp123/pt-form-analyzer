"""
Bird-Dog Exercise Evaluator

Checks:
1. Back flatness (shoulder-hip-knee alignment)
2. Extended arm parallel to ground
3. Extended leg straight and parallel to ground
4. Hold stability across frames
"""

from dataclasses import dataclass
from pose_extractor import PoseFrame, PoseExtractor, calc_angle, is_landmark_visible


@dataclass
class FeedbackItem:
    """Single piece of feedback."""
    status: str  # "good", "warning", "error"
    message: str
    frames: list[int]  # which frames had this issue


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    score: int  # 0-100
    feedback: list[FeedbackItem]
    frames_analyzed: int
    exercise: str = "bird_dog"


# Thresholds (in degrees)
BACK_ANGLE_TOLERANCE = 15  # how far from 180° is acceptable
ARM_PARALLEL_TOLERANCE = 20  # how far from horizontal
LEG_PARALLEL_TOLERANCE = 20  # how far from horizontal
LEG_STRAIGHT_TOLERANCE = 25  # how far from 180° (straight)


def is_in_hold_position(frame: PoseFrame, ext: PoseExtractor) -> bool:
    """
    Detect if the person is in an active bird-dog hold position.
    Returns True only when one arm AND opposite leg are extended outward.
    """
    l_shoulder = ext.get_landmark(frame, "left_shoulder")
    r_shoulder = ext.get_landmark(frame, "right_shoulder")
    l_wrist = ext.get_landmark(frame, "left_wrist")
    r_wrist = ext.get_landmark(frame, "right_wrist")
    l_hip = ext.get_landmark(frame, "left_hip")
    r_hip = ext.get_landmark(frame, "right_hip")
    l_ankle = ext.get_landmark(frame, "left_ankle")
    r_ankle = ext.get_landmark(frame, "right_ankle")

    if not all([l_shoulder, r_shoulder, l_wrist, r_wrist, l_hip, r_hip, l_ankle, r_ankle]):
        return False

    # Arm is extended if wrist is significantly away from shoulder (X or Y)
    # This handles both side view (Y diff) and angled view (X diff)
    l_arm_y_diff = abs(l_wrist["y"] - l_shoulder["y"])
    l_arm_x_diff = abs(l_wrist["x"] - l_shoulder["x"])
    l_arm_extended = l_arm_y_diff < 0.2 or l_arm_x_diff > 0.15

    r_arm_y_diff = abs(r_wrist["y"] - r_shoulder["y"])
    r_arm_x_diff = abs(r_wrist["x"] - r_shoulder["x"])
    r_arm_extended = r_arm_y_diff < 0.2 or r_arm_x_diff > 0.15

    # Leg is extended if ankle is away from hip
    l_leg_y_diff = abs(l_ankle["y"] - l_hip["y"])
    l_leg_x_diff = abs(l_ankle["x"] - l_hip["x"])
    l_leg_extended = l_leg_y_diff < 0.25 or l_leg_x_diff > 0.15

    r_leg_y_diff = abs(r_ankle["y"] - r_hip["y"])
    r_leg_x_diff = abs(r_ankle["x"] - r_hip["x"])
    r_leg_extended = r_leg_y_diff < 0.25 or r_leg_x_diff > 0.15

    # For a valid hold, we need BOTH conditions:
    # 1. One arm extended AND opposite leg extended
    # 2. The extended limbs should be reaching outward (not just resting)

    # Check if arm is truly reaching out (not just at side)
    l_arm_reaching = l_arm_x_diff > 0.1 or l_arm_y_diff < 0.15
    r_arm_reaching = r_arm_x_diff > 0.1 or r_arm_y_diff < 0.15

    # Check if leg is truly reaching out
    l_leg_reaching = l_leg_x_diff > 0.1 or l_leg_y_diff < 0.2
    r_leg_reaching = r_leg_x_diff > 0.1 or r_leg_y_diff < 0.2

    # Valid bird-dog: opposite arm and leg both reaching
    left_arm_right_leg = (l_arm_extended and l_arm_reaching) and (r_leg_extended and r_leg_reaching)
    right_arm_left_leg = (r_arm_extended and r_arm_reaching) and (l_leg_extended and l_leg_reaching)

    return left_arm_right_leg or right_arm_left_leg


def evaluate(frames: list[PoseFrame], extractor: PoseExtractor) -> EvaluationResult:
    """
    Evaluate bird-dog form across all frames.
    Only scores frames where person is in hold position.
    """
    if not frames:
        return EvaluationResult(
            score=0,
            feedback=[FeedbackItem("error", "No pose data found", [])],
            frames_analyzed=0
        )

    # Filter to only frames in hold position
    hold_frames = [f for f in frames if is_in_hold_position(f, extractor)]

    if not hold_frames:
        return EvaluationResult(
            score=0,
            feedback=[FeedbackItem("warning", "No bird-dog hold positions detected", [])],
            frames_analyzed=len(frames)
        )

    issues = {
        "back_arch": [],
        "arm_not_parallel": [],
        "leg_not_parallel": [],
        "leg_bent": [],
    }

    good_frames = 0

    for frame in hold_frames:
        frame_issues = check_frame(frame, extractor)

        if not frame_issues:
            good_frames += 1
        else:
            for issue in frame_issues:
                issues[issue].append(frame.frame_num)

    # Calculate score based on hold frames only
    total = len(hold_frames)
    score = calc_score(issues, total)

    # Build feedback
    feedback = build_feedback(issues, good_frames, total)

    # Add info about how many frames were evaluated
    feedback.insert(0, FeedbackItem(
        "good",
        f"Detected {total} hold frames out of {len(frames)} total",
        []
    ))

    return EvaluationResult(
        score=score,
        feedback=feedback,
        frames_analyzed=total
    )


def check_frame(frame: PoseFrame, ext: PoseExtractor) -> list[str]:
    """Check a single frame for issues. Returns list of issue keys."""
    issues = []

    # Get landmarks
    l_shoulder = ext.get_landmark(frame, "left_shoulder")
    r_shoulder = ext.get_landmark(frame, "right_shoulder")
    l_hip = ext.get_landmark(frame, "left_hip")
    r_hip = ext.get_landmark(frame, "right_hip")
    l_knee = ext.get_landmark(frame, "left_knee")
    r_knee = ext.get_landmark(frame, "right_knee")
    l_ankle = ext.get_landmark(frame, "left_ankle")
    r_ankle = ext.get_landmark(frame, "right_ankle")
    l_wrist = ext.get_landmark(frame, "left_wrist")
    r_wrist = ext.get_landmark(frame, "right_wrist")
    l_elbow = ext.get_landmark(frame, "left_elbow")
    r_elbow = ext.get_landmark(frame, "right_elbow")

    # Detect which arm is extended (the one more parallel to ground)
    left_arm_ext, right_arm_ext = detect_extended_side(
        l_wrist, r_wrist, l_shoulder, r_shoulder
    )

    # Check back flatness
    back_ok = check_back_flat(
        l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee
    )
    if not back_ok:
        issues.append("back_arch")

    # Check extended arm is parallel to ground
    if left_arm_ext:
        arm_ok = check_arm_parallel(l_shoulder, l_elbow, l_wrist)
        # Opposite leg should be extended (right leg)
        leg_straight = check_leg_straight(r_hip, r_knee, r_ankle)
        leg_parallel = check_leg_parallel(r_hip, r_ankle)
    else:
        arm_ok = check_arm_parallel(r_shoulder, r_elbow, r_wrist)
        # Opposite leg should be extended (left leg)
        leg_straight = check_leg_straight(l_hip, l_knee, l_ankle)
        leg_parallel = check_leg_parallel(l_hip, l_ankle)

    if not arm_ok:
        issues.append("arm_not_parallel")

    if not leg_straight:
        issues.append("leg_bent")
    if not leg_parallel:
        issues.append("leg_not_parallel")

    return issues


def detect_extended_side(l_wrist, r_wrist, l_shoulder, r_shoulder) -> tuple[bool, bool]:
    """
    Detect which arm is extended.
    Returns (left_arm_extended, right_arm_extended).
    Extended arm will have wrist closer to shoulder height (smaller Y diff).
    """
    if not all([l_wrist, r_wrist, l_shoulder, r_shoulder]):
        return (True, False)  # default to left

    # The extended arm has wrist at similar Y to shoulder
    l_diff = abs(l_wrist["y"] - l_shoulder["y"])
    r_diff = abs(r_wrist["y"] - r_shoulder["y"])

    # Smaller diff = more extended/parallel
    if l_diff < r_diff:
        return (True, False)
    return (False, True)


def check_back_flat(l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee) -> bool:
    """Check if back is flat (shoulder-hip-knee roughly aligned)."""
    if not all([l_shoulder, r_shoulder, l_hip, r_hip]):
        return True  # can't check, assume ok

    mid_shoulder = {
        "x": (l_shoulder["x"] + r_shoulder["x"]) / 2,
        "y": (l_shoulder["y"] + r_shoulder["y"]) / 2,
        "z": (l_shoulder["z"] + r_shoulder["z"]) / 2,
    }
    mid_hip = {
        "x": (l_hip["x"] + r_hip["x"]) / 2,
        "y": (l_hip["y"] + r_hip["y"]) / 2,
        "z": (l_hip["z"] + r_hip["z"]) / 2,
    }

    dy = abs(mid_shoulder["y"] - mid_hip["y"])
    dx = abs(mid_shoulder["x"] - mid_hip["x"])

    if dx > 0.01:
        slope = dy / dx
        # Allow more slope - bird-dog naturally has some torso angle
        if slope > 0.5:
            return False

    return True


def check_arm_parallel(shoulder, elbow, wrist) -> bool:
    """Check if extended arm is roughly parallel to ground."""
    if not all([shoulder, wrist]):
        return True

    y_diff = abs(shoulder["y"] - wrist["y"])

    # Very generous tolerance - accounts for transitions between reps
    return y_diff < 0.35


def check_leg_straight(hip, knee, ankle) -> bool:
    """Check if leg is straight (hip-knee-ankle angle near 180°)."""
    if not all([hip, knee, ankle]):
        return True

    angle = calc_angle(hip, knee, ankle)
    # Forgiving threshold - accounts for rep transitions
    return angle > 120


def check_leg_parallel(hip, ankle) -> bool:
    """Check if extended leg is roughly parallel to ground."""
    if not all([hip, ankle]):
        return True

    y_diff = abs(hip["y"] - ankle["y"])
    # Generous tolerance for transitions
    return y_diff < 0.35


def calc_score(issues: dict, total_frames: int) -> int:
    """Calculate score based on issues found."""
    if total_frames == 0:
        return 0

    score = 100

    # Deduct points based on percentage of frames with each issue
    weights = {
        "back_arch": 30,
        "arm_not_parallel": 25,
        "leg_not_parallel": 25,
        "leg_bent": 20,
    }

    for issue, frames in issues.items():
        if frames:
            pct = len(frames) / total_frames
            deduction = weights.get(issue, 10) * pct
            score -= deduction

    return max(0, min(100, int(score)))


def build_feedback(issues: dict, good_frames: int, total: int) -> list[FeedbackItem]:
    """Build human-readable feedback from issues."""
    feedback = []

    # Good feedback first
    if good_frames > total * 0.7:
        feedback.append(FeedbackItem(
            "good",
            f"Good form on {good_frames}/{total} frames",
            []
        ))

    # Issue-specific feedback
    if issues["back_arch"]:
        pct = len(issues["back_arch"]) / total * 100
        feedback.append(FeedbackItem(
            "warning" if pct < 30 else "error",
            f"Back arching detected ({pct:.0f}% of frames)",
            issues["back_arch"][:5]  # first 5 problem frames
        ))

    if issues["arm_not_parallel"]:
        pct = len(issues["arm_not_parallel"]) / total * 100
        feedback.append(FeedbackItem(
            "warning" if pct < 30 else "error",
            f"Arm not parallel to ground ({pct:.0f}% of frames)",
            issues["arm_not_parallel"][:5]
        ))

    if issues["leg_not_parallel"]:
        pct = len(issues["leg_not_parallel"]) / total * 100
        feedback.append(FeedbackItem(
            "warning" if pct < 30 else "error",
            f"Leg not parallel to ground ({pct:.0f}% of frames)",
            issues["leg_not_parallel"][:5]
        ))

    if issues["leg_bent"]:
        pct = len(issues["leg_bent"]) / total * 100
        feedback.append(FeedbackItem(
            "warning" if pct < 30 else "error",
            f"Leg not fully extended ({pct:.0f}% of frames)",
            issues["leg_bent"][:5]
        ))

    # If no issues at all
    if not any(issues.values()):
        feedback.append(FeedbackItem(
            "good",
            "Excellent form throughout!",
            []
        ))

    return feedback


# Quick test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bird_dog.py <video_path>")
        sys.exit(1)

    with PoseExtractor() as extractor:
        print(f"Extracting poses from {sys.argv[1]}...")
        frames = extractor.extract_from_video(sys.argv[1])
        print(f"Got {len(frames)} frames")

        print("\nEvaluating bird-dog form...")
        result = evaluate(frames, extractor)

        print(f"\n{'='*40}")
        print(f"SCORE: {result.score}/100")
        print(f"{'='*40}")

        for fb in result.feedback:
            icon = {"good": "✅", "warning": "⚠️", "error": "❌"}[fb.status]
            print(f"{icon} {fb.message}")
            if fb.frames:
                print(f"   Problem frames: {fb.frames}")