"""
Cat-cow (dynamic spine) evaluator.

MediaPipe has no spine joints; we use shoulder vs hip geometry over time.
The metric tracks vertical separation (mid_hip.y - mid_shoulder.y); cat/cow should
produce clear oscillation. Thresholds are starting points for tuning with real video.
"""

from __future__ import annotations

from pose_extractor import PoseFrame, PoseExtractor

try:
    from .common_types import EvaluationResult, FeedbackItem
except ImportError:
    from common_types import EvaluationResult, FeedbackItem

SMOOTH_WINDOW = 7
MIN_AMPLITUDE = 0.025  # normalized; min peak-to-trough for "real" motion
MIN_CYCLES = 1.5  # need enough peaks+troughs to suggest alternating cat/cow
VISIBILITY_THRESHOLD = 0.5


def _mid(lm1: dict | None, lm2: dict | None) -> dict | None:
    if not lm1 or not lm2:
        return None
    return {
        "x": (lm1["x"] + lm2["x"]) / 2,
        "y": (lm1["y"] + lm2["y"]) / 2,
        "z": (lm1["z"] + lm2["z"]) / 2,
    }


def _visible(lm: dict | None) -> bool:
    if lm is None:
        return False
    return lm.get("visibility", 0) >= VISIBILITY_THRESHOLD


def spine_metric(frame: PoseFrame, ext: PoseExtractor) -> float | None:
    """Larger values => hips lower relative to shoulders (camera-dependent)."""
    l_s = ext.get_landmark(frame, "left_shoulder")
    r_s = ext.get_landmark(frame, "right_shoulder")
    l_h = ext.get_landmark(frame, "left_hip")
    r_h = ext.get_landmark(frame, "right_hip")

    if not all(_visible(x) for x in (l_s, r_s, l_h, r_h)):
        return None

    mid_s = _mid(l_s, r_s)
    mid_h = _mid(l_h, r_h)
    if not mid_s or not mid_h:
        return None

    return mid_h["y"] - mid_s["y"]


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values or window < 1:
        return []
    half = window // 2
    out: list[float] = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def _count_extrema(series: list[float]) -> tuple[int, int]:
    """Local maxima and minima on smoothed series (simple neighbor comparison)."""
    if len(series) < 3:
        return 0, 0
    peaks = troughs = 0
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            peaks += 1
        elif series[i] < series[i - 1] and series[i] < series[i + 1]:
            troughs += 1
    return peaks, troughs


def evaluate(frames: list[PoseFrame], extractor: PoseExtractor) -> EvaluationResult:
    if not frames:
        return EvaluationResult(
            0,
            [FeedbackItem("error", "No pose data found", [])],
            0,
            "cat_cow",
        )

    raw: list[tuple[int, float]] = []
    for f in frames:
        m = spine_metric(f, extractor)
        if m is not None:
            raw.append((f.frame_num, m))

    if len(raw) < 10:
        return EvaluationResult(
            0,
            [
                FeedbackItem(
                    "warning",
                    "Not enough frames with visible shoulders/hips for cat-cow",
                    [],
                )
            ],
            len(raw),
            "cat_cow",
        )

    values = [v for _, v in raw]
    smoothed = _moving_average(values, SMOOTH_WINDOW)
    span = max(smoothed) - min(smoothed)
    peaks, troughs = _count_extrema(smoothed)
    oscillations = min(peaks, troughs)
    cycle_score = (peaks + troughs) / 2.0

    feedback: list[FeedbackItem] = [
        FeedbackItem(
            "good",
            f"Analyzed {len(raw)} frames with full shoulder/hip visibility",
            [],
        )
    ]

    score = 100
    issue_frames: list[int] = []
    frame_nums = [fn for fn, _ in raw]

    # Frames with near-zero local motion relative to a short window.
    flat_frames: list[int] = []
    for i in range(len(smoothed)):
        lo = max(0, i - 3)
        hi = min(len(smoothed), i + 4)
        local = smoothed[lo:hi]
        if max(local) - min(local) < MIN_AMPLITUDE * 0.35:
            flat_frames.append(frame_nums[i])

    if span < MIN_AMPLITUDE:
        score -= 40
        issue_frames = frame_nums[:]
        feedback.append(
            FeedbackItem(
                "error",
                "Very little spine motion detected — exaggerate cat and cow or move fully in frame",
                frame_nums[:5],
            )
        )

    if cycle_score < MIN_CYCLES:
        score -= 35
        if not issue_frames:
            issue_frames = flat_frames or frame_nums[:: max(1, len(frame_nums) // 5)]
        feedback.append(
            FeedbackItem(
                "warning",
                "Unclear alternating cat/cow rhythm — aim for slow, repeated rounds",
                (flat_frames or frame_nums)[:5],
            )
        )

    if peaks == 0 or troughs == 0:
        score -= 25
        if not issue_frames:
            issue_frames = flat_frames or frame_nums[:5]
        feedback.append(
            FeedbackItem(
                "warning",
                "Could not see both flexion and extension phases (need side or angled view helps)",
                (flat_frames or frame_nums)[:5],
            )
        )

    score = max(0, min(100, score))

    if score >= 75:
        feedback.append(
            FeedbackItem(
                "good",
                f"Motion range OK (estimated amplitude {span:.3f}); ~{oscillations} clear half-cycles",
                [],
            )
        )

    return EvaluationResult(
        score,
        feedback,
        len(raw),
        "cat_cow",
        scored_frames=frame_nums,
        issue_frames=sorted(set(issue_frames)),
    )


if __name__ == "__main__":
    import sys

    from pose_extractor import PoseExtractor

    if len(sys.argv) < 2:
        print("Usage: python cat_cow.py <video_path>")
        sys.exit(1)

    with PoseExtractor() as ex:
        print(f"Extracting poses from {sys.argv[1]}...")
        frames = ex.extract_from_video(sys.argv[1])
        print(f"Got {len(frames)} frames with pose")
        r = evaluate(frames, ex)
        print(f"\nScore: {r.score}/100  (frames analyzed: {r.frames_analyzed})")
        for fb in r.feedback:
            print(f"  [{fb.status}] {fb.message}")
