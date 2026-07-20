"""
Serialize pose data and build per-frame timeline for the results UI.
"""

from __future__ import annotations

from pose_extractor import PoseFrame

# MediaPipe BlazePose connections (landmark index pairs)
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


def serialize_pose_frames(frames: list[PoseFrame]) -> list[dict]:
    """Compact pose payload for client-side skeleton overlay."""
    out = []
    for f in frames:
        out.append({
            "frame_num": f.frame_num,
            "timestamp_ms": round(f.timestamp_ms, 1),
            "landmarks": [
                [
                    round(lm["x"], 4),
                    round(lm["y"], 4),
                    round(lm.get("visibility", 0), 2),
                ]
                for lm in f.landmarks
            ],
        })
    return out


def build_timeline(
    frames: list[PoseFrame],
    scored_frames: list[int],
    issue_frames: list[int],
) -> list[dict]:
    """
    One entry per extracted pose frame.
    status: good | issue | neutral (not scored by evaluator)
    """
    scored = set(scored_frames)
    issues = set(issue_frames)

    timeline = []
    for f in frames:
        if f.frame_num in issues:
            status = "issue"
        elif f.frame_num in scored:
            status = "good"
        else:
            status = "neutral"
        timeline.append({
            "frame_num": f.frame_num,
            "timestamp_ms": round(f.timestamp_ms, 1),
            "status": status,
        })
    return timeline


def get_pose_connections() -> list[list[int]]:
    """Return skeleton edge list for the API."""
    return [list(pair) for pair in POSE_CONNECTIONS]
