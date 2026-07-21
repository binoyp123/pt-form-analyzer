"""
Pose extraction using MediaPipe.
Takes a video file and returns pose landmarks for each frame.
Tuned for low-memory hosts (e.g. Render free tier).
"""

import cv2
import mediapipe as mp
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class PoseFrame:
    """Pose data for a single frame."""
    frame_num: int
    timestamp_ms: float
    landmarks: list
    image_width: int
    image_height: int


class PoseExtractor:
    # Key landmark indices for PT exercises
    LANDMARKS = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }

    def __init__(
        self,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=None,
    ):
        if model_complexity is None:
            model_complexity = int(os.getenv("POSE_MODEL_COMPLEXITY", "0"))
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.max_width = int(os.getenv("POSE_MAX_WIDTH", "480"))

    def extract_from_video(self, video_path: str | Path, target_fps: float = 10) -> list[PoseFrame]:
        """
        Extract poses from a video, sampling at *target_fps* to avoid
        processing every frame of high-fps footage (e.g. 60 fps iPhone video).
        Frames are downscaled before inference to reduce RAM.
        Set target_fps=0 to process every frame.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if target_fps is None:
            target_fps = float(os.getenv("POSE_TARGET_FPS", "10"))
        step = max(1, int(fps / target_fps)) if target_fps > 0 else 1
        max_frames = int(os.getenv("POSE_MAX_FRAMES", "90"))
        frames = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % step == 0:
                h, w = frame.shape[:2]
                if w > self.max_width:
                    scale = self.max_width / w
                    frame = cv2.resize(
                        frame,
                        (self.max_width, int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb)

                if results.pose_landmarks:
                    landmarks = [
                        {
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility,
                        }
                        for lm in results.pose_landmarks.landmark
                    ]

                    frames.append(PoseFrame(
                        frame_num=frame_num,
                        timestamp_ms=(frame_num / fps) * 1000,
                        landmarks=landmarks,
                        image_width=frame.shape[1],
                        image_height=frame.shape[0],
                    ))
                    if len(frames) >= max_frames:
                        break

            frame_num += 1

        cap.release()
        return frames

    def get_landmark(self, pose_frame: PoseFrame, name: str) -> dict | None:
        """Get a specific landmark by name from a PoseFrame."""
        if name not in self.LANDMARKS:
            return None
        idx = self.LANDMARKS[name]
        if idx < len(pose_frame.landmarks):
            return pose_frame.landmarks[idx]
        return None

    def get_pixel_coords(self, pose_frame: PoseFrame, name: str) -> tuple[int, int] | None:
        """Get landmark position in pixel coordinates."""
        lm = self.get_landmark(pose_frame, name)
        if lm is None:
            return None
        x = int(lm["x"] * pose_frame.image_width)
        y = int(lm["y"] * pose_frame.image_height)
        return (x, y)

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

# Utility functions for evaluators
def calc_angle(p1: dict, p2: dict, p3: dict) -> float:
    """
    Calculate angle at p2 formed by p1-p2-p3.
    Returns angle in degrees (0-180).
    """
    import math

    v1 = (p1["x"] - p2["x"], p1["y"] - p2["y"])
    v2 = (p3["x"] - p2["x"], p3["y"] - p2["y"])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if mag1 * mag2 == 0:
        return 0.0

    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def is_landmark_visible(landmark: dict, threshold: float = 0.5) -> bool:
    """Check if a landmark is visible enough to use."""
    return landmark is not None and landmark.get("visibility", 0) >= threshold


# Quick test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pose_extractor.py <video_path>")
        sys.exit(1)

    with PoseExtractor() as extractor:
        frames = extractor.extract_from_video(sys.argv[1])
        print(f"Extracted {len(frames)} frames with pose data")

        if frames:
            sample = frames[0]
            print(f"\nSample frame {sample.frame_num}:")
            print(f"  Timestamp: {sample.timestamp_ms:.1f}ms")
            print(f"  Image size: {sample.image_width}x{sample.image_height}")

            shoulder = extractor.get_landmark(sample, "left_shoulder")
            if shoulder:
                print(f"  Left shoulder: ({shoulder['x']:.3f}, {shoulder['y']:.3f})")
