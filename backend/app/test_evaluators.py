"""
Run evaluator tests against video files in the project root.

Usage from backend/app/:
    python test_evaluators.py              # run all tests
    python test_evaluators.py bridge       # only tests matching "bridge"
    python test_evaluators.py cat          # only tests matching "cat"
"""

import sys
import os

sys.stdout.reconfigure(line_buffering=True)

from pose_extractor import PoseExtractor
from evaluators.bird_dog import evaluate as eval_bird_dog
from evaluators.bridge import evaluate as eval_bridge
from evaluators.cat_cow import evaluate as eval_cat_cow

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TESTS = [
    ("Bird Dog - Good Form", "bird_dog.mp4", eval_bird_dog),
    ("Bird Dog - Bad Form",  "bad_bird_dog.mp4", eval_bird_dog),
    ("Bridge - Good Form",   "good_bridge.mp4", eval_bridge),
    ("Bridge - Bad Form",    "bad_bridge.mp4", eval_bridge),
    ("Cat-Cow",              "cat_cow.mp4", eval_cat_cow),
]

FILTER = sys.argv[1].lower() if len(sys.argv) > 1 else None


def run():
    with PoseExtractor() as ext:
        for label, filename, evaluator in TESTS:
            if FILTER and FILTER not in label.lower():
                continue

            path = os.path.join(PROJECT_ROOT, filename)
            if not os.path.exists(path):
                print(f"\nSKIP: {label} — {filename} not found")
                continue

            print(f"\n{'='*50}")
            print(f"TEST: {label}  ({filename})")
            print(f"{'='*50}")

            frames = ext.extract_from_video(path)
            print(f"  Frames extracted: {len(frames)}")

            result = evaluator(frames, ext)
            print(f"  SCORE: {result.score}/100")
            print(f"  Frames analyzed: {result.frames_analyzed}")

            for fb in result.feedback:
                icon = {"good": "+", "warning": "!", "error": "X"}[fb.status]
                print(f"  [{icon}] {fb.message}")

    print(f"\n{'='*50}")
    print("All tests complete.")


if __name__ == "__main__":
    run()
