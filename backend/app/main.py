from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
import os
import uuid

import cv2

from pose_extractor import PoseExtractor
from evaluators import bird_dog, bridge, cat_cow
from overlay import build_timeline, get_pose_connections, serialize_pose_frames

logger = logging.getLogger("pt_form_analyzer")

app = FastAPI(title="PT Form Analyzer API")

# Comma-separated origins; empty / * → allow all (local dev).
_cors_raw = os.getenv("CORS_ORIGINS", "*").strip()
if _cors_raw in ("", "*"):
    _cors_origins = ["*"]
    _cors_credentials = False
else:
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
    _cors_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

TMP_DIR = Path(__file__).parent / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "35")) * 1024 * 1024
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "45"))
ALLOWED_SUFFIXES = {".mp4", ".mov", ".webm", ".avi", ".m4v"}

SUPPORTED_EXERCISES = {
    "bird_dog": {
        "name": "Bird Dog",
        "evaluator": bird_dog,
        "description": "Arm and leg extension exercise for core stability",
    },
    "bridge": {
        "name": "Glute Bridge",
        "evaluator": bridge,
        "description": "Hip lift with feet planted — glutes and hamstrings",
    },
    "cat_cow": {
        "name": "Cat-Cow",
        "evaluator": cat_cow,
        "description": "Quadruped spine flexion and extension flow",
    },
}


def _video_looks_valid(filename: str | None, content_type: str | None) -> bool:
    suffix = Path(filename or "").suffix.lower()
    if suffix in ALLOWED_SUFFIXES:
        return True
    if content_type and content_type.startswith("video/"):
        return True
    return False


def _probe_duration_sec(path: Path) -> float | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0 or frames <= 0:
        return None
    return frames / fps


@app.get("/")
def root():
    return {"message": "PT Form Analyzer API is running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/exercises")
def list_exercises():
    """List all supported exercises."""
    return {
        "exercises": [
            {"id": k, "name": v["name"], "description": v["description"]}
            for k, v in SUPPORTED_EXERCISES.items()
        ]
    }


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    exercise: str = Form(default="bird_dog"),
):
    """
    Analyze exercise form from uploaded video.

    - video: MP4/MOV video file
    - exercise: Exercise type (default: bird_dog)

    Returns score (0-100) and detailed feedback.
    """
    if exercise not in SUPPORTED_EXERCISES:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unknown exercise: {exercise}",
                "supported": list(SUPPORTED_EXERCISES.keys()),
            },
        )

    if not _video_looks_valid(video.filename, video.content_type):
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload a video file (MP4, MOV, or WebM)"},
        )

    suffix = Path(video.filename).suffix.lower() if video.filename else ".mp4"
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".mp4"
    filename = f"form_video_{uuid.uuid4().hex}{suffix}"
    video_path = TMP_DIR / filename

    try:
        contents = await video.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Video is too large. Please upload a file under {max_mb} MB.",
                },
            )
        if len(contents) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Uploaded file is empty"},
            )

        video_path.write_bytes(contents)

        duration = _probe_duration_sec(video_path)
        if duration is not None and duration > MAX_DURATION_SEC:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"Video is too long ({duration:.0f}s). "
                        f"Please keep clips under {int(MAX_DURATION_SEC)} seconds."
                    ),
                },
            )

        with PoseExtractor() as extractor:
            frames = extractor.extract_from_video(video_path)

            if not frames:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            "Could not detect any poses in the video. "
                            "Try a side/angled view with your full body in frame."
                        ),
                    },
                )

            evaluator = SUPPORTED_EXERCISES[exercise]["evaluator"]
            result = evaluator.evaluate(frames, extractor)

        return {
            "success": True,
            "exercise": exercise,
            "score": result.score,
            "frames_analyzed": result.frames_analyzed,
            "feedback": [
                {
                    "status": fb.status,
                    "message": fb.message,
                    "problem_frames": fb.frames[:5] if fb.frames else [],
                }
                for fb in result.feedback
            ],
            "timeline": build_timeline(
                frames, result.scored_frames, result.issue_frames
            ),
            "pose_frames": serialize_pose_frames(frames),
            "pose_connections": get_pose_connections(),
        }

    except Exception:
        logger.exception("Analysis failed")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Analysis failed. Please try a shorter clip or a different video.",
            },
        )

    finally:
        if video_path.exists():
            video_path.unlink()
