from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import time

from pose_extractor import PoseExtractor
from evaluators import bird_dog

app = FastAPI(title="PT Form Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TMP_DIR = Path(__file__).parent / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXERCISES = {
    "bird_dog": {
        "name": "Bird Dog",
        "evaluator": bird_dog,
        "description": "Arm and leg extension exercise for core stability",
    },
    # Add more exercises here as you build them
    # "bridge": {"name": "Bridge", "evaluator": bridge, ...},
    # "cat_cow": {"name": "Cat Cow", "evaluator": cat_cow, ...},
}


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
        exercise: str = Form(default="bird_dog")
):
    """
    Analyze exercise form from uploaded video.

    - video: MP4/MOV video file
    - exercise: Exercise type (default: bird_dog)

    Returns score (0-100) and detailed feedback.
    """
    # Validate exercise type
    if exercise not in SUPPORTED_EXERCISES:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unknown exercise: {exercise}",
                "supported": list(SUPPORTED_EXERCISES.keys())
            }
        )

    # Validate video file
    if not video.content_type or not video.content_type.startswith("video/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload a video file"}
        )

    # Save video temporarily
    suffix = Path(video.filename).suffix or ".mp4"
    filename = f"form_video_{int(time.time())}{suffix}"
    video_path = TMP_DIR / filename

    try:
        contents = await video.read()
        video_path.write_bytes(contents)

        # Extract poses
        with PoseExtractor() as extractor:
            frames = extractor.extract_from_video(video_path)

            if not frames:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Could not detect any poses in video"}
                )

            # Run evaluation
            evaluator = SUPPORTED_EXERCISES[exercise]["evaluator"]
            result = evaluator.evaluate(frames, extractor)

        # Format response
        return {
            "success": True,
            "exercise": exercise,
            "score": result.score,
            "frames_analyzed": result.frames_analyzed,
            "feedback": [
                {
                    "status": fb.status,
                    "message": fb.message,
                    "problem_frames": fb.frames[:5] if fb.frames else []
                }
                for fb in result.feedback
            ]
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}"}
        )

    finally:
        # Clean up temp file
        if video_path.exists():
            video_path.unlink()