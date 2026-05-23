# PT Form Analyzer

Analyze physical therapy exercise form from uploaded videos using MediaPipe pose detection.

## Backend

```bash
cd backend/app
source ../../.venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — API requests proxy to the backend at port 8000.

## CLI tests (evaluators)

From `backend/app/` with the venv active:

```bash
python test_evaluators.py
```

Place test videos (`bird_dog.mp4`, `good_bridge.mp4`, etc.) in the project root.
