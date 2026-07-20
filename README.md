# PT Form Analyzer

Full-stack physical therapy form analyzer: upload a rehab video (or use live webcam coaching) and get joint-angle feedback powered by MediaPipe pose estimation.

**Stack:** React (Vite) · FastAPI · MediaPipe · rule-based biomechanics evaluators

> **Demo note:** The free API host may sleep after idle periods. The first analyze request can take 30–60s while the server wakes. Live coaching runs entirely in the browser and does not need the API.

<!-- After deploy, replace with your live URLs:
**Live app:** https://YOUR-APP.vercel.app  
**API health:** https://YOUR-API.onrender.com/health
-->

---

## Architecture

```text
Upload path
  Browser ──POST /analyze──▶ FastAPI
                               ├─ OpenCV sample @ ~15 fps
                               ├─ MediaPipe Pose landmarks
                               ├─ Exercise evaluator (bird dog / bridge / cat-cow)
                               └─ JSON: score, feedback, timeline, pose_frames
  Browser ◀────────────────────┘
                               Client canvas draws skeleton + form timeline

Live path (no server)
  Webcam ──▶ MediaPipe Pose Landmarker (WASM) ──▶ JS rule checks ──▶ live cues
```

### Design choices

| Choice | Why |
|--------|-----|
| Rule-based scoring vs trained classifier | Interpretable cues (“straighten the leg”) without labeled PT datasets |
| 2D joint angles | MediaPipe landmarks are image-plane; side-view projection compresses angles — thresholds are intentionally wide |
| Hold-only scoring (bird dog / bridge) | Transitions are noisy; quality matters most at the end range |
| Live pose in the browser | Avoids free-tier API RAM/cold-start killing a real-time demo |
| Video analysis on the server | Reuses the Python evaluators + returns a full scored report |

---

## Supported exercises

1. **Bird dog** — back flatness, arm/leg parallel, leg extension (hold frames)
2. **Glute bridge** — hip height, knee consistency vs video median, shoulder level, alignment
3. **Cat-cow** — spine motion amplitude and alternating rhythm over time

---

## Local development

### Backend

```bash
cd backend/app
python -m venv ../../.venv
source ../../.venv/bin/activate   # Windows: ..\..\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — Vite proxies `/api` to port 8000.

For a production-style frontend build against a remote API:

```bash
cp .env.example .env.local
# set VITE_API_URL=https://your-api.onrender.com
npm run build
```

---

## Tests

From `backend/app` with the venv active and fixtures present under `backend/tests/fixtures/`:

```bash
PYTHONPATH=. pytest ../tests -q
```

Score-band smoke tests assert good/bad fixture behavior for bridge, bird dog, and cat-cow.

Manual CLI (prints scores):

```bash
python test_evaluators.py
```

---

## Deploy (free)

### API — Render (Docker)

1. Push this repo to GitHub.
2. In Render: **New → Blueprint** (uses [`render.yaml`](render.yaml)) **or** Web Service with:
   - Runtime: Docker
   - Dockerfile path: `backend/Dockerfile`
   - Docker context: `backend`
   - Health check: `/health`
3. Set env var `CORS_ORIGINS` to your Vercel URL (e.g. `https://your-app.vercel.app`).

### Frontend — Vercel

1. Import the repo; set **Root Directory** to `frontend`.
2. Framework: Vite (see [`frontend/vercel.json`](frontend/vercel.json)).
3. Env: `VITE_API_URL=https://YOUR-SERVICE.onrender.com` (no trailing slash).
4. Deploy.

---

## Interview talking points

- Why rules over a black-box classifier (data scarcity, controllable feedback)
- 2D projection limits and camera-angle robustness (bridge median-relative knees)
- Hold-only scoring vs scoring transitions
- Free-tier deploy constraints (RAM, cold start) and why live pose is client-side
- Failure modes: occlusion, bad camera angle, clothing contrast

---

## Project layout

```text
backend/
  Dockerfile
  app/            # FastAPI, MediaPipe extractor, evaluators, overlay
  tests/          # pytest + fixture videos
frontend/
  public/samples/ # recruiter “Try sample video” clips
  src/            # React UI, live coaching, skeleton overlay
```
