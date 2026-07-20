# PT Form Analyzer

Full-stack physical therapy form analyzer: upload a rehab video (or use live webcam coaching) and get joint-angle feedback powered by MediaPipe pose estimation.

**Stack:** React (Vite) · FastAPI · MediaPipe · rule-based biomechanics evaluators

**Live app:** https://pt-form-analyzer.vercel.app  
**API health:** https://pt-form-analyzer-api.onrender.com/health

> **Demo note:** The free API host may sleep after idle periods. The first analyze request can take 30–60s while the server wakes. Live coaching runs entirely in the browser and does not need the API.

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
  Webcam ──▶ MediaPipe Pose Landmarker (WASM) ──▶ JS rules + voice cues
```

### Design choices

| Choice | Why |
|--------|-----|
| Rule-based scoring vs trained classifier | Interpretable cues (“straighten the leg”) without labeled PT datasets |
| 2D joint angles | MediaPipe landmarks are image-plane; side-view projection compresses angles — thresholds are intentionally wide |
| Hold-only scoring (bird dog / bridge) | Transitions are noisy; quality matters most at the end range |
| Live pose in the browser | Avoids free-tier API RAM/cold-start killing a real-time demo |
| Debounced live status + speech | Fewer red/green flickers; spoken cues with mute + cooldown |
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
# set VITE_API_URL=https://pt-form-analyzer-api.onrender.com
npm run build
```

---

## Tests

From `backend/app` with the venv active and fixtures present under `backend/tests/fixtures/`:

```bash
PYTHONPATH=. pytest ../tests -q
```

---

## Deploy (free)

### API — Render (Docker)

- Blueprint / Docker service from this repo (`backend/Dockerfile`)
- Health check: `/health`
- Env: `CORS_ORIGINS=https://pt-form-analyzer.vercel.app`

### Frontend — Vercel

- Root directory: `frontend`
- Env: `VITE_API_URL=https://pt-form-analyzer-api.onrender.com`

---

## Interview talking points

- Why rules over a black-box classifier (data scarcity, controllable feedback)
- 2D projection limits and camera-angle robustness (bridge median-relative knees)
- Hold-only scoring vs scoring transitions
- Free-tier deploy constraints (RAM, cold start) and why live pose is client-side
- Live coaching: on-device MediaPipe, status debounce, optional speech cues
- Personal motivation: home PT practice after a lower-back injury; assistive tool, not a medical device

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
