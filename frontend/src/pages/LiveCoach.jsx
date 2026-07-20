import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { POSE_CONNECTIONS, VISIBILITY_MIN } from "../constants/poseConnections.js";
import { FALLBACK_EXERCISES } from "../api.js";
import { createLiveContext, evaluateLiveFrame } from "../live/rules.js";
import { getGuide } from "../data/exerciseGuides.js";

const STATUS_LABEL = {
  ready: "Ready",
  hold: "Good hold",
  issue: "Adjust form",
};

export default function LiveCoach() {
  const { id } = useParams();
  const exercise =
    FALLBACK_EXERCISES.find((e) => e.id === id) || FALLBACK_EXERCISES[0];
  const guide = getGuide(id);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(0);
  const ctxRef = useRef(createLiveContext());
  const lastVideoTimeRef = useRef(-1);

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState("ready");
  const [cues, setCues] = useState(["Allow camera access to begin"]);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
        });
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;
        setReady(true);
      } catch (e) {
        if (!cancelled) {
          setError(
            e.message ||
              "Could not load pose model. Check your network connection."
          );
        }
      }
    }

    init();
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
      const stream = videoRef.current?.srcObject;
      stream?.getTracks()?.forEach((t) => t.stop());
    };
  }, []);

  useEffect(() => {
    ctxRef.current = createLiveContext();
  }, [id]);

  async function startCamera() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      setRunning(true);
      loop();
    } catch (e) {
      setError(
        e.name === "NotAllowedError"
          ? "Camera permission denied. Enable it in the browser and try again."
          : e.message || "Could not access camera"
      );
    }
  }

  function stopCamera() {
    cancelAnimationFrame(rafRef.current);
    const stream = videoRef.current?.srcObject;
    stream?.getTracks()?.forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setRunning(false);
    setStatus("ready");
    setCues(["Camera stopped"]);
  }

  function loop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !canvas || !landmarker || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(loop);
      return;
    }

    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;
      const result = landmarker.detectForVideo(video, performance.now());
      const landmarks = result.landmarks?.[0] || null;
      const evaluation = evaluateLiveFrame(id, landmarks, ctxRef.current);
      setStatus(evaluation.status);
      setCues(evaluation.cues.slice(0, 2));
      draw(canvas, video, landmarks, evaluation.status);
    }

    rafRef.current = requestAnimationFrame(loop);
  }

  function draw(canvas, video, landmarks, formStatus) {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    // Mirror to match selfie camera feel
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);

    if (landmarks) {
      const color =
        formStatus === "issue"
          ? "#c0392b"
          : formStatus === "hold"
            ? "#1a8f4a"
            : "#0d7a6b";
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = Math.max(3, w * 0.004);

      for (const [a, b] of POSE_CONNECTIONS) {
        const pa = landmarks[a];
        const pb = landmarks[b];
        if (!pa || !pb) continue;
        if ((pa.visibility ?? 1) < VISIBILITY_MIN) continue;
        if ((pb.visibility ?? 1) < VISIBILITY_MIN) continue;
        ctx.beginPath();
        ctx.moveTo(pa.x * w, pa.y * h);
        ctx.lineTo(pb.x * w, pb.y * h);
        ctx.stroke();
      }

      for (const p of landmarks) {
        if ((p.visibility ?? 1) < VISIBILITY_MIN) continue;
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, Math.max(4, w * 0.006), 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  return (
    <div className="app-shell live-shell">
      <header className="app-header">
        <Link to={`/exercise/${id}`}>← Back to {exercise.name}</Link>
      </header>

      <h1>Live: {exercise.name}</h1>
      <p className="lead">
        On-device pose estimation — no upload, no server round-trip. Position
        the camera for a side/angled full-body view.
      </p>

      {guide?.film?.items?.[0] && (
        <p className="meta" style={{ marginTop: "-0.5rem" }}>
          Tip: {guide.film.items[0]}
        </p>
      )}

      {error && <div className="error-banner">{error}</div>}

      <div className="live-stage">
        <video ref={videoRef} playsInline muted className="live-video-hidden" />
        <canvas ref={canvasRef} className="live-canvas" />
        {!running && (
          <div className="live-placeholder">
            {ready
              ? "Camera ready — tap Start"
              : "Loading pose model…"}
          </div>
        )}
      </div>

      <div className={`live-status live-status--${status}`}>
        <strong>{STATUS_LABEL[status] || status}</strong>
        <ul className="live-cues">
          {cues.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      </div>

      <div className="actions-row">
        {!running ? (
          <button
            type="button"
            className="btn btn-primary"
            disabled={!ready}
            onClick={startCamera}
          >
            Start camera
          </button>
        ) : (
          <button type="button" className="btn btn-secondary" onClick={stopCamera}>
            Stop
          </button>
        )}
        <Link to={`/exercise/${id}`} className="btn btn-secondary">
          Upload a video instead
        </Link>
      </div>
    </div>
  );
}
