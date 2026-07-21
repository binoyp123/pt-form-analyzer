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

const STABLE_FRAMES = 10;
const SPEECH_COOLDOWN_MS = 2800;

function speakCue(text, enabled) {
  if (!enabled || !text || typeof window === "undefined") return;
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.05;
  u.pitch = 1;
  u.volume = 0.9;
  window.speechSynthesis.speak(u);
}

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

  const pendingStatusRef = useRef("ready");
  const pendingCountRef = useRef(0);
  const displayStatusRef = useRef("ready");
  const displayCueRef = useRef("");
  const lastSpeechAtRef = useRef(0);
  const lastSpokenRef = useRef("");
  const voiceEnabledRef = useRef(true);

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState("ready");
  const [cues, setCues] = useState(["Allow camera access to begin"]);
  const [voiceOn, setVoiceOn] = useState(true);

  useEffect(() => {
    voiceEnabledRef.current = voiceOn;
    if (!voiceOn && typeof window !== "undefined") {
      window.speechSynthesis?.cancel();
    }
  }, [voiceOn]);

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
      window.speechSynthesis?.cancel();
      const stream = videoRef.current?.srcObject;
      stream?.getTracks()?.forEach((t) => t.stop());
    };
  }, []);

  useEffect(() => {
    ctxRef.current = createLiveContext();
    pendingStatusRef.current = "ready";
    pendingCountRef.current = 0;
    displayStatusRef.current = "ready";
    displayCueRef.current = "";
    lastSpokenRef.current = "";
    setStatus("ready");
    setCues(["Allow camera access to begin"]);
  }, [id]);

  async function startCamera() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
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
    window.speechSynthesis?.cancel();
    setRunning(false);
    setStatus("ready");
    setCues(["Camera stopped"]);
    displayStatusRef.current = "ready";
    displayCueRef.current = "";
  }

  function applyStableEvaluation(evaluation) {
    if (evaluation.status === pendingStatusRef.current) {
      pendingCountRef.current += 1;
    } else {
      pendingStatusRef.current = evaluation.status;
      pendingCountRef.current = 1;
    }

    const stable =
      pendingCountRef.current >= STABLE_FRAMES ||
      evaluation.status === displayStatusRef.current;

    if (!stable) {
      draw(
        canvasRef.current,
        videoRef.current,
        evaluation.landmarks,
        displayStatusRef.current
      );
      return;
    }

    const nextStatus = evaluation.status;
    const nextCues = evaluation.cues.slice(0, 2);
    const primaryCue = nextCues[0] || "";

    displayStatusRef.current = nextStatus;
    displayCueRef.current = primaryCue;
    setStatus(nextStatus);
    setCues(nextCues.length ? nextCues : ["Keep going"]);

    const now = performance.now();
    const isPraise =
      nextStatus === "hold" &&
      /good|nice|solid|hold it/i.test(primaryCue);
    const shouldSpeak =
      voiceEnabledRef.current &&
      primaryCue &&
      primaryCue !== lastSpokenRef.current &&
      now - lastSpeechAtRef.current > SPEECH_COOLDOWN_MS &&
      (nextStatus === "issue" ||
        nextStatus === "ready" ||
        (isPraise && lastSpokenRef.current !== primaryCue));

    if (shouldSpeak) {
      lastSpeechAtRef.current = now;
      lastSpokenRef.current = primaryCue;
      const spoken =
        nextStatus === "issue" && nextCues[1]
          ? `${primaryCue} ${nextCues[1]}`
          : primaryCue;
      speakCue(spoken, true);
    }

    if (nextStatus === "hold" && !isPraise) {
      lastSpokenRef.current = "";
    }

    draw(
      canvasRef.current,
      videoRef.current,
      evaluation.landmarks,
      nextStatus
    );
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
      applyStableEvaluation({ ...evaluation, landmarks });
    }

    rafRef.current = requestAnimationFrame(loop);
  }

  function draw(canvas, video, landmarks, formStatus) {
    if (!canvas || !video) return;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
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
      <p className="page-crumb">
        <Link to={`/exercise/${id}`}>← {exercise.name}</Link>
      </p>

      <h1>Live: {exercise.name}</h1>
      <p className="lead">
        On-device pose estimation with spoken cues. Use a side or angled view
        with your full body in frame.
      </p>

      <div className="live-exercise-switch" role="navigation" aria-label="Live exercises">
        {FALLBACK_EXERCISES.map((ex) => (
          <Link
            key={ex.id}
            to={`/live/${ex.id}`}
            className={`live-switch-chip${ex.id === id ? " is-active" : ""}`}
          >
            {ex.name}
          </Link>
        ))}
      </div>

      {guide?.film?.items?.[0] && (
        <p className="meta meta-left">{guide.film.items[0]}</p>
      )}

      {error && <div className="error-banner">{error}</div>}

      <div className="live-stage">
        <video ref={videoRef} playsInline muted className="live-video-hidden" />
        <canvas ref={canvasRef} className="live-canvas" />
        {!running && (
          <div className="live-placeholder">
            {ready ? "Camera ready — tap Start" : "Loading pose model…"}
          </div>
        )}
        {running && cues[0] && (
          <div className={`live-banner live-banner--${status}`}>{cues[0]}</div>
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
        <button
          type="button"
          className={`btn btn-secondary${voiceOn ? "" : " is-muted"}`}
          onClick={() => setVoiceOn((v) => !v)}
          aria-pressed={voiceOn}
        >
          {voiceOn ? "Voice on" : "Voice muted"}
        </button>
        <Link to={`/exercise/${id}`} className="btn btn-secondary">
          Upload instead
        </Link>
      </div>
    </div>
  );
}
