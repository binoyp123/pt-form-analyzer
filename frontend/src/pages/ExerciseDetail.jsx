import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { analyzeVideo, fetchExercises } from "../api.js";
import FilmingGuide from "../components/FilmingGuide.jsx";
import UploadButton from "../components/UploadButton.jsx";
import VideoPlayer from "../components/VideoPlayer.jsx";

const SAMPLE_PATHS = {
  bird_dog: "/samples/bird_dog.mp4",
  bridge: "/samples/bridge.mp4",
  cat_cow: "/samples/cat_cow.mp4",
};

const MAX_CLIENT_BYTES = 35 * 1024 * 1024;

export default function ExerciseDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [exercise, setExercise] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null);
  const [error, setError] = useState(null);
  const [loadingSample, setLoadingSample] = useState(false);

  useEffect(() => {
    fetchExercises()
      .then((list) => {
        const found = list.find((e) => e.id === id);
        if (!found) throw new Error("Exercise not found");
        setExercise(found);
      })
      .catch((e) => setError(e.message));
  }, [id]);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function validateFile(next) {
    if (!next) return false;
    if (next.size > MAX_CLIENT_BYTES) {
      setError("Video is too large. Please upload a file under 35 MB.");
      return false;
    }
    const typeOk =
      next.type?.startsWith("video/") ||
      /\.(mp4|mov|webm|m4v)$/i.test(next.name || "");
    if (!typeOk) {
      setError("Please choose a video file (MP4, MOV, or WebM).");
      return false;
    }
    setError(null);
    return true;
  }

  function handleFileSelect(next) {
    if (!validateFile(next)) return;
    setFile(next);
  }

  async function handleTrySample() {
    const path = SAMPLE_PATHS[id];
    if (!path) {
      setError("No sample video for this exercise yet.");
      return;
    }
    setLoadingSample(true);
    setError(null);
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error("Sample video not found");
      const blob = await res.blob();
      const sampleFile = new File([blob], `${id}_sample.mp4`, {
        type: blob.type || "video/mp4",
      });
      setFile(sampleFile);
    } catch (e) {
      setError(e.message || "Could not load sample video");
    } finally {
      setLoadingSample(false);
    }
  }

  async function handleAnalyze() {
    if (!file || !id) return;
    setLoading(true);
    setError(null);
    setStatusMsg(null);
    try {
      const result = await analyzeVideo(id, file, {
        onStatus: setStatusMsg,
      });
      navigate("/results", {
        state: {
          ...result,
          exerciseName: exercise?.name || id,
          videoFile: file,
        },
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setStatusMsg(null);
    }
  }

  if (error && !exercise) {
    return (
      <div className="app-shell">
        <div className="error-banner">{error}</div>
        <Link to="/exercises">← Back to exercises</Link>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <p className="page-crumb">
        <Link to="/exercises">← All exercises</Link>
      </p>

      <h1>{exercise?.name || "Loading…"}</h1>
      <p className="lead">{exercise?.description}</p>

      {id && <FilmingGuide exerciseId={id} />}

      <div className="actions-row" style={{ marginBottom: "1.25rem" }}>
        <Link to={`/live/${id}`} className="btn btn-secondary">
          Live camera coaching
        </Link>
      </div>

      <h2 className="section-title">Upload your video</h2>

      {error && <div className="error-banner">{error}</div>}

      <div className={`upload-zone ${file ? "has-file" : ""}`}>
        {previewUrl && <VideoPlayer src={previewUrl} />}
        {file ? (
          <p className="file-name">{file.name}</p>
        ) : (
          <p>MP4 or MOV · under 35 MB · ~10–30 seconds · side view, full body</p>
        )}
        <div className="actions-row" style={{ marginTop: "0.75rem" }}>
          <UploadButton
            onFileSelect={handleFileSelect}
            disabled={loading || loadingSample}
            label={file ? "Change video" : "Choose video"}
          />
          <button
            type="button"
            className="btn btn-secondary"
            disabled={loading || loadingSample || !SAMPLE_PATHS[id]}
            onClick={handleTrySample}
          >
            {loadingSample ? "Loading sample…" : "Try sample video"}
          </button>
        </div>
      </div>

      <div className="actions-row">
        <button
          type="button"
          className="btn btn-primary"
          disabled={!file || loading}
          onClick={handleAnalyze}
        >
          {loading ? (
            <>
              <span className="spinner" /> Analyzing…
            </>
          ) : (
            "Analyze form"
          )}
        </button>
      </div>

      {loading && (
        <p className="meta" style={{ marginTop: "1rem" }}>
          {statusMsg ||
            "This may take 15–60 seconds (longer if the free server is waking up)."}
        </p>
      )}
    </div>
  );
}
