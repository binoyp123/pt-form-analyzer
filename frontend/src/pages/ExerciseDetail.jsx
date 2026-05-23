import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { analyzeVideo, fetchExercises } from "../api.js";
import UploadButton from "../components/UploadButton.jsx";
import VideoPlayer from "../components/VideoPlayer.jsx";

const TIPS = {
  bird_dog: [
    "Film from the side at hip height",
    "Hold each extension 2–3 seconds before switching",
    "Keep your back flat — avoid arching",
  ],
  bridge: [
    "Film from the side, full body visible",
    "Hold the top position for a few seconds per rep",
    "Feet hip-width, knees bent, drive hips up evenly",
  ],
  cat_cow: [
    "Film from the side in quadruped position",
    "Move slowly through 3–5 full cycles",
    "Let your spine move — arch up, then round down",
  ],
};

export default function ExerciseDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [exercise, setExercise] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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

  async function handleAnalyze() {
    if (!file || !id) return;
    setLoading(true);
    setError(null);
    try {
      const result = await analyzeVideo(id, file);
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
      <header className="app-header">
        <Link to="/exercises">← All exercises</Link>
      </header>

      <h1>{exercise?.name || "Loading…"}</h1>
      <p className="lead">{exercise?.description}</p>

      {error && <div className="error-banner">{error}</div>}

      <div className={`upload-zone ${file ? "has-file" : ""}`}>
        {previewUrl && <VideoPlayer src={previewUrl} />}
        {file ? (
          <p className="file-name">{file.name}</p>
        ) : (
          <p>Upload an MP4 or MOV (under ~30 seconds works best)</p>
        )}
        <UploadButton
          onFileSelect={setFile}
          disabled={loading}
          label={file ? "Change video" : "Choose video"}
        />
      </div>

      <ul className="tips">
        {(TIPS[id] || []).map((tip) => (
          <li key={tip}>{tip}</li>
        ))}
      </ul>

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
          This may take 15–45 seconds depending on video length.
        </p>
      )}
    </div>
  );
}
