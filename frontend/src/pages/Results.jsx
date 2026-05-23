import { useEffect, useState } from "react";
import { Link, useLocation, Navigate } from "react-router-dom";
import FeedbackCard from "../components/FeedbackCard.jsx";
import VideoPlayer from "../components/VideoPlayer.jsx";

function scoreClass(score) {
  if (score >= 85) return "good";
  if (score >= 65) return "ok";
  return "poor";
}

export default function Results() {
  const { state } = useLocation();

  if (!state?.success) {
    return <Navigate to="/exercises" replace />;
  }

  const { score, exercise, exerciseName, frames_analyzed, feedback, videoFile } =
    state;

  const [videoPreview, setVideoPreview] = useState(null);

  useEffect(() => {
    if (!videoFile) return;
    const url = URL.createObjectURL(videoFile);
    setVideoPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <Link to={`/exercise/${exercise}`}>← Analyze again</Link>
      </header>

      <h1>{exerciseName} results</h1>

      {videoPreview && <VideoPlayer src={videoPreview} />}

      <div className="card" style={{ textAlign: "center" }}>
        <div className={`score-ring ${scoreClass(score)}`}>
          <span className="value">{score}</span>
          <span className="label">out of 100</span>
        </div>
        <p className="meta">
          Analyzed {frames_analyzed} pose frame
          {frames_analyzed === 1 ? "" : "s"}
        </p>
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <h2>Feedback</h2>
        <div className="feedback-list">
          {feedback.map((item, i) => (
            <FeedbackCard
              key={`${item.status}-${i}`}
              status={item.status}
              message={item.message}
            />
          ))}
        </div>
      </div>

      <div className="actions-row">
        <Link to="/exercises" className="btn btn-secondary">
          Try another exercise
        </Link>
        <Link to={`/exercise/${exercise}`} className="btn btn-primary">
          Upload new video
        </Link>
      </div>
    </div>
  );
}
