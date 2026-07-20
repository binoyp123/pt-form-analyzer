import { useEffect, useRef, useState } from "react";
import { Link, useLocation, Navigate } from "react-router-dom";
import FeedbackCard from "../components/FeedbackCard.jsx";
import FormTimeline from "../components/FormTimeline.jsx";
import VideoPlayer from "../components/VideoPlayer.jsx";
import VideoWithSkeleton from "../components/VideoWithSkeleton.jsx";

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

  const {
    score,
    exercise,
    exerciseName,
    frames_analyzed,
    feedback,
    videoFile,
    timeline,
    pose_frames,
    pose_connections,
  } = state;

  const [videoPreview, setVideoPreview] = useState(null);
  const [currentTimeMs, setCurrentTimeMs] = useState(0);
  const [durationSec, setDurationSec] = useState(0);
  const videoRef = useRef(null);

  const hasOverlay = pose_frames?.length > 0 && timeline?.length > 0;

  useEffect(() => {
    if (!videoFile) return;
    const url = URL.createObjectURL(videoFile);
    setVideoPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  function handleSeek(seconds) {
    const video = videoRef.current;
    if (video) {
      video.currentTime = seconds;
      setCurrentTimeMs(seconds * 1000);
    }
  }

  function handleSeekFrame(frameNum) {
    const entry =
      timeline?.find((t) => t.frame_num === frameNum) ||
      pose_frames?.find((p) => p.frame_num === frameNum);
    if (!entry) return;
    handleSeek((entry.timestamp_ms || 0) / 1000);
  }

  function handleTimeUpdate(t) {
    setCurrentTimeMs(t * 1000);
    const video = videoRef.current;
    if (video?.duration && !Number.isNaN(video.duration)) {
      setDurationSec(video.duration);
    }
  }

  return (
    <div className="app-shell">
      <p className="page-crumb">
        <Link to={`/exercise/${exercise}`}>← Analyze again</Link>
      </p>

      <h1>{exerciseName} results</h1>

      {videoPreview && hasOverlay ? (
        <VideoWithSkeleton
          src={videoPreview}
          poseFrames={pose_frames}
          connections={pose_connections}
          timeline={timeline}
          onTimeUpdate={handleTimeUpdate}
          videoRef={videoRef}
        />
      ) : (
        videoPreview && <VideoPlayer src={videoPreview} />
      )}

      {hasOverlay && (
        <FormTimeline
          timeline={timeline}
          durationSec={durationSec}
          currentTimeMs={currentTimeMs}
          onSeek={handleSeek}
        />
      )}

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
        <p className="meta" style={{ marginTop: 0, marginBottom: "0.75rem" }}>
          Click a warning to jump to the first problem moment in the video.
        </p>
        <div className="feedback-list">
          {feedback.map((item, i) => (
            <FeedbackCard
              key={`${item.status}-${i}`}
              status={item.status}
              message={item.message}
              problemFrames={item.problem_frames}
              onSeekFrame={handleSeekFrame}
            />
          ))}
        </div>
      </div>

      <div className="actions-row">
        <Link to={`/live/${exercise}`} className="btn btn-secondary">
          Try live coaching
        </Link>
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
