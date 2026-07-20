import { useCallback, useEffect, useRef } from "react";
import { POSE_CONNECTIONS, VISIBILITY_MIN } from "../constants/poseConnections.js";
import { findPoseAtTime, findTimelineEntry } from "../utils/pose.js";

const COLORS = {
  good: "#1a8f4a",
  issue: "#c0392b",
  neutral: "#0d7a6b",
};

export default function VideoWithSkeleton({
  src,
  poseFrames,
  connections = POSE_CONNECTIONS,
  timeline,
  onTimeUpdate,
  videoRef: externalVideoRef,
}) {
  const internalVideoRef = useRef(null);
  const canvasRef = useRef(null);
  const videoRef = externalVideoRef ?? internalVideoRef;

  const drawOverlay = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !poseFrames?.length) return;

    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const timeMs = video.currentTime * 1000;
    const pose = findPoseAtTime(poseFrames, timeMs);
    if (!pose?.landmarks) return;

    const entry = findTimelineEntry(timeline, pose.frame_num);
    const color = COLORS[entry?.status ?? "neutral"];

    const toPx = (lm) => ({
      x: lm[0] * w,
      y: lm[1] * h,
      v: lm[2],
    });

    const points = pose.landmarks.map(toPx);

    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.strokeStyle = color;
    for (const [a, b] of connections) {
      const p1 = points[a];
      const p2 = points[b];
      if (!p1 || !p2 || p1.v < VISIBILITY_MIN || p2.v < VISIBILITY_MIN) continue;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    for (const p of points) {
      if (p.v < VISIBILITY_MIN) continue;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, [poseFrames, connections, timeline, videoRef]);

  const syncCanvasSize = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    drawOverlay();
  }, [drawOverlay, videoRef]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const onTime = () => {
      drawOverlay();
      onTimeUpdate?.(video.currentTime);
    };

    video.addEventListener("timeupdate", onTime);
    video.addEventListener("seeked", drawOverlay);
    video.addEventListener("loadedmetadata", syncCanvasSize);
    window.addEventListener("resize", syncCanvasSize);

    return () => {
      video.removeEventListener("timeupdate", onTime);
      video.removeEventListener("seeked", drawOverlay);
      video.removeEventListener("loadedmetadata", syncCanvasSize);
      window.removeEventListener("resize", syncCanvasSize);
    };
  }, [src, drawOverlay, syncCanvasSize, onTimeUpdate, videoRef]);

  return (
    <div className="video-skeleton">
      <video
        ref={videoRef}
        className="video-skeleton__video"
        src={src}
        controls
        playsInline
      />
      <canvas ref={canvasRef} className="video-skeleton__canvas" aria-hidden="true" />
    </div>
  );
}
