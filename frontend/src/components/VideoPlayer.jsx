export default function VideoPlayer({ src }) {
  if (!src) return null;
  return (
    <video className="video-preview" src={src} controls playsInline />
  );
}
