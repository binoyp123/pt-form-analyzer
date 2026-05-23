export default function UploadButton({ onFileSelect, disabled, label = "Choose video" }) {
  return (
    <label className="btn btn-secondary" style={{ cursor: disabled ? "not-allowed" : "pointer" }}>
      {label}
      <input
        type="file"
        accept="video/*"
        hidden
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFileSelect(file);
        }}
      />
    </label>
  );
}
