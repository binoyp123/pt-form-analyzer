const API_BASE = import.meta.env.VITE_API_URL || "/api";

export async function fetchExercises() {
  const res = await fetch(`${API_BASE}/exercises`);
  if (!res.ok) throw new Error("Could not load exercises");
  const data = await res.json();
  return data.exercises;
}

export async function analyzeVideo(exerciseId, videoFile) {
  const form = new FormData();
  form.append("video", videoFile);
  form.append("exercise", exerciseId);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "Analysis failed");
  }
  return data;
}
