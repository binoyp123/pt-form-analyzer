const API_BASE = import.meta.env.VITE_API_URL || "/api";

const FALLBACK_EXERCISES = [
  {
    id: "bird_dog",
    name: "Bird Dog",
    description: "Arm and leg extension exercise for core stability",
  },
  {
    id: "bridge",
    name: "Glute Bridge",
    description: "Hip lift with feet planted — glutes and hamstrings",
  },
  {
    id: "cat_cow",
    name: "Cat-Cow",
    description: "Quadruped spine flexion and extension flow",
  },
];

async function fetchWithTimeout(url, options = {}, timeoutMs = 90000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export async function wakeApi() {
  try {
    await fetchWithTimeout(`${API_BASE}/health`, {}, 45000);
  } catch {
    // Cold start may still succeed on the next request.
  }
}

export async function fetchExercises() {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/exercises`, {}, 45000);
    if (!res.ok) throw new Error("Could not load exercises");
    const data = await res.json();
    return data.exercises?.length ? data.exercises : FALLBACK_EXERCISES;
  } catch {
    return FALLBACK_EXERCISES;
  }
}

export async function analyzeVideo(exerciseId, videoFile, { onStatus } = {}) {
  onStatus?.("Waking the analysis server (first request after idle can take ~30–60s)…");
  await wakeApi();
  onStatus?.("Running pose estimation and form checks…");

  const form = new FormData();
  form.append("video", videoFile);
  form.append("exercise", exerciseId);

  let res;
  try {
    res = await fetchWithTimeout(
      `${API_BASE}/analyze`,
      { method: "POST", body: form },
      180000
    );
  } catch (err) {
    if (err?.name === "AbortError") {
      throw new Error(
        "Analysis timed out. The server may be waking up — please try again."
      );
    }
    throw new Error(
      "Could not reach the analysis server. Check your connection and try again."
    );
  }

  let data;
  try {
    data = await res.json();
  } catch {
    throw new Error("Unexpected response from the analysis server");
  }

  if (!res.ok) {
    throw new Error(data.error || "Analysis failed");
  }
  return data;
}

export { FALLBACK_EXERCISES };
