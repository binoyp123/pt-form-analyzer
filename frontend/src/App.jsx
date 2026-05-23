import { Routes, Route } from "react-router-dom";
import Landing from "./pages/Landing.jsx";
import ExerciseSelect from "./pages/ExerciseSelect.jsx";
import ExerciseDetail from "./pages/ExerciseDetail.jsx";
import Results from "./pages/Results.jsx";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/exercises" element={<ExerciseSelect />} />
      <Route path="/exercise/:id" element={<ExerciseDetail />} />
      <Route path="/results" element={<Results />} />
    </Routes>
  );
}
