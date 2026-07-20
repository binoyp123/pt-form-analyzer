import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Landing from "./pages/Landing.jsx";
import ExerciseSelect from "./pages/ExerciseSelect.jsx";
import ExerciseDetail from "./pages/ExerciseDetail.jsx";
import Results from "./pages/Results.jsx";
import LiveCoach from "./pages/LiveCoach.jsx";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Landing />} />
        <Route path="/exercises" element={<ExerciseSelect />} />
        <Route path="/exercise/:id" element={<ExerciseDetail />} />
        <Route path="/live/:id" element={<LiveCoach />} />
        <Route path="/results" element={<Results />} />
      </Route>
    </Routes>
  );
}
