import { Outlet } from "react-router-dom";
import SiteNav from "./SiteNav.jsx";

export default function Layout() {
  return (
    <div className="site">
      <SiteNav />
      <main className="site-main">
        <Outlet />
      </main>
      <footer className="site-footer">
        <p>
          Built for home rehab practice — not a medical device. Pose feedback is
          advisory; follow your clinician’s guidance.
        </p>
      </footer>
    </div>
  );
}
