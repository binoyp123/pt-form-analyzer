import { useEffect, useRef, useState } from "react";
import { Link, NavLink } from "react-router-dom";
import { FALLBACK_EXERCISES } from "../api.js";

export default function SiteNav() {
  const [open, setOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    function onDocClick(e) {
      if (!menuRef.current?.contains(e.target)) setOpen(false);
    }
    function onKey(e) {
      if (e.key === "Escape") {
        setOpen(false);
        setMobileOpen(false);
      }
    }
    document.addEventListener("click", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("click", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  return (
    <header className="site-nav">
      <div className="site-nav__inner">
        <Link to="/" className="site-nav__brand" onClick={() => setMobileOpen(false)}>
          <span className="site-nav__mark" aria-hidden="true" />
          <span className="site-nav__name">PT Form Analyzer</span>
        </Link>

        <button
          type="button"
          className="site-nav__burger"
          aria-expanded={mobileOpen}
          aria-label="Menu"
          onClick={() => setMobileOpen((v) => !v)}
        >
          <span />
          <span />
        </button>

        <nav className={`site-nav__links ${mobileOpen ? "is-open" : ""}`}>
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `site-nav__link${isActive ? " is-active" : ""}`
            }
            onClick={() => setMobileOpen(false)}
          >
            Home
          </NavLink>

          <div className="site-nav__dropdown" ref={menuRef}>
            <button
              type="button"
              className={`site-nav__link site-nav__trigger${open ? " is-open" : ""}`}
              aria-expanded={open}
              onClick={() => setOpen((v) => !v)}
            >
              Exercises
              <span className="site-nav__chevron" aria-hidden="true">
                ▾
              </span>
            </button>
            {open && (
              <div className="site-nav__menu" role="menu">
                <Link
                  to="/exercises"
                  role="menuitem"
                  onClick={() => {
                    setOpen(false);
                    setMobileOpen(false);
                  }}
                >
                  All exercises
                </Link>
                {FALLBACK_EXERCISES.map((ex) => (
                  <Link
                    key={ex.id}
                    to={`/exercise/${ex.id}`}
                    role="menuitem"
                    onClick={() => {
                      setOpen(false);
                      setMobileOpen(false);
                    }}
                  >
                    {ex.name}
                  </Link>
                ))}
              </div>
            )}
          </div>

          <NavLink
            to="/live/bird_dog"
            className={({ isActive }) =>
              `site-nav__link${isActive ? " is-active" : ""}`
            }
            onClick={() => setMobileOpen(false)}
          >
            Live coaching
          </NavLink>

          <Link
            to="/exercises"
            className="btn btn-primary site-nav__cta"
            onClick={() => setMobileOpen(false)}
          >
            Analyze form
          </Link>
        </nav>
      </div>
    </header>
  );
}
