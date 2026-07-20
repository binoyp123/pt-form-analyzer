"""Score-band smoke tests for exercise evaluators."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(APP_DIR))

from pose_extractor import PoseExtractor  # noqa: E402
from evaluators.bird_dog import evaluate as eval_bird_dog  # noqa: E402
from evaluators.bridge import evaluate as eval_bridge  # noqa: E402
from evaluators.cat_cow import evaluate as eval_cat_cow  # noqa: E402


def _require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip(f"Missing fixture: {path.name}")
    return path


@pytest.fixture(scope="module")
def extractor():
    with PoseExtractor() as ext:
        yield ext


def _run(ext, filename: str, evaluator):
    path = _require(FIXTURES / filename)
    frames = ext.extract_from_video(path)
    assert frames, f"No poses extracted from {filename}"
    return evaluator(frames, ext)


def test_good_bridge_scores_high(extractor):
    result = _run(extractor, "good_bridge.mp4", eval_bridge)
    assert result.score >= 70, f"expected good bridge ≥70, got {result.score}"


def test_bad_bridge_scores_lower(extractor):
    good = _run(extractor, "good_bridge.mp4", eval_bridge)
    bad = _run(extractor, "bad_bridge.mp4", eval_bridge)
    assert bad.score <= good.score
    assert bad.score <= 85, f"expected bad bridge ≤85, got {bad.score}"


def test_bird_dog_detects_holds(extractor):
    result = _run(extractor, "bird_dog.mp4", eval_bird_dog)
    assert result.frames_analyzed > 0
    assert result.score >= 40


def test_bad_bird_dog_not_perfect(extractor):
    result = _run(extractor, "bad_bird_dog.mp4", eval_bird_dog)
    assert result.score <= 95


def test_cat_cow_produces_score(extractor):
    result = _run(extractor, "cat_cow.mp4", eval_cat_cow)
    assert 0 <= result.score <= 100
    assert result.frames_analyzed >= 10
