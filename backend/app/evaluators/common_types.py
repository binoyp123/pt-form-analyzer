"""Shared result types for exercise evaluators."""

from dataclasses import dataclass


@dataclass
class FeedbackItem:
    status: str  # "good", "warning", "error"
    message: str
    frames: list[int]


@dataclass
class EvaluationResult:
    score: int  # 0-100
    feedback: list[FeedbackItem]
    frames_analyzed: int
    exercise: str
