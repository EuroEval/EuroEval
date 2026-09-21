"""Enums shared across the leaderboard pipeline."""

import enum


class LeaderboardCategory(enum.StrEnum):
    """The leaderboard category a model's row/rank belongs to."""

    CHAT = "chat"
    GENERATIVE = "generative"
    ALL_MODELS = "all_models"
