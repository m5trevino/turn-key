"""Profile loader — reads, parses and validates Turn Key / Peacock profiles.

Slice: TB-TK-003
Scope: public ``load_profile(path)`` that returns a validated profile dict.
Hard gates:
- Must support both ``deck`` / ``attempts`` fields for backward compatibility.
- Must validate profile structure before use.
- Must not load any external resources.
- Must not execute any attempt or call the engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import ProfileValidationError


def load_profile(path: str | Path | None = None) -> dict[str, Any]:
    """Load, parse and validate a profile from disk.

    Supports **two schema flavours** (mutually exclusive in validation):

    #. **TurnKey** (new) — the profile must contain ``deck``, ``stages`` and
       ``pipeline`` keys.
    #. **Peacock** (legacy) — the profile must contain an ``attempts`` key.

    A profile that contains *neither* schema flavour raises
    ``ProfileValidationError``.

    Args:
        path: Absolute or relative path to the profile file.  When ``None``
            the default convention ``peacock_profile.json`` in the current
            working directory is used.

    Returns:
        The parsed (and validated) profile dict.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ProfileValidationError: if the file is malformed or the schema is
            invalid (missing required fields, wrong types, empty deck).
        ValueError: if the file contents are not valid JSON.
    """
    if path is None:
        path = "peacock_profile.json"

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    text = p.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ProfileValidationError("Profile root must be a JSON object", path=path)

    _validate(raw, path)
    return raw


# ------------------------------------------------------------------ private

_REQUIRED_FIELDS: dict[str, list[str]] = {
    "turnkey": ["deck", "stages", "pipeline"],
    "peacock": ["attempts"],
}


def _validate(profile: dict[str, Any], path: str | Path) -> None:
    """Run schema validation against *profile*.

    Validates that at least one recognised schema flavour is present and that
    all required fields exist with the correct types.
    """
    has_deck = "deck" in profile
    has_attempts = "attempts" in profile

    if not has_deck and not has_attempts:
        raise ProfileValidationError(
            "Profile must contain either 'deck' (TurnKey schema) "
            "or 'attempts' (Peacock legacy schema)",
            path=path,
        )

    # --- TurnKey (new) schema ---
    if has_deck:
        _validate_turnkey(profile, path)

    # --- Peacock (legacy) schema ---
    if has_attempts:
        _validate_peacock(profile, path)


def _validate_turnkey(profile: dict[str, Any], path: str | Path) -> None:
    """Validate the new TurnKey schema flavour."""
    for field in _REQUIRED_FIELDS["turnkey"]:
        if field not in profile:
            raise ProfileValidationError(
                f"Missing required field for TurnKey schema: {field}",
                path=path,
            )

    deck = profile["deck"]
    if not isinstance(deck, list):
        raise ProfileValidationError(
            "'deck' must be a list",
            path=path,
        )
    if len(deck) == 0:
        raise ProfileValidationError(
            "'deck' must contain at least one card",
            path=path,
        )

    stages = profile["stages"]
    if not isinstance(stages, (list, dict)):
        raise ProfileValidationError(
            "'stages' must be a list or object",
            path=path,
        )

    pipeline = profile["pipeline"]
    if not isinstance(pipeline, dict):
        raise ProfileValidationError(
            "'pipeline' must be an object",
            path=path,
        )


def _validate_peacock(profile: dict[str, Any], path: str | Path) -> None:
    """Validate the legacy Peacock schema flavour."""
    attempts = profile["attempts"]
    if not isinstance(attempts, list):
        raise ProfileValidationError(
            "'attempts' must be a list",
            path=path,
        )
    if len(attempts) == 0:
        raise ProfileValidationError(
            "'attempts' must contain at least one attempt",
            path=path,
        )

    for idx, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise ProfileValidationError(
                f"attempts[{idx}] must be an object",
                path=path,
            )
        if "model" not in attempt or not isinstance(attempt["model"], str) or attempt["model"] == "":
            raise ProfileValidationError(
                f"attempts[{idx}] requires a non-empty 'model' string",
                path=path,
            )
