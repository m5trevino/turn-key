"""Deck rotation utilities for TurnKey.

Slice: TB-TK-004
Scope: ``DeckRotator`` class that draws from deck in order, wraps around,
respects stage pinning, and is thread-safe.
Hard gates:
- Must not mutate the deck list during rotation.
- Must track position with atomic operations.
- Must handle empty deck gracefully.
"""

from __future__ import annotations

import threading
from typing import Any


class DeckRotator:
    """Rotates through a deck of models, respecting per-stage pinning.

    Behaviour:

    #. **Pinned stages** — when a stage has ``pinnedModels`` and
       ``useDeck`` is ``False``, always return the first pinned model.
    #. **Unpinned stages** — draw the next card from the deck in sequence
       and wrap around the end *without* reshuffling.

    Position is tracked with a ``threading.Lock`` so concurrent calls
    from multiple threads never return the same card twice.

    An empty deck raises ``RuntimeError`` on the first draw rather than
    silently failing.

    Args:
        deck: Ordered list of model identifiers.  A shallow copy is made
            so the caller's list cannot be mutated.
        stages_config: Mapping of stage name → stage config.  Each config
            dict may contain ``pinnedModels`` (list[str]) and
            ``useDeck`` (bool, defaults to ``True``).

    Example:
        >>> rotator = DeckRotator(
        ...     deck=["gpt-4", "claude-3", "gemini-1.5"],
        ...     stages_config={"distiller": {"pinnedModels": ["claude-3"], "useDeck": False}},
        ... )
        >>> rotator.next("distiller")
        'claude-3'
        >>> rotator.next("summary")
        'gpt-4'
        >>> rotator.next("summary")
        'claude-3'
    """

    def __init__(self, deck: list[str], stages_config: dict[str, Any]) -> None:
        if not isinstance(deck, list):
            raise TypeError("deck must be a list of strings")

        # Shallow copy so the internal list cannot be mutated by the caller.
        self._deck: list[str] = list(deck)
        self._stages: dict[str, Any] = stages_config
        self._pos: int = 0
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def next(self, stage: str) -> str:
        """Return the next model for *stage*.

        Pinned stages always return their first pinned model.
        Unpinned stages draw sequentially from the deck, wrapping around.

        Args:
            stage: The stage name (e.g. ``"distiller"``).

        Returns:
            A model identifier string.

        Raises:
            RuntimeError: If the deck is empty and the stage is unpinned.
        """
        cfg = self._stages.get(stage, {})
        pinned: list[str] = cfg.get("pinnedModels", [])
        use_deck: bool = cfg.get("useDeck", True)

        # Pinned stage: always return first pinned model.
        if pinned and not use_deck:
            return pinned[0]

        # Unpinned: draw sequentially with thread-safe wrap.
        with self._lock:
            if len(self._deck) == 0:
                raise RuntimeError(
                    f"Deck is empty for unpinned stage '{stage}'; "
                    "provide at least one model."
                )
            model = self._deck[self._pos]
            self._pos = (self._pos + 1) % len(self._deck)
            return model

    # ------------------------------------------------------------------ introspection

    @property
    def deck_size(self) -> int:
        """Return the number of cards in the deck."""
        return len(self._deck)

    @property
    def current_position(self) -> int:
        """Return the current (0-based) draw position."""
        with self._lock:
            return self._pos
