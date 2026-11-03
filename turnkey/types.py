"""
SDK data types, profile schemas, and exceptions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List

@dataclass
class NormalizedResponse:
    """Standardized response format for SDK operations"""
    content: str | None
    model: str | None
    usage: dict[str, Any] | None
    duration_ms: float | None

@dataclass
class StageError(Exception):
    """Exception raised when a stage execution fails after all retries are exhausted.

    Captures the full retry history, every model that was attempted, the root-cause
    exception that killed the last attempt, and a human-readable message suitable
    for logging and user display.

    The exception is **JSON-serializable** via ``to_dict()`` so it can be logged
    or transmitted across process boundaries without loss of diagnostic data.

    Attributes:
        stage: Name of the stage that failed.
        message: Human-readable explanation of the failure.
        retry_history: List of dicts, one per retry attempt, each containing
            the model that was tried and the exception that occurred.
        failed_models: Set of unique model identifiers that were attempted.
        root_cause: The original exception from the last retry attempt (if any).
        error_code: Optional structured error code for programmatic handling.
        details: Optional dict with extra diagnostic data.

    Example:
        >>> try:
        ...     raise StageError(
        ...         stage="distiller",
        ...         message="All retries exhausted for stage 'distiller'",
        ...         retry_history=[
        ...             {"model": "gpt-4", "error": "HTTP_503"},
        ...             {"model": "claude-3", "error": "TIMEOUT"},
        ...         ],
        ...     )
        ... except StageError as exc:
        ...     print(exc.to_dict()["failed_models"])
        ...
        ['gpt-4', 'claude-3']
    """

    stage: str
    message: str
    retry_history: Optional[list[dict[str, str]]] = None
    failed_models: Optional[set[str]] = None
    root_cause: Optional[Exception] = None
    error_code: Optional[str] = None
    details: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.retry_history is None:
            self.retry_history = []
        if self.failed_models is None:
            self.failed_models = {
                entry["model"] for entry in self.retry_history if "model" in entry
            }
        if self.error_code is None:
            self.error_code = "STAGE_FAILURE"
        # Ensure the exception message is available via str(exc).
        self.args = (self.message,)

    # --------------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation.

        Useful for structured logging, serializing to disk, or sending over a
        network.  The ``root_cause`` exception is converted to its ``str()``
        representation to avoid pickling problems.
        """
        return {
            "stage": self.stage,
            "message": self.message,
            "error_code": self.error_code,
            "retry_history": self.retry_history,
            "failed_models": sorted(self.failed_models) if self.failed_models else [],
            "root_cause": str(self.root_cause) if self.root_cause else None,
            "details": self.details,
        }

    def __str__(self) -> str:
        parts = [self.message]
        if self.retry_history:
            attempts = "; ".join(
                f"{entry.get('model', '?')}: {entry.get('error', 'unknown')}"
                for entry in self.retry_history
            )
            parts.append(f"[{len(self.retry_history)} attempt(s): {attempts}]")
        if self.root_cause:
            parts.append(f"(root cause: {self.root_cause})")
        return " | ".join(parts)


class ProfileValidationError(Exception):
    """Raised when a profile fails schema validation.

    Attributes:
        message: Human-readable explanation of the validation failure.
        path: The file path that was being validated, if any.
    """

    def __init__(self, message: str, *, path: Optional[str] = None) -> None:
        self.message = message
        self.path = path
        detail = f" in '{path}'" if path else ""
        super().__init__(f"Profile validation failed{detail}: {message}")

@dataclass
class Profile:
    """User or system profile information"""
    id: str
    name: str
    email: Optional[str] = None
    role: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class DeckCard:
    """Individual card in a deck for processing"""
    id: str
    title: str
    content: str
    tags: List[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class StageConfig:
    """Configuration for individual processing stages"""
    stage_name: str
    enabled: bool = True
    timeout: int = 30
    retries: int = 3
    max_concurrent: int = 1
    config_data: Optional[Dict[str, Any]] = None

class SDKConfig:
    """Configuration for TurnKey SDK"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None,
                 engine_url: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3):
        self.api_key = api_key
        self.base_url = base_url
        self.engine_url = engine_url or "http://localhost:3099"
        self.timeout = timeout
        self.max_retries = max_retries

class SDKError(Exception):
    """Base exception for TurnKey SDK errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
