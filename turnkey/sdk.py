"""TurnKey SDK main module.

Slice: TB-TK-007 (SDK Main Class) — implements the main SDK class.
Previous slices established the package structure, types, profile loader, rotator, 
and stage pinning.

Hard gates:
- Must integrate with existing profile structure
- Must maintain all existing functionality
- Must load profile at init
- Must initialize rotator with deck and stages
- Must provide run_stage() interface

Enhancements (TB-TK-011, TB-TK-017, TB-TK-024):
- Normalize ``deck`` entries (card objects ``{"modelId": ...}`` → model-id strings)
  so profiles like the Dendritic turn-key profiles work without modification.
- Normalize ``stages`` (list of ``{"stageName": ..., "pinnedModels": ...,
  "useDeck": ...}`` → dict keyed by stage name) for the ``DeckRotator``.
- Load ``timeout``, ``max_retries`` and ``engine_url`` from the profile's
  ``pipeline`` section when present, falling back to sensible defaults.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List

from .loader import load_profile
from .gateway_rotator import GatewayDeckRotator
from .types import SDKConfig, SDKError, Profile, NormalizedResponse, StageError


#: Default engine URL (matches the Peacock engine convention).
_DEFAULT_ENGINE_URL: str = "http://engine.save-aichats.com:3099"
#: Default per-call timeout in seconds.
_DEFAULT_TIMEOUT: int = 180
#: Default maximum retry attempts.
_DEFAULT_MAX_RETRIES: int = 10


# Models that cannot serve a chat completion regardless of what the catalog
# says: embedding, reranking, compression, transcription and vision utilities.
# They appear "active" in /v1/models because that endpoint lists everything the
# gateway can bill for, not everything that answers a chat request. Leaving
# them in the deck means spending a call on a hard 400 every lap.
_NON_CHAT = re.compile(
    r"(embed|rerank|scaledown/|classify|moderation|whisper|tts|ocr|"
    r"dall-e|flux|stable-diffusion|image-|video-)",
    re.IGNORECASE,
)


def is_chat_model(model_id: str) -> bool:
    """Whether a model id can answer a /v1/chat/completions request."""
    return bool(model_id) and not _NON_CHAT.search(model_id)


def _normalize_gateway_models(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize profile models into gateway structure for GatewayDeckRotator.
    
    Converts the deck structure into the expected gateway format with:
    - gateway: provider name
    - models: list of model dictionaries with 'id' and 'name'
    
    Args:
        profile: The loaded profile dictionary
        
    Returns:
        A list of gateway configurations for the GatewayDeckRotator
    """
    # Get the deck from the profile
    deck = profile.get("deck", [])
    
    if not isinstance(deck, list):
        return []
    
    # Group models by provider/gateway
    gateway_groups: dict[str, list[dict[str, str]]] = {}
    skipped_utility: list[str] = []
    
    for item in deck:
        if isinstance(item, dict):
            # Extract model ID and provider
            model_id = item.get("modelId") or item.get("modelName")
            provider = item.get("provider", "unknown")
            
            if model_id and not is_chat_model(model_id):
                # Utility model: skip it rather than burn a call on a 400.
                skipped_utility.append(model_id)
            elif model_id:
                # Create model entry
                model_entry = {
                    "id": model_id,
                    "name": model_id  # Using model_id as name for simplicity
                }
                
                # Group by provider
                if provider not in gateway_groups:
                    gateway_groups[provider] = []
                gateway_groups[provider].append(model_entry)
    
    # Convert to the expected format for GatewayDeckRotator
    result = []
    for gateway_name, models in gateway_groups.items():
        result.append({
            "gateway": gateway_name,
            "models": models
        })
    
    return result


def _normalize_deck(deck: List[Any]) -> List[str]:
    """Normalize a profile ``deck`` into a list of model-id strings.

    Accepts either a list of plain model-id strings or a list of card
    objects of the form ``{"uid": ..., "modelId": ..., "modelName": ...,
    "provider": ...}`` (the Dendritic turn-key profile format).  Card
    objects are mapped to their ``modelId`` (falling back to ``modelName``).

    Args:
        deck: The raw ``deck`` value from the profile.

    Returns:
        A list of model-id strings.
    """
    if not isinstance(deck, list):
        return []
    normalized: List[str] = []
    for card in deck:
        if isinstance(card, str):
            normalized.append(card)
        elif isinstance(card, dict):
            model_id = card.get("modelId") or card.get("modelName") or ""
            if isinstance(model_id, str) and model_id:
                normalized.append(model_id)
    return normalized


def _normalize_stages(stages: Any) -> Dict[str, Any]:
    """Normalize a profile ``stages`` value into a stage-keyed dict.

    Accepts either a dict (already keyed by stage name) or a list of
    stage-config objects of the form ``{"stageName": ..., "pinnedModels":
    [...], "useDeck": bool}`` (the Dendritic turn-key profile format).

    Args:
        stages: The raw ``stages`` value from the profile.

    Returns:
        A dict keyed by stage name, each value a config dict with
        ``pinnedModels`` and ``useDeck``.
    """
    if isinstance(stages, dict):
        return stages
    if not isinstance(stages, list):
        return {}
    out: Dict[str, Any] = {}
    for entry in stages:
        if not isinstance(entry, dict):
            continue
        name = entry.get("stageName") or entry.get("name") or entry.get("stage")
        if not isinstance(name, str) or not name:
            continue
        out[name] = {
            "pinnedModels": entry.get("pinnedModels", []) or [],
            "useDeck": bool(entry.get("useDeck", True)),
        }
    return out


def _load_dotenv() -> None:
    """Load every .env found, nearest first, without stopping at the first.

    An unrelated ~/.env used to win purely because it was found first, leaving
    ENGINE_KEY unset and every call returning 401. All candidate files are now
    merged and no file aborts the search. Existing environment values still
    win, so an explicit export overrides any file.
    """
    candidates: list[Path] = []
    try:
        candidates.append(Path.cwd() / ".env")
    except OSError:
        pass
    for base in (Path(__file__).resolve(), Path.cwd()):
        for parent in list(base.parents)[:6]:
            candidates.append(parent / ".env")

    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(),
                                      value.strip().strip('"').strip("'"))
        except OSError:
            continue


_load_dotenv()


class SDK:
    """Main SDK class for TurnKey.

    Orchestrates profile loading, model rotation, and HTTP calls to the
    engine service.

    Args:
        config: An ``SDKConfig`` instance.  When ``None`` a default config
            is built from the profile's ``pipeline`` section (falling back
            to sensible defaults: 180s timeout, 10 retries, the Peacock
            engine URL).
        profile_path: Path to the profile file. When ``None`` the default
            convention ``peacock_profile.json`` in the current working directory
            is used.

    Example:
        >>> sdk = SDK(SDKConfig(engine_url="http://localhost:3099"))
        >>> raw = sdk._call_engine("gpt-4", messages=[{"role": "user", "content": "hello"}])
        >>> isinstance(raw, dict)
        True
    """

    def __init__(self, config: Optional[SDKConfig] = None,
                 profile_path: Optional[str] = None) -> None:
        self.profile_path = profile_path
        self._call_count: int = 0  # tracks how many engine calls have been made

        # Load profile at init (raises on missing/invalid profiles —
        # TB-TK-024 fail-fast).
        self.profile = load_profile(profile_path)

        # --- configuration (TB-TK-011) -----------------------------------
        # Prefer an explicit config; otherwise derive from the profile's
        # ``pipeline`` section, then fall back to defaults.
        pipeline_cfg = self.profile.get("pipeline") or {}
        if not isinstance(pipeline_cfg, dict):
            pipeline_cfg = {}
        profile_timeout = pipeline_cfg.get("timeout", _DEFAULT_TIMEOUT)
        profile_max_retries = pipeline_cfg.get("maxRetries",
                                               pipeline_cfg.get("max_retries",
                                                                _DEFAULT_MAX_RETRIES))
        profile_engine_url = (pipeline_cfg.get("engineUrl")
                              or pipeline_cfg.get("engine_url")
                              or os.environ.get("ENGINE_URL")
                              or _DEFAULT_ENGINE_URL)
        # Engine key for inference routes. The engine rejects unauthenticated
        # chat/completions calls; every other route stays open. The profile is
        # authoritative, and ENGINE_KEY is honoured so a freshly built profile
        # does not fail with 401s.
        self._engine_key = str(pipeline_cfg.get("engineKey")
                               or pipeline_cfg.get("engine_key")
                               or os.environ.get("ENGINE_KEY")
                               or "")

        # Stage configs carry pinnedModels / useDeck. They were normalized by
        # _normalize_stages but never stored, so pinning a model in a profile
        # had no effect: every stage drew from the deck regardless.
        self.stages = _normalize_stages(self.profile.get("stages"))

        if config is not None:
            self.config = config
        else:
            self.config = SDKConfig(
                engine_url=profile_engine_url,
                timeout=profile_timeout,
                max_retries=profile_max_retries,
            )

        # --- normalized gateway models (TB-TK-017, TB-TK-024) -----------
        # Use gateway-based rotation for fair distribution
        gateway_models = _normalize_gateway_models(self.profile)
        
        if not gateway_models:
            # Try to fall back to the deck directly if no gateways found
            deck = self.profile.get("deck", [])
            if isinstance(deck, list):
                # Extract model IDs from deck items and wrap as a single
                # "default" gateway so the GatewayDeckRotator doesn't choke.
                fallback_models = []
                for item in deck:
                    if isinstance(item, str):
                        fallback_models.append(item)
                    elif isinstance(item, dict) and "modelId" in item:
                        fallback_models.append(item["modelId"])
                gateway_models = [{
                    "gateway": "default",
                    "models": [{"id": m, "name": m} for m in fallback_models],
                }]
        
        if not gateway_models:
            raise SDKError(
                "Profile must contain at least one model",
                error_code="INVALID_PROFILE",
            )

        # Initialize rotator with gateway models
        self.rotator = GatewayDeckRotator(gateway_models)
        _pins = {
            name: cfg.get("pinnedModels") or []
            for name, cfg in (self.stages or {}).items()
        }
        self.rotator.set_stage_pins(_pins)

    # ------------------------------------------------------------------ engine

    def _auth_headers(self) -> dict[str, str]:
        """Request headers for the engine, including the key when configured."""
        headers = {"Content-Type": "application/json"}
        key = getattr(self, "_engine_key", "")
        if key:
            headers["X-Peacock-Key"] = key
        return headers

    def _call_engine(
        self,
        model: str,
        messages: list[dict[str, str]],
        timeout: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        temperature: Optional[float] = None,
        pea_session: Optional[str] = None,
        stage_name: str = "",
    ) -> dict[str, Any]:
        """Make an HTTP POST to the engine's ``/v1/chat/completions`` endpoint.

        Builds a conformant ``chat.completions`` JSON payload, serialises it,
        POSTs to ``{engine_url}/v1/chat/completions``, and returns the parsed
        JSON response as a plain ``dict``.

        **Error handling**

        | Condition | Raised |
        |---|---|
        | Connection refused / DNS failure | ``SDKError(error_code="CONNECTION_ERROR")`` |
        | Timeout (exceeds *timeout*) | ``SDKError(error_code="TIMEOUT")`` |
        | Non-2xx HTTP status | ``SDKError(error_code="HTTP_<status>")`` |
        | Non-JSON response body | ``SDKError(error_code="PARSE_ERROR")`` |

        Args:
            model: Model identifier (drawn from the deck or pinned stage).
            messages: List of message dicts with ``role`` and ``content`` keys.
            timeout: Per-call timeout in seconds.  Falls back to
                ``self.config.timeout`` when ``None``.
            tools: Optional list of tool definitions to pass through.
            tool_choice: Optional tool-choice directive to pass through.
            temperature: Optional sampling temperature to pass through.

        Returns:
            The raw JSON response from the engine as a ``dict``.

        Raises:
            SDKError: On any network, HTTP, or parsing failure.
            ValueError: If *messages* is empty.
        """
        # --- input guard --------------------------------------------------
        if not messages:
            raise ValueError("messages must contain at least one entry")

        # --- build payload ------------------------------------------------
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        # Attach optional fields only when provided.
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if pea_session is not None:
            payload["pea_session"] = pea_session
            # The engine labels the PEA file with the stage name, so it can be
            # sorted per session. The engine reads it from the request body.
            payload["stage"] = stage_name

        body = json.dumps(payload).encode("utf-8")
        url = f"{self.config.engine_url}/v1/chat/completions"
        effective_timeout = timeout if timeout is not None else self.config.timeout

        # --- request ------------------------------------------------------
        req = urllib.request.Request(
            url,
            data=body,
            headers=self._auth_headers(),
            method="POST",
        )

        self._call_count += 1

        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                raw_bytes = resp.read()
                try:
                    return json.loads(raw_bytes.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise SDKError(
                        f"Engine returned non-JSON response: {exc}",
                        error_code="PARSE_ERROR",
                    ) from exc

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SDKError(
                f"HTTP {exc.code}: {detail}",
                error_code=f"HTTP_{exc.code}",
            ) from exc

        except urllib.error.URLError as exc:
            reason_str = str(exc.reason)
            if isinstance(exc.reason, urllib.error.HTTPError):
                # Re-wrapped HTTPError inside URLError.
                http_err: urllib.error.HTTPError = exc.reason
                raise SDKError(
                    f"HTTP {http_err.code}: {http_err.read().decode()}",
                    error_code=f"HTTP_{http_err.code}",
                ) from exc
            if "timed out" in reason_str.lower():
                raise SDKError(
                    f"Connection timed out after {effective_timeout}s",
                    error_code="TIMEOUT",
                ) from exc
            raise SDKError(
                f"Connection error: {reason_str}",
                error_code="CONNECTION_ERROR",
            ) from exc

        except TimeoutError as exc:
            raise SDKError(
                f"Request timed out after {effective_timeout}s",
                error_code="TIMEOUT",
            ) from exc

    # ------------------------------------------------------------------ stage execution

    def run_stage(self, stage_name: str,
                  messages: list[dict[str, str]],
                  timeout: Optional[int] = None,
                  tools: Optional[list[dict[str, Any]]] = None,
                  tool_choice: Optional[str | dict[str, Any]] = None,
                  temperature: Optional[float] = None,
                  pea_session: Optional[str] = None) -> dict[str, Any]:
        """Run a specific stage using the configured model from the deck.

        This method:
        1. Gets the appropriate model for the stage from the rotator
        2. Makes an engine call with that model
        3. Retries on transient failures (connection errors, timeouts, 5xx),
           rotating models on each attempt
        4. Returns the response

        Args:
            stage_name: The name of the stage to run
            messages: List of message dicts with ``role`` and ``content`` keys.
            timeout: Per-call timeout in seconds.  Falls back to
                ``self.config.timeout`` when ``None``.
            tools: Optional list of tool definitions to pass through.
            tool_choice: Optional tool-choice directive to pass through.
            temperature: Optional sampling temperature to pass through.
            pea_session: Optional PEA session name — injected into the engine
                request so the engine writes forensic PEA files under a named
                session directory.

        Returns:
            The raw JSON response from the engine as a ``dict``.

        Raises:
            StageError: When all retries are exhausted. The exception contains
                the full retry history, every model that was attempted, and
                the root-cause exception.
            ValueError: If *messages* is empty.
        """
        # Get the appropriate model for this stage
        model_info = self.rotator.next_model(stage_name)
        model = model_info["model_id"]
        gateway = model_info["gateway"]

        # Execute the engine call with the determined model
        # Implement retry logic with model rotation
        max_retries = self.config.max_retries
        last_error = None

        # Keep track of attempted models for StageError reporting
        retry_history: list[dict[str, str]] = []
        failed_models: set[str] = set()
        failed_models.add(model)

        for attempt in range(max_retries + 1):
            try:
                result = self._call_engine(model, messages, timeout,
                                           tools, tool_choice, temperature,
                                           pea_session, stage_name)
                # Success — report it and reset this model's failure counter
                self.rotator.report_success(gateway, model)
                return result
            except SDKError as e:
                # Report failure to the rotator's circuit breaker
                self.rotator.report_failure(gateway, model)

                # Check if we should retry based on error type
                should_retry = (
                    e.error_code in ("CONNECTION_ERROR", "TIMEOUT") or 
                    (e.error_code and e.error_code.startswith("HTTP_5"))
                )
                
                if not should_retry or attempt >= max_retries:
                    # Should not retry — build StageError with full diagnostic info
                    failed_models.add(model)
                    retry_history.append({
                        "model": model,
                        "error": e.error_code or "UNKNOWN",
                        "message": str(e),
                    })
                    raise StageError(
                        stage=stage_name,
                        message=f"All retries exhausted for stage '{stage_name}'",
                        retry_history=retry_history,
                        failed_models=failed_models,
                        root_cause=e,
                        error_code=e.error_code or "STAGE_FAILURE",
                        details={
                            "attempt_number": attempt + 1,
                            "max_retries": max_retries,
                        },
                    ) from e
                
                # Rotate to next model in deck and try again
                failed_models.add(model)
                retry_history.append({
                    "model": model,
                    "error": e.error_code or "UNKNOWN",
                    "message": str(e),
                })
                model_info = self.rotator.next_model(stage_name)
                model = model_info["model_id"]
                gateway = model_info["gateway"]
                last_error = e
        
        # Defensive fallback — should never be reached due to the raise above.
        raise StageError(
            stage=stage_name,
            message=f"Unknown error in retry loop for stage '{stage_name}'",
            retry_history=retry_history,
            failed_models=failed_models,
            root_cause=last_error,
            error_code="STAGE_FAILURE",
        )

    def _normalize_response(self, raw_response: Dict[str, Any], model: str, duration_ms: float) -> NormalizedResponse:
        """Normalize raw engine response to canonical format.
        
        Args:
            raw_response: Raw JSON response from the engine
            model: The model that was used
            duration_ms: Duration of the call in milliseconds
            
        Returns:
            NormalizedResponse object with content, model, usage, and latency
        """
        # Extract content from response
        content = ""
        if "choices" in raw_response and raw_response["choices"]:
            # Handle choices format (standard OpenAI format)
            content = raw_response["choices"][0]["message"].get("content", "")
        elif "content" in raw_response:
            # Handle direct content format
            content = raw_response.get("content", "")
            
        # Extract usage data
        usage = raw_response.get("usage", {})
        
        # Create normalized response with all data preserved
        return NormalizedResponse(
            content=content,
            model=model,
            usage=usage,
            duration_ms=duration_ms
        )

    # ------------------------------------------------------------------ introspection

    @property
    def call_count(self) -> int:
        """Return the total number of engine calls made by this SDK instance."""
        return self._call_count

    @property
    def profile_info(self) -> dict[str, Any]:
        """Return basic information about the loaded profile."""
        return {
            "gateway_info": self.rotator.get_gateway_info(),
            "profile_path": self.profile_path
        }
