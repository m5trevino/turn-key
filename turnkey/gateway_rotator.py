"""
Gateway-Based Deck Rotation for TurnKey SDK

This module implements a gateway-focused rotation system that ensures fair distribution
across different infrastructure providers (gateways) with varying numbers of models.
"""

import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GatewayModel:
    """Represents a model within a specific gateway."""
    gateway: str
    model_id: str
    model_name: str


class GatewayDeckRotator:
    """
    Gateway-based deck rotator that rotates through models by gateway order,
    handling varying model counts per gateway.
    
    This ensures fair distribution across different infrastructure providers
    regardless of how many models they provide.
    """
    
    def __init__(self, gateway_models: List[Dict[str, Any]]):
        """
        Initialize gateway rotator with models grouped by gateway.
        
        Args:
            gateway_models: List of dictionaries containing:
                - gateway: Gateway name (e.g., "OpenRouter", "Ollama")
                - models: List of model dictionaries with 'id' and 'name' keys
        """
        # Group models by gateway
        self._gateways: Dict[str, List[Dict[str, Any]]] = {}
        self._gateway_order: List[str] = []
        
        # Track current positions for each gateway
        self._positions: Dict[str, int] = {}
        
        # Track maximum models per gateway for proper wrapping
        self._max_models_per_gateway: Dict[str, int] = {}
        
        # Failure tracking for circuit breaking
        self._failures: Dict[str, Dict[str, int]] = {}
        self._shutdown_models: set[str] = set()

        # Flat deck: every model in every gateway, in profile order. The
        # cursor is the only thing that decides which model runs next.
        self._flat: List[tuple[str, Dict[str, Any]]] = []
        self._cursor: int = 0
        # stage name -> list of pinned model ids, in preference order.
        self._stage_pins: Dict[str, List[str]] = {}
        # stage name -> index of the next pinned model to serve.
        self._pin_pos: Dict[str, int] = {}

        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize the rotation setup
        self._setup_gateways(gateway_models)
        
        # Validate that we have at least one model
        if not any(len(models) > 0 for models in self._gateways.values()):
            raise ValueError("Must have at least one model across all gateways")
    
    def _setup_gateways(self, gateway_models: List[Dict[str, Any]]) -> None:
        """
        Set up the gateway structure and initialize tracking.
        
        Args:
            gateway_models: List of gateway model configurations
        """
        for gateway_config in gateway_models:
            gateway_name = gateway_config["gateway"]
            models = gateway_config["models"]
            
            # Store models for this gateway
            self._gateways[gateway_name] = models
            self._gateway_order.append(gateway_name)
            
            # Initialize position tracking
            self._positions[gateway_name] = 0
            self._failures[gateway_name] = {m["id"]: 0 for m in models}
            
            # Store max models for this gateway
            self._max_models_per_gateway[gateway_name] = len(models)

            for model in models:
                self._flat.append((gateway_name, model))
            
            # Ensure we have at least one model
            if not models:
                raise ValueError(f"Gateway '{gateway_name}' must have at least one model")

    def report_failure(self, gateway: str, model_id: str) -> None:
        """Report a failure for a specific model."""
        with self._lock:
            if gateway not in self._failures:
                return
            
            self._failures[gateway][model_id] = self._failures[gateway].get(model_id, 0) + 1
            if self._failures[gateway][model_id] >= 3:
                model_key = f"{gateway}:{model_id}"
                if model_key not in self._shutdown_models:
                    self._shutdown_models.add(model_key)
                    print(f"[DECK ROTATOR] Model {model_id} from {gateway} shut down after 3 consecutive failures.")

    def report_success(self, gateway: str, model_id: str) -> None:
        """Report a success for a specific model, resetting its failure counter."""
        with self._lock:
            if gateway in self._failures and model_id in self._failures[gateway]:
                self._failures[gateway][model_id] = 0
                if f"{gateway}:{model_id}" in self._shutdown_models:
                    self._shutdown_models.remove(f"{gateway}:{model_id}")
    
    def set_stage_pins(self, pins: Dict[str, List[str]]) -> None:
        """Record which models a stage is allowed to use.

        A pinned stage draws only from its own list. Without this the deck
        decided everything and ``pinnedModels`` in a profile was inert: a
        stage could be pinned to one model and still be served another.
        """
        with self._lock:
            self._stage_pins = {
                k: [m for m in (v or []) if m] for k, v in (pins or {}).items()
            }

    def _pinned_for(self, stage: str,
                    advance: bool = True) -> Dict[str, Any] | None:
        """Resolve the next pinned model for a stage.

        A stage pinned to several models rotates among them in listed order;
        a single pin is served every time. Returning the first entry forever
        would make a five-model rotation behave like a one-model pin, so the
        position advances once per served call.

        ``advance=False`` peeks for display without consuming a slot, which
        keeps the announced model equal to the one actually served.

        Returns None when the stage has no pins, so callers fall through to
        normal rotation.
        """
        wanted = self._stage_pins.get(stage) or []
        if not wanted:
            return None

        start = self._pin_pos.get(stage, 0)
        for offset in range(len(wanted)):
            pos = (start + offset) % len(wanted)
            model_id = wanted[pos]
            for gateway, model in self._flat:
                if model["id"] != model_id:
                    continue
                if f"{gateway}:{model_id}" in self._shutdown_models:
                    break
                if advance:
                    self._pin_pos[stage] = (pos + 1) % len(wanted)
                return {
                    "gateway": gateway,
                    "model_id": model_id,
                    "model_name": model.get("name", model_id),
                    "model_info": model,
                }
        # Every pin is shut down: fall back rather than deadlock the stage.
        return None

    def next_model(self, stage: str) -> Dict[str, Any]:
        """
        Get the next model for a stage based on gateway rotation, skipping shutdown models.
        
        Args:
            stage: The stage name.
            
        Returns:
            Dictionary with gateway and model information
            
        Raises:
            RuntimeError: If no models are available or all models are shut down
        """
        with self._lock:
            pinned = self._pinned_for(stage)
            if pinned is not None:
                return pinned
            return self._next_flat()
    
    def _try_get_valid_model_from_gateway(self, gateway_name: str) -> Optional[Dict[str, Any]]:
        """Try to get a valid (non-shutdown) model from a gateway."""
        models = self._gateways[gateway_name]
        start_pos = self._positions[gateway_name]
        
        # Check all models in this gateway to see if one is valid
        for i in range(len(models)):
            pos = (start_pos + i) % len(models)
            model_info = models[pos]
            model_id = model_info["id"]
            
            if f"{gateway_name}:{model_id}" not in self._shutdown_models:
                # We found a valid model, update the position to point here
                self._positions[gateway_name] = pos
                return {
                    "gateway": gateway_name,
                    "model_id": model_id,
                    "model_name": model_info.get("name", model_id),
                    "model_info": model_info
                }
        
        return None
    
    def _get_next_gateway(self) -> str:
        """
        Determine which gateway should be used next.

        Per-gateway positions are deck indices that wrap back to zero, so they
        cannot express how much of a gateway has been consumed. Comparing them
        makes a one-model gateway (whose index stays at 0) look permanently
        least-used, and it then supplies almost all traffic.

        Fairness therefore comes from the flat cursor below: every model in the
        deck, regardless of gateway, gets exactly one turn before any model
        repeats.

        Returns:
            Gateway name for next call
        """
        min_position = min(self._positions.values())

        for gateway_name in self._gateway_order:
            if self._positions[gateway_name] == min_position:
                return gateway_name

        return self._gateway_order[0]

    def _next_flat(self) -> Dict[str, Any]:
        """Return the next model in deck order, skipping shut-down cards.

        This is the rotation that actually decides traffic. It walks one flat
        list of every model across every gateway, so a gateway with thirteen
        cards is used thirteen times as often as a gateway with one — which is
        what "all 36 models" means in practice.

        Raises:
            RuntimeError: when every card has been shut down.
        """
        total = self._flat_count()
        for _ in range(total):
            gateway, model_info = self._flat[self._cursor % total]
            self._cursor += 1
            model_id = model_info["id"]
            if f"{gateway}:{model_id}" in self._shutdown_models:
                continue
            return {
                "gateway": gateway,
                "model_id": model_id,
                "model_name": model_info.get("name", model_id),
                "model_info": model_info,
            }
        raise RuntimeError(
            "All models have been shut down due to consecutive failures.")

    def _flat_count(self) -> int:
        return len(self._flat)

    def peek_model(self, stage: str = "") -> Dict[str, Any]:
        """Return the model the next call would use, without consuming it.

        Display code needs to name the model while a stage is running, and
        the model is only known after the rotator hands one out. Peeking
        keeps the announced model and the served model in agreement: an
        operator reading "nemotron" on screen must not receive a different
        model because the peek advanced the deck.
        """
        with self._lock:
            pinned = self._pinned_for(stage, advance=False) if stage else None
            if pinned is not None:
                return pinned
            saved = self._cursor
            try:
                return self._next_flat()
            finally:
                self._cursor = saved
    
    def _get_model_from_gateway(self, gateway_name: str) -> Dict[str, Any]:
        """
        Get the appropriate model from a gateway based on current position.
        
        Args:
            gateway_name: Name of the gateway
            
        Returns:
            Model information dictionary
        """
        models = self._gateways[gateway_name]
        position = self._positions[gateway_name]
        
        # Handle circular rotation within gateway
        model_index = position % len(models)
        model_info = models[model_index]
        
        # Return model with gateway information
        return {
            "gateway": gateway_name,
            "model_id": model_info["id"],
            "model_name": model_info.get("name", model_info["id"]),
            "model_info": model_info
        }
    
    def _advance_gateway_position(self, gateway_name: str) -> None:
        """
        Advance the position for a gateway, wrapping appropriately.
        
        Args:
            gateway_name: Name of the gateway to advance
        """
        self._positions[gateway_name] += 1
    
    def get_gateway_info(self) -> Dict[str, Any]:
        """
        Get current rotation information.
        
        Returns:
            Dictionary with current positions and gateway info
        """
        with self._lock:
            return {
                "gateways": self._gateway_order,
                "current_positions": self._positions.copy(),
                "models_per_gateway": {
                    gw: len(models) for gw, models in self._gateways.items()
                }
            }
    
    def reset_positions(self) -> None:
        """
        Reset all gateway positions to start from the beginning.
        """
        with self._lock:
            for gateway in self._positions:
                self._positions[gateway] = 0
    
    @property
    def total_models(self) -> int:
        """Get total number of models across all gateways."""
        return sum(len(models) for models in self._gateways.values())
    
    @property
    def gateway_count(self) -> int:
        """Get number of gateways."""
        return len(self._gateways)


def example_usage():
    """
    Example of how to use the gateway-based deck rotator.
    """
    print("=== Gateway-Based Deck Rotator Demo ===")
    
    # Example gateway structure with varying model counts
    gateway_config = [
        {
            "gateway": "OpenRouter",
            "models": [
                {"id": "gpt-4", "name": "GPT-4"},
                {"id": "claude-3", "name": "Claude-3"},
                {"id": "gemini-1.5", "name": "Gemini 1.5"},
                {"id": "llama-2-70b", "name": "LLaMA-2 70B"}
            ]
        },
        {
            "gateway": "Ollama",
            "models": [
                {"id": "llama3", "name": "LLaMA 3"},
                {"id": "mistral", "name": "Mistral"},
                {"id": "phi3", "name": "Phi-3"}
            ]
        },
        {
            "gateway": "Novita",
            "models": [
                {"id": "mixtral-8x7b", "name": "Mixtral 8x7B"},
                {"id": "qwen-72b", "name": "Qwen 72B"}
            ]
        },
        {
            "gateway": "Groq",
            "models": [
                {"id": "llama3-70b", "name": "LLaMA 3 70B"},
                {"id": "llama3-8b", "name": "LLaMA 3 8B"},
                {"id": "mixtral-8x7b", "name": "Mixtral 8x7B"},
                {"id": "gemma-2b", "name": "Gemma 2B"}
            ]
        }
    ]
    
    # Create rotator
    rotator = GatewayDeckRotator(gateway_config)
    
    print(f"Total models: {rotator.total_models}")
    print(f"Total gateways: {rotator.gateway_count}")
    print()
    
    # Demonstrate rotation
    print("Rotation sequence:")
    for i in range(15):  # Show first 15 model selections
        model = rotator.next_model(f"stage_{i%3}")
        print(f"{i+1:2d}. {model['gateway']}: {model['model_name']} ({model['model_id']})")
    
    print()
    print("Current state:")
    info = rotator.get_gateway_info()
    for gateway, position in info["current_positions"].items():
        models = info["models_per_gateway"][gateway]
        print(f"  {gateway}: position {position} of {models}")


if __name__ == "__main__":
    example_usage()