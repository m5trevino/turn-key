"""
MachineTalkClient SDK Integration - TB-TK-016

This module implements the integration of the TurnKey SDK with MachineTalkClient,
replacing the old PeacockClient and ModelRotator components.
"""

from typing import Any, Dict, List, Optional
from turnkey.sdk import SDK
from turnkey.types import SDKConfig


class MachineTalkClient:
    """
    Updated MachineTalkClient that integrates with the new TurnKey SDK.
    
    This class replaces the older PeacockClient and ModelRotator implementations
    by using the new SDK for all engine interactions.
    """
    
    def __init__(self, sdk: SDK):
        """
        Initialize MachineTalkClient with SDK instance.
        
        Args:
            sdk: Instance of the TurnKey SDK to use for engine calls
        """
        self.sdk = sdk
        
    def call_stage(self, stage_name: str, messages: List[Dict[str, str]], 
                   timeout: Optional[int] = None,
                   tools: Optional[List[Dict[str, Any]]] = None,
                   tool_choice: Optional[str] = None,
                   temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Call a specific stage using the SDK's run_stage method.
        
        This replaces the old client.chat() approach with the new SDK-based approach.
        
        Args:
            stage_name: The name of the stage to run
            messages: List of message dicts with 'role' and 'content' keys
            timeout: Per-call timeout in seconds
            tools: Optional list of tool definitions to pass through
            tool_choice: Optional tool-choice directive to pass through
            temperature: Optional sampling temperature to pass through
            
        Returns:
            The raw JSON response from the engine as a dict
        """
        # Use the SDK's run_stage method instead of direct client.chat()
        return self.sdk.run_stage(
            stage_name=stage_name,
            messages=messages,
            timeout=timeout,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature
        )
        
    def __repr__(self):
        return f"MachineTalkClient(sdk={self.sdk})"