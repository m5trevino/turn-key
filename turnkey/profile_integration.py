"""
Profile Loading Integration - TB-TK-017

This module handles the integration of profile loading into Dendritic's pipeline.
"""

import os
from pathlib import Path
from typing import Optional
from turnkey.loader import load_profile
from turnkey.types import ProfileValidationError


def load_turnkey_profile(profile_path: Optional[str] = None) -> dict:
    """
    Load a TurnKey profile from the standard location.
    
    This function integrates with Dendritic's pipeline to load profiles
    from the ~/dendritic/turn-key-profiles/ directory.
    
    Args:
        profile_path: Optional explicit path to profile file. If None,
                     defaults to loading from the standard location.
                     
    Returns:
        Dictionary containing the loaded profile data
                     
    Raises:
        FileNotFoundError: If profile file doesn't exist
        ProfileValidationError: If profile is invalid
        ValueError: If profile contains invalid JSON
    """
    if profile_path is None:
        # Default location for Dendritic profiles
        profile_path = os.path.expanduser("~/dendritic/turn-key-profiles/")
        # Look for the first JSON file in the directory
        profile_dir = Path(profile_path)
        if profile_dir.exists():
            profile_files = list(profile_dir.glob("*.json"))
            if profile_files:
                profile_path = str(profile_files[0])
            else:
                raise FileNotFoundError("No profile files found in turn-key-profiles directory")
        else:
            raise FileNotFoundError(f"Profile directory not found: {profile_path}")
    
    # Load and validate the profile
    return load_profile(profile_path)


def initialize_sdk_with_profile(profile_path: Optional[str] = None) -> 'SDK':
    """
    Initialize the SDK with a profile loaded from the standard location.
    
    This function combines profile loading with SDK initialization.
    
    Args:
        profile_path: Optional explicit path to profile file
        
    Returns:
        Initialized SDK instance
    """
    # Load the profile
    profile = load_turnkey_profile(profile_path)
    
    # Create SDK with the loaded profile
    from turnkey.sdk import SDK
    from turnkey.types import SDKConfig
    
    # Extract pipeline configuration if present
    pipeline_cfg = profile.get("pipeline", {})
    
    # Engine URL resolution order:
    #   1. profile pipeline.engineUrl / engine_url
    #   2. ENGINE_URL from the environment (loaded from .env by the SDK)
    #   3. the public default, only as a last resort
    engine_url = (pipeline_cfg.get("engineUrl")
                  or pipeline_cfg.get("engine_url")
                  or os.environ.get("ENGINE_URL")
                  or "http://engine.save-aichats.com:3099")

    # Create SDK configuration
    sdk_config = SDKConfig(
        engine_url=engine_url,
        timeout=pipeline_cfg.get("timeout", 180),
        max_retries=pipeline_cfg.get("maxRetries", pipeline_cfg.get("max_retries", 10))
    )
    
    # Initialize and return SDK instance
    return SDK(config=sdk_config, profile_path=profile_path)