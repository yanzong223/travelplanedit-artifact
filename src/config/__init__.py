"""
Configuration module for TPE system.

Provides environment-based settings management and validation.
"""

from .settings import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
