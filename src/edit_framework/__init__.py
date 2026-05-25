"""Standalone edit framework package."""

from edit_framework.base import EditFramework, EditInput, EditResult


def create_edit_framework(*args, **kwargs):
    from edit_framework.factory import create_edit_framework as _create_edit_framework

    return _create_edit_framework(*args, **kwargs)

__all__ = ["EditFramework", "EditInput", "EditResult", "create_edit_framework"]
