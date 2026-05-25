"""
LLM debug logging helpers.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable


def create_llm_debug_logger(step: str, sample_id: str, call_name: str) -> Optional[Callable[[Dict[str, Any]], None]]:
    """
    Create a per-call debug logger callback for LLMClient.call_with_retry.

    Logging is enabled only when PIPELINE_LLM_DEBUG_ROOT is configured.
    """
    debug_root = os.getenv("PIPELINE_LLM_DEBUG_ROOT")
    if not debug_root:
        return None

    step_run_id = os.getenv("PIPELINE_STEP_RUN_ID", "manual")
    output_dir = Path(debug_root) / f"step_{step}" / step_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    def _logger(payload: Dict[str, Any]) -> None:
        attempt = int(payload.get("attempt", 0) or 0)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        output_file = output_dir / f"{sample_id}__{call_name}__attempt_{attempt:02d}__{ts}.json"

        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "step_run_id": step_run_id,
            "sample_id": sample_id,
            "call_name": call_name,
            **payload
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    return _logger

