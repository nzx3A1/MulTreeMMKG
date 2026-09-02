"""VLM request options scoped to the table-embedded hybrid extractor."""
from __future__ import annotations

import os


TABLE_EMBEDDED_HYBRID_VLM_TIMEOUT_ENV = "STRATIGRAPHIC_TABLE_VLM_TIMEOUT_SECS"
DEFAULT_TABLE_EMBEDDED_HYBRID_VLM_TIMEOUT_SECS = 600.0


def table_embedded_hybrid_vlm_timeout_secs() -> float:
    """Return the dedicated timeout for long table-hybrid VLM requests."""

    timeout = float(
        os.getenv(
            TABLE_EMBEDDED_HYBRID_VLM_TIMEOUT_ENV,
            str(DEFAULT_TABLE_EMBEDDED_HYBRID_VLM_TIMEOUT_SECS),
        )
    )
    if timeout <= 0:
        raise ValueError(f"{TABLE_EMBEDDED_HYBRID_VLM_TIMEOUT_ENV} must be greater than 0")
    return timeout
