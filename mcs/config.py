"""Runtime configuration, resolved from environment variables.

See SPEC.md section 3 for the full table. Flags handled by run.sh / cli.py are
exported into these same env vars before pytest runs, so this is the single
source of truth.
"""

import os
import sys
from dataclasses import dataclass


def _env(*names, default=None):
    """Return the first set, non-empty environment variable among names."""
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return default


_warned_deprecated_key = False


def _api_key() -> str:
    """Bearer token: MCS_API_KEY, then SWISSAI_RESEARCH_API_KEY. The old
    CSCS_SERVING_API name still works but warns (once) that it's deprecated."""
    global _warned_deprecated_key
    key = _env("MCS_API_KEY", "SWISSAI_RESEARCH_API_KEY")
    if key:
        return key
    key = _env("CSCS_SERVING_API", default="")
    if key and not _warned_deprecated_key:
        _warned_deprecated_key = True
        print(
            "warning: CSCS_SERVING_API is deprecated; "
            "set SWISSAI_RESEARCH_API_KEY instead",
            file=sys.stderr,
        )
    return key


@dataclass(frozen=True)
class Config:
    api_base: str
    api_key: str
    model: str
    timeout: float
    rate_limit: float

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            api_base=_env(
                "MCS_API_BASE", default="https://api.swissai.svc.cscs.ch/v1"
            ).rstrip("/"),
            api_key=_api_key(),
            model=_env(
                "MCS_MODEL", default="CSCS-Inference/swiss-ai/Apertus-8B-Instruct-2509"
            ),
            timeout=float(_env("MCS_TIMEOUT", default="120")),
            # Requests per MINUTE; 0 = unrestricted (the default). Some serving
            # endpoints cap at 15 req/min per user -- pass --rate-limit (or set
            # MCS_RATE_LIMIT) to stay under such caps.
            rate_limit=float(_env("MCS_RATE_LIMIT", default="0")),
        )
