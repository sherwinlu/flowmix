"""Scoring profile support for FlowMix.

Profiles are TOML files that let the same transition engine behave differently
for EDM/workout mixes, vocal trance sets, lounge/background journeys, jazz/blues,
emotional ballad sets, or cinematic continuous-listening mixes without hard-coding
style preferences into the DSP code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import sys

import tomllib  # Python 3.11+


@dataclass
class ScoringProfile:
    name: str = "edm"
    description: str = "Beat-aware continuous mix profile."
    weights: Dict[str, float] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    limits: Dict[str, float] = field(default_factory=dict)

    def weight(self, key: str, default: float) -> float:
        try:
            return float(self.weights.get(key, default))
        except Exception:
            return default

    def pref(self, key: str, default: Any) -> Any:
        return self.preferences.get(key, default)

    def limit(self, key: str, default: float) -> float:
        try:
            return float(self.limits.get(key, default))
        except Exception:
            return default


def builtin_profile_path(name: str) -> Path:
    """Return the best available path for a built-in profile.

    Source-tree runs keep configs/ next to these scripts. Editable/installed
    runs may install the TOML files under share/flowmix/configs. This helper
    checks both locations so `pip install .` is usable without requiring users
    to copy config files manually.
    """
    filename = f"{name}.toml"
    candidates = [
        Path(__file__).resolve().parent / "configs" / filename,
        Path(__file__).resolve().parent / "share" / "flowmix" / "configs" / filename,
        Path.cwd() / "configs" / filename,
        Path(sys.prefix) / "share" / "flowmix" / "configs" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_scoring_profile(profile: str = "edm", config_path: Optional[str] = None) -> ScoringProfile:
    path = Path(config_path).expanduser() if config_path else builtin_profile_path(profile)
    if not path.exists():
        raise FileNotFoundError(f"Scoring profile not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return ScoringProfile(
        name=str(data.get("name") or profile),
        description=str(data.get("description") or ""),
        weights=dict(data.get("weights") or {}),
        preferences=dict(data.get("preferences") or {}),
        limits=dict(data.get("limits") or {}),
    )
