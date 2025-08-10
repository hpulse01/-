from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class BrandColors:
    quantum_blue: str = "#00F5FF"
    neural_red: str = "#FF6B6B"
    mystic_purple: str = "#764BA2"
    predict_green: str = "#4ECDC4"
    deep_space_black: str = "#0A0A1A"


@dataclass
class AppSettings:
    version: str = "1.1"
    seed: int = 42
    data_dir: Path = Path.cwd() / "data"
    reports_dir: Path = Path.cwd() / "reports"
    logs_dir: Path = Path.cwd() / "runs"
    vi: BrandColors = BrandColors()

    def as_dict(self) -> Dict[str, str]:
        return {
            "version": self.version,
            "seed": str(self.seed),
            "data_dir": str(self.data_dir),
            "reports_dir": str(self.reports_dir),
            "logs_dir": str(self.logs_dir),
        }


SETTINGS = AppSettings()
SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.reports_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.logs_dir.mkdir(parents=True, exist_ok=True)