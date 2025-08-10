from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich import print

from .utils.settings import SETTINGS
from .data_collection.eastern import collect_eastern_features
from .data_collection.western import collect_western_features
from .quantum_engine.simulator import simulate_quantum_life
from .prediction_ai.train import run_training
from .prediction_ai.infer import run_inference
from .output_generation.report import TimelineEvent, build_timeline, plot_timeline, export_pdf, sign_and_anchor
from .output_generation.api import app

cli = typer.Typer(add_completion=False)


@cli.command()
def demo(name: str = "Demo User", birth_time: str = "1995-05-01T08:30:00+00:00", lat: float = 35.68, lon: float = 139.76):
    dt = datetime.fromisoformat(birth_time)
    east = collect_eastern_features(dt)
    west = collect_western_features(dt, lat, lon)
    ai = run_inference(SETTINGS.seed)
    quantum = simulate_quantum_life({"east": east.__dict__, "west": west.__dict__, "ai": ai}, bio={"name": name}, birth_spacetime={"birth_time": birth_time, "lat": lat, "lon": lon}, seed=SETTINGS.seed)

    events = []
    base_time = dt
    for i, p in enumerate(ai["probs"]):
        events.append(TimelineEvent(
            id=f"evt-{i}", time=(base_time if base_time.tzinfo else base_time.replace(tzinfo=timezone.utc)).isoformat(), place="Tokyo", domain="life", impact=float(p), advice="Focus on growth", confidence=float(p), error_margin=float(ai["ci"][i][1] - ai["ci"][i][0]),
        ))
        base_time = base_time.replace(year=base_time.year + 1)

    timeline = build_timeline(events)
    reports_dir = SETTINGS.reports_dir
    out_json = reports_dir / "timeline.json"
    out_html = reports_dir / "timeline.html"
    out_pdf = reports_dir / "report.pdf"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)
    plot_timeline(events, out_html)
    export_pdf(events, out_pdf)

    anchor = sign_and_anchor(timeline)
    print({"signature": anchor["signature"], "tx_hash": anchor["tx_hash"], "synthetic": anchor["synthetic"]})
    print(f"Saved: {out_json}, {out_html}, {out_pdf}")


@cli.command()
def train(epochs: int = 2, seed: int = 42):
    result = run_training(epochs=epochs, seed=seed)
    print(result)


@cli.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()