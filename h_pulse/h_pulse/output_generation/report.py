from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from ..utils.settings import SETTINGS
from ..utils.crypto_anchor import anchor_with_web3_or_synthetic


@dataclass
class TimelineEvent:
    id: str
    time: str
    place: str
    domain: str
    impact: float
    advice: str
    confidence: float
    error_margin: float
    synthetic: bool = False


def build_timeline(events: List[TimelineEvent]) -> Dict:
    return {
        "$schema": "https://schema.h-pulse/timeline.json",
        "version": SETTINGS.version,
        "events": [e.__dict__ for e in events],
    }


def plot_timeline(events: List[TimelineEvent], out_html: Path) -> Path:
    times = [e.time for e in events]
    impacts = [e.impact for e in events]
    confs = [e.confidence for e in events]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=impacts, mode="lines+markers", name="Impact", line=dict(color=SETTINGS.vi.quantum_blue)))
    fig.add_trace(go.Bar(x=times, y=confs, name="Confidence", marker=dict(color=SETTINGS.vi.predict_green)))
    fig.update_layout(template="plotly_dark", paper_bgcolor=SETTINGS.vi.deep_space_black, plot_bgcolor=SETTINGS.vi.deep_space_black)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return out_html


def export_pdf(events: List[TimelineEvent], out_pdf: Path) -> Path:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    width, height = A4
    c.setFillColorRGB(0, 0.96, 1)  # #00F5FF
    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, height - 20 * mm, "H-Pulse Quantum Prediction System")
    c.setFillColorRGB(1, 1, 1)
    y = height - 35 * mm
    c.setFont("Helvetica", 10)
    for e in events[:30]:
        c.drawString(20 * mm, y, f"{e.time} [{e.domain}] impact={e.impact:.2f} conf={e.confidence:.2f} ±{e.error_margin:.2f}")
        y -= 7 * mm
        if y < 30 * mm:
            c.setFillColorRGB(0.3, 0.3, 0.3)
            c.setFont("Helvetica-Oblique", 9)
            c.drawRightString(width - 15 * mm, 15 * mm, "Precision · Uniqueness · Irreversibility")
            c.showPage()
            c.setFillColorRGB(0, 0.96, 1)
            c.setFont("Helvetica-Bold", 18)
            c.drawString(20 * mm, height - 20 * mm, "H-Pulse Quantum Prediction System")
            c.setFillColorRGB(1, 1, 1)
            y = height - 35 * mm
    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.setFont("Helvetica-Oblique", 9)
    c.drawRightString(width - 15 * mm, 15 * mm, "Precision · Uniqueness · Irreversibility")
    c.showPage()
    c.save()
    return out_pdf


def sign_and_anchor(timeline: Dict) -> Dict:
    anchor = anchor_with_web3_or_synthetic(timeline)
    return {
        "digest": anchor.sha3_256_hex,
        "signature": anchor.signature_hex,
        "public_key": anchor.public_key_hex,
        "tx_hash": anchor.tx_hash,
        "block_number": anchor.block_number,
        "synthetic": anchor.synthetic,
        "timestamp": anchor.timestamp,
    }