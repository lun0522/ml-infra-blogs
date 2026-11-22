"""Lightweight ONNX multi-label 5-pillar classifier for filtering and tagging blog posts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

PILLARS: tuple[str, ...] = ("inference", "training", "hardware", "systems", "agents")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "post_classifier"


def format_post_text(title: str, summary: str = "") -> str:
    """Canonical text representation shared between training and inference."""
    clean_title = " ".join((title or "").split())
    clean_summary = " ".join((summary or "").split())
    if clean_title and clean_summary:
        return f"{clean_title}\n{clean_summary}"
    return clean_title or clean_summary


@dataclass(frozen=True)
class ClassificationResult:
    """Predicted topic pillars (empty list => Filtered/drop) and per-pillar probabilities."""

    topics: list[str]
    probabilities: dict[str, float]

    @property
    def label(self) -> str:
        return "infra" if self.topics else "drop"


def decode_pillar_probabilities(
    probs: np.ndarray,
    pillars: Sequence[str],
    keep_threshold: float,
    pillar_thresholds: Sequence[float],
) -> list[list[str]]:
    """Decode multi-label probabilities into ordered pillar tags ([] when max_p < keep_threshold)."""
    decoded: list[list[str]] = []
    for row in probs:
        if float(np.max(row)) < keep_threshold:
            decoded.append([])
            continue
        selected = [
            (float(row[idx]), pillars[idx])
            for idx in range(len(pillars))
            if float(row[idx]) >= float(pillar_thresholds[idx])
        ]
        if not selected:
            # Normalize by each pillar's calibrated threshold so high-prior classes don't win raw argmax.
            ratios = np.asarray(row, dtype=np.float64) / np.maximum(
                np.asarray(pillar_thresholds, dtype=np.float64), 1e-6
            )
            best_idx = int(np.argmax(ratios))
            decoded.append([pillars[best_idx]])
        else:
            selected.sort(key=lambda item: item[0], reverse=True)
            decoded.append([name for _score, name in selected])
    return decoded


class OnnxPostClassifier:
    """CPU-friendly ONNX multi-label classifier backed by a fine-tuned transformer encoder."""

    def __init__(self, model_dir: Path) -> None:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.pillars: tuple[str, ...] = tuple(config.get("pillars", PILLARS))
        self.max_length: int = int(config.get("max_length", 512))
        self.keep_threshold: float = float(config.get("keep_threshold", 0.15))

        raw_thresholds = config.get("pillar_thresholds")
        if isinstance(raw_thresholds, dict):
            self.pillar_thresholds: list[float] = [
                float(raw_thresholds.get(p, 0.45)) for p in self.pillars
            ]
        else:
            default_th = float(config.get("pillar_threshold", 0.45))
            self.pillar_thresholds = [default_th for _ in self.pillars]

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(length=None)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def predict_batch(
        self,
        items: Sequence[tuple[str, str]],
        batch_size: int = 32,
    ) -> list[ClassificationResult]:
        """Predict topic pillars for a sequence of (title, summary/excerpt) pairs."""
        if not items:
            return []

        results: list[ClassificationResult] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            texts = [format_post_text(title, summary) for title, summary in chunk]
            encoded = self.tokenizer.encode_batch(texts)

            input_ids = np.array([enc.ids for enc in encoded], dtype=np.int64)
            attention_mask = np.array([enc.attention_mask for enc in encoded], dtype=np.int64)

            logits = np.asarray(
                self.session.run(
                    None,
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                )[0],
                dtype=np.float64,
            )
            probs = _sigmoid(logits)
            decoded_topics = decode_pillar_probabilities(
                probs,
                pillars=self.pillars,
                keep_threshold=self.keep_threshold,
                pillar_thresholds=self.pillar_thresholds,
            )
            for row, topics in zip(probs, decoded_topics, strict=True):
                prob_map = {pillar: float(row[idx]) for idx, pillar in enumerate(self.pillars)}
                results.append(ClassificationResult(topics=topics, probabilities=prob_map))

        return results


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_default_classifier(model_dir: Path = DEFAULT_MODEL_DIR) -> OnnxPostClassifier | None:
    """Load the committed ONNX post classifier if present, otherwise return None."""
    if not (model_dir / "model.onnx").exists():
        return None
    return OnnxPostClassifier(model_dir)
