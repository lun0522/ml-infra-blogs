"""Unit tests for the multi-label 5-pillar post classifier decoding."""

from __future__ import annotations

import unittest

import numpy as np

from app.classifier import PILLARS, ClassificationResult, decode_pillar_probabilities


class ClassifierTests(unittest.TestCase):
    def test_decode_pillar_probabilities_zero_or_multi_label(self) -> None:
        probs = np.array(
            [
                # All scores below keep_threshold => [] (Filtered / drop)
                [0.05, 0.03, 0.08, 0.02, 0.04],
                # Exceeds keep_threshold (0.15) via max, falls back to strongest pillar
                [0.22, 0.05, 0.34, 0.08, 0.10],
                # Multiple pillars above per-pillar thresholds => ordered descending
                [0.82, 0.55, 0.12, 0.91, 0.20],
            ],
            dtype=np.float64,
        )
        decoded = decode_pillar_probabilities(
            probs,
            pillars=PILLARS,
            keep_threshold=0.15,
            pillar_thresholds=[0.45] * len(PILLARS),
        )
        self.assertEqual(decoded[0], [])
        self.assertEqual(decoded[1], ["hardware"])
        self.assertEqual(decoded[2], ["systems", "inference", "training"])

        # Fallback respects calibrated threshold ratios instead of raw score bias.
        decoded_calibrated = decode_pillar_probabilities(
            np.array([[0.50, 0.10, 0.10, 0.10, 0.45]], dtype=np.float64),
            pillars=PILLARS,
            keep_threshold=0.15,
            pillar_thresholds=[0.78, 0.60, 0.75, 0.55, 0.55],
        )
        self.assertEqual(decoded_calibrated[0], ["agents"])

        res_drop = ClassificationResult(topics=decoded[0], probabilities={})
        res_keep = ClassificationResult(topics=decoded[1], probabilities={})
        self.assertEqual(res_drop.label, "drop")
        self.assertEqual(res_keep.label, "infra")


if __name__ == "__main__":
    unittest.main()
