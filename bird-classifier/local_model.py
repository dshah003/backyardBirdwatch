"""Local TFHub bird classifier for fast first-pass classification."""

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_URL = "https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1"
LABELS_URL = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv"
INPUT_SIZE = (224, 224)


class LocalBirdModel:
    """TFHub bird classifier (~900 species, runs on CPU)."""

    def __init__(self) -> None:
        self._model: tf.Module | None = None
        self._labels: list[str] = []

    def load(self) -> None:
        """Load the model and labels. Call once at startup."""
        logger.info("Loading local bird model from TFHub...")
        # hub.KerasLayer handles both TF1-format Hub modules and TF2 SavedModels,
        # whereas hub.load() returns an AutoTrackable that is not directly callable
        # for TF1-format modules.
        self._model = hub.KerasLayer(MODEL_URL)
        self._load_labels()
        logger.info("Local bird model loaded (%d labels)", len(self._labels))

    def _load_labels(self) -> None:
        """Download and parse the label map CSV."""
        import requests

        try:
            response = requests.get(LABELS_URL, timeout=30)
            response.raise_for_status()
            lines = response.text.strip().split("\n")
            # CSV format: id,common_name — skip header
            self._labels = []
            for line in lines[1:]:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    self._labels.append(parts[1].strip())
                else:
                    self._labels.append("Unknown")
        except requests.RequestException:
            logger.exception("Failed to download label map")
            self._labels = []

    def classify(self, image_path: Path) -> list[tuple[str, float]]:
        """Classify a bird image.

        Returns list of (species_name, confidence) tuples, sorted by
        confidence descending.
        """
        if self._model is None:
            logger.error("Model not loaded — call load() first")
            return []

        try:
            img = Image.open(image_path).convert("RGB").resize(INPUT_SIZE)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            logits = self._model(img_array)
            probabilities = tf.nn.softmax(logits[0]).numpy()

            top_indices = np.argsort(probabilities)[::-1][:5]
            results: list[tuple[str, float]] = []
            for idx in top_indices:
                label = self._labels[idx] if idx < len(self._labels) else "Unknown"
                results.append((label, float(probabilities[idx])))

            return results

        except Exception:
            logger.exception("Local model classification failed")
            return []
