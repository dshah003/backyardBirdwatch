"""HuggingFace image-classification bird classifier (ViT / EfficientNet).

Works with any AutoModelForImageClassification-compatible checkpoint.
Default: chriamue/bird-species-classifier (EfficientNet-B4, ~525 species).

To try a different checkpoint set NABIRDS_MODEL in your environment, e.g.:
    NABIRDS_MODEL=dennisjooo/Birds-Classifier-EfficientNetB2

Install:
    pip install transformers torch pillow
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "chriamue/bird-species-classifier"


class NABirdsClassifier:
    """HuggingFace bird-species classifier with optional species allowlist.

    The allowlist filters predictions to the model labels that match entries
    in species_list.txt (case-insensitive substring match).  Filtered-out
    label probabilities are zeroed and the remainder is renormalised.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        species_list_path: Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._species_list_path = species_list_path
        self._model = None
        self._processor = None
        self._id2label: dict[int, str] = {}
        self._allowed_indices: set[int] | None = None
        self._device = "cpu"

    def load(self) -> None:
        """Download (if needed) and load the HuggingFace model."""
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch

        logger.info("Loading HuggingFace classifier: %s …", self._model_name)
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModelForImageClassification.from_pretrained(self._model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device).eval()
        self._id2label = {int(k): v for k, v in self._model.config.id2label.items()}

        self._load_allowlist()
        logger.info(
            "NABirds/HF classifier ready: %s (%d labels)", self._model_name, len(self._id2label)
        )

    def _load_allowlist(self) -> None:
        if self._species_list_path is None or not self._species_list_path.exists():
            logger.warning(
                "No species_list_path — classifying against all %d model labels",
                len(self._id2label),
            )
            return

        allowed_terms = {
            line.strip().lower()
            for line in self._species_list_path.read_text().splitlines()
            if line.strip()
        }

        # Match by substring so "northern cardinal" matches a label like
        # "Northern Cardinal" or "Cardinalis cardinalis".
        self._allowed_indices = {
            idx
            for idx, label in self._id2label.items()
            if any(term in label.lower() for term in allowed_terms)
        }

        logger.info(
            "Species allowlist: %d/%d labels active",
            len(self._allowed_indices),
            len(self._id2label),
        )

    def classify(
        self,
        frame_bgr: np.ndarray,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Classify a BGR crop and return top-N (species, confidence) pairs."""
        if self._model is None:
            logger.error("Model not loaded — call load() first")
            return []
        if frame_bgr.size == 0:
            return []

        try:
            import cv2
            import torch
            from PIL import Image

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()

            if self._allowed_indices:
                mask = np.zeros_like(probs)
                for i in self._allowed_indices:
                    if i < len(probs):
                        mask[i] = probs[i]
                total = mask.sum()
                probs = mask / total if total > 0 else mask

            ranked = np.argsort(probs)[::-1][:top_n]
            return [
                (self._id2label.get(int(i), "Unknown"), float(probs[i]))
                for i in ranked
            ]
        except Exception:
            logger.exception("NABirds/HF classification failed")
            return []
