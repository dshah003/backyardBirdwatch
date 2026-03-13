"""TFHub AIY Birds V1 species classifier.

Accepts a BGR numpy array (OpenCV frame or crop), converts internally to
RGB, resizes to 224×224, runs the TFHub model, and returns the top-N
(species, confidence) tuples sorted by confidence descending.

Label map: https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv
Model:     https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_URL = "https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1"
LABELS_URL = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv"
INPUT_SIZE = (224, 224)

# The label map uses scientific names; species_list.txt uses common names.
# This mapping lets users keep the list human-readable while the classifier
# resolves to the correct label-map entries.  Multiple scientific names per
# entry handle taxonomy changes (e.g. Picoides → Dryobates woodpeckers).
_COMMON_TO_SCIENTIFIC: dict[str, list[str]] = {
    "northern cardinal":      ["Cardinalis cardinalis"],
    "blue jay":               ["Cyanocitta cristata"],
    "tufted titmouse":        ["Baeolophus bicolor"],
    "black-capped chickadee": ["Poecile atricapillus"],
    "carolina wren":          ["Thryothorus ludovicianus"],
    "downy woodpecker":       ["Dryobates pubescens", "Picoides pubescens"],
    "hairy woodpecker":       ["Dryobates villosus",  "Picoides villosus"],
    "red-bellied woodpecker": ["Melanerpes carolinus"],
    "european starling":      ["Sturnus vulgaris"],
    "house sparrow":          ["Passer domesticus"],
    "house finch":            ["Haemorhous mexicanus", "Carpodacus mexicanus"],
    "white-breasted nuthatch":["Sitta carolinensis"],
    "dark-eyed junco":        ["Junco hyemalis"],
    "american goldfinch":     ["Spinus tristis", "Carduelis tristis"],
    "white-throated sparrow": ["Zonotrichia albicollis"],
    "american robin":         ["Turdus migratorius"],
    "brown-headed cowbird":   ["Molothrus ater"],
    "chipping sparrow":       ["Spizella passerina"],
    "song sparrow":           ["Melospiza melodia"],
}

# Reverse lookup: scientific name (lowercase) → display common name.
_SCIENTIFIC_TO_COMMON: dict[str, str] = {
    sci.lower(): common.title()
    for common, sci_list in _COMMON_TO_SCIENTIFIC.items()
    for sci in sci_list
}


class SpeciesClassifier:
    """TFHub AIY Birds V1 species identifier (~965 North American species).

    Pass a species_list_path to restrict predictions to a known allowlist of
    common names (one per line).  Without it the model picks from all 965
    global species, which causes seabirds and shorebirds to appear at a
    backyard feeder.
    """

    def __init__(self, species_list_path: Path | None = None) -> None:
        self._model = None
        self._labels: list[str] = []
        self._allowed_indices: set[int] | None = None
        self._species_list_path = species_list_path

    def load(self) -> None:
        """Download (if needed) and load the TFHub model + label map."""
        import tensorflow_hub as hub

        logger.info("Loading TFHub bird classifier …")
        # hub.KerasLayer handles TF1-format Hub modules correctly; hub.load()
        # returns an AutoTrackable that is not directly callable for TF1 modules.
        self._model = hub.KerasLayer(MODEL_URL)
        self._load_labels()
        self._load_allowlist()
        logger.info("Species classifier ready (%d labels)", len(self._labels))

    def _load_labels(self) -> None:
        import requests

        try:
            resp = requests.get(LABELS_URL, timeout=30)
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            # CSV format: id,common_name — first line is a header
            self._labels = []
            for line in lines[1:]:
                parts = line.split(",", 1)
                self._labels.append(parts[1].strip() if len(parts) == 2 else "Unknown")
        except Exception:
            logger.exception("Failed to download label map; classification will return Unknown")
            self._labels = []

    def _load_allowlist(self) -> None:
        if self._species_list_path is None or not self._species_list_path.exists():
            logger.warning("No species_list_path set — classifying against all 965 global species")
            return

        entries = [
            line.strip().lower()
            for line in self._species_list_path.read_text().splitlines()
            if line.strip()
        ]

        # Resolve each entry: common name → one or more scientific names,
        # or treat as a scientific name directly if no mapping exists.
        allowed_scientific: set[str] = set()
        unresolved: list[str] = []
        for entry in entries:
            sci_list = _COMMON_TO_SCIENTIFIC.get(entry)
            if sci_list:
                allowed_scientific.update(s.lower() for s in sci_list)
            else:
                allowed_scientific.add(entry)  # already scientific, or typo
                unresolved.append(entry)

        self._allowed_indices = {
            i for i, label in enumerate(self._labels)
            if label.lower() in allowed_scientific
        }

        logger.info(
            "Species allowlist: %d/%d labels active",
            len(self._allowed_indices), len(self._labels),
        )
        if unresolved:
            logger.warning(
                "No scientific-name mapping for (using as-is): %s",
                ", ".join(unresolved),
            )
        not_in_model = allowed_scientific - {
            self._labels[i].lower() for i in self._allowed_indices
        }
        if not_in_model:
            logger.warning(
                "Scientific names not found in label map: %s",
                ", ".join(sorted(not_in_model)),
            )

    def classify(
        self,
        frame_bgr: np.ndarray,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Classify a BGR crop and return top-N (species, confidence) pairs.

        If an allowlist is loaded, only species from that list are considered.
        Returns results sorted by confidence descending.
        Empty list on error or if the model is not loaded.
        """
        if self._model is None:
            logger.error("Model not loaded — call load() first")
            return []
        if frame_bgr.size == 0:
            return []

        try:
            import cv2

            # OpenCV uses BGR; TFHub model expects RGB
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, INPUT_SIZE)
            img_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

            # The AIY Birds V1 model outputs probabilities directly (softmax is
            # applied inside the model).  Use the output directly as probabilities.
            probs = self._model(img_tensor)[0].numpy()

            # Sort all indices by probability descending, then filter to allowlist.
            ranked = np.argsort(probs)[::-1]
            if self._allowed_indices:
                ranked = [i for i in ranked if i in self._allowed_indices]
            top_idx = ranked[:top_n]

            results = []
            for i in top_idx:
                sci_name = self._labels[i] if i < len(self._labels) else "Unknown"
                common_name = _SCIENTIFIC_TO_COMMON.get(sci_name.lower(), sci_name)
                results.append((common_name, float(probs[i])))
            return results
        except Exception:
            logger.exception("Species classification failed")
            return []
