"""Lightweight YOLOv8 detector — first-pass bird/predator detection.

Uses YOLOv8n (nano, ~6 MB) to detect generic COCO classes.  Only 'bird'
and known predator/visitor classes are returned.

COCO class IDs (0-indexed, as used by ultralytics):
    14 → bird
    15 → cat
    16 → dog
    21 → bear

Small-blob demoting: if a predator-class detection has a bounding-box area
smaller than predator_min_area it is almost certainly a small bird
misclassified by YOLO (e.g. a titmouse labelled "cat").  Those detections
are relabelled "bird" so the species classifier handles them instead.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# COCO class IDs we care about.
_BIRD_CLASS = 14
_PREDATOR_CLASSES = {15: "cat", 16: "dog", 21: "bear"}
_ALL_CLASSES = {_BIRD_CLASS: "bird", **_PREDATOR_CLASSES}


@dataclass(frozen=True)
class Detection:
    label: str       # "bird", "cat", "dog", or "bear"
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    area: int


class BirdDetector:
    """YOLOv8n detector filtered to bird and predator classes.

    Runs on the full frame so it catches birds even if the motion region
    did not capture the complete bounding box.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.25,
        min_area: int = 500,
        max_area: int = 80000,
        predator_min_area: int = 15000,
    ) -> None:
        self._model_name = model_name
        self._confidence = confidence
        self.min_area = min_area
        self.max_area = max_area
        self._predator_min_area = predator_min_area
        self._model = None
        logger.info(
            "BirdDetector: model=%s confidence=%.2f area=[%d, %d] predator_min_area=%d",
            model_name, confidence, min_area, max_area, predator_min_area,
        )

    def load(self) -> None:
        """Download (if needed) and load the YOLO model.  Call once at startup."""
        from ultralytics import YOLO

        logger.info("Loading YOLO model: %s (downloads on first run) …", self._model_name)
        self._model = YOLO(self._model_name)
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)
        logger.info("YOLO model ready")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference and return bird/predator detections within size bounds."""
        if self._model is None:
            logger.error("Model not loaded — call load() first")
            return []

        try:
            results = self._model(frame, conf=self._confidence, verbose=False)
        except Exception:
            logger.exception("YOLO inference failed")
            return []

        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = _ALL_CLASSES.get(cls_id)
                if label is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if not (self.min_area <= area <= self.max_area):
                    continue

                # Demote small predator detections to "bird".  A titmouse or
                # chickadee at typical feeder distance is 2 000–8 000 px²;
                # a real cat fills 20 000 px²+.  YOLO-nano frequently mislabels
                # small birds as "cat" — this catches that case.
                if label != "bird" and area < self._predator_min_area:
                    logger.debug(
                        "Demoting %s (area=%d < predator_min_area=%d) → bird",
                        label, area, self._predator_min_area,
                    )
                    label = "bird"

                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf[0]),
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        area=area,
                    )
                )

        return detections
