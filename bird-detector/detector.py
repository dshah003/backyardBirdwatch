"""Lightweight YOLOv8 detector — first-pass bird/predator detection.

Uses YOLOv8n (nano, ~6 MB) to detect generic COCO classes.  Only 'bird'
and known predator/visitor classes are returned.

COCO class IDs (0-indexed, as used by ultralytics):
    14 → bird
    15 → cat
    16 → dog
    21 → bear

Note: squirrel and raccoon are not COCO classes.  Restricting inference to
only {bird, cat} forces YOLO to misclassify squirrels as cats.  Allowing
all classes lets YOLO abstain when it isn't confident in any of its known
labels, which is the correct behaviour for animals it wasn't trained on.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# COCO class IDs we care about.
_CLASSES = {14: "bird", 15: "cat", 16: "dog", 21: "bear"}


@dataclass
class Detection:
    label: str       # "bird" or "cat"
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
    ) -> None:
        self._model_name = model_name
        self._confidence = confidence
        self.min_area = min_area
        self.max_area = max_area
        self._model = None
        logger.info(
            "BirdDetector: model=%s confidence=%.2f area=[%d, %d]",
            model_name, confidence, min_area, max_area,
        )

    def load(self) -> None:
        """Download (if needed) and load the YOLO model.  Call once at startup."""
        from ultralytics import YOLO  # import deferred — heavy dependency

        logger.info("Loading YOLO model: %s (downloads on first run) …", self._model_name)
        self._model = YOLO(self._model_name)
        # Prime the model with a dummy pass to avoid latency on the first real frame.
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)
        logger.info("YOLO model ready")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference and return bird/cat detections within size bounds."""
        if self._model is None:
            logger.error("Model not loaded — call load() first")
            return []

        try:
            results = self._model(
                frame,
                conf=self._confidence,
                # No class filter: letting YOLO evaluate all 80 COCO classes
                # prevents it from force-fitting squirrels/raccoons into the
                # nearest allowed class (cat).  We filter to _CLASSES below.
                verbose=False,
            )
        except Exception:
            logger.exception("YOLO inference failed")
            return []

        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = _CLASSES.get(cls_id)
                if label is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if not (self.min_area <= area <= self.max_area):
                    continue

                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf[0]),
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        area=area,
                    )
                )

        return detections
