"""OpenCV motion detection using the MOG2 background subtractor."""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionRegion:
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


class MotionDetector:
    """MOG2-based motion detector with contour filtering.

    MOG2 builds a statistical background model over time and marks pixels
    that deviate significantly from it as foreground.  Dilation merges
    nearby blobs so a bird's body and wings register as one region.
    """

    def __init__(
        self,
        history: int = 300,
        var_threshold: float = 32,
        min_area: int = 800,
        max_area: int = 120000,
        dilate_iterations: int = 2,
    ) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,  # shadows slow things down; not needed here
        )
        self.min_area = min_area
        self.max_area = max_area
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._dilate_iter = dilate_iterations
        logger.info(
            "MotionDetector: history=%d var_threshold=%.1f area=[%d, %d]",
            history, var_threshold, min_area, max_area,
        )

    def detect(self, frame: np.ndarray) -> list[MotionRegion]:
        """Apply background subtraction and return motion regions.

        Returns regions whose area falls within [min_area, max_area].
        The frame is converted to greyscale internally before processing.
        """
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._bg.apply(grey)

        # Remove shadow pixels (value 127) — keep only definite foreground (255).
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        # Dilate to connect fragmented blobs (e.g. wings + body).
        mask = cv2.dilate(mask, self._kernel, iterations=self._dilate_iter)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions: list[MotionRegion] = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(c)
                regions.append(MotionRegion(x=x, y=y, w=w, h=h, area=int(area)))

        return regions

    def warmup(self, frame: np.ndarray, passes: int = 30) -> None:
        """Feed the same frame N times to seed the background model.

        Call this with the first frame before starting the main loop so the
        model does not treat the entire initial scene as foreground.
        """
        for _ in range(passes):
            self._bg.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        logger.info("Motion detector warmed up with %d background frames", passes)
