"""Bird track lifecycle manager.

Associates YOLO detections across frames using IoU overlap, accumulates
classification votes per track, and emits a finalized ClosedTrack when a
bird leaves the frame.  The pipeline logs one entry per ClosedTrack instead
of one per frame.

Design:
- Greedy highest-IoU-first matching (sufficient for ≤5 simultaneous birds)
- Confidence-weighted voting: each frame adds conf to the species bucket
- Winning species = highest cumulative vote score
- Reported confidence = average per-frame conf for the winning species
- Track confirmed after min_frames_confirm  (filters single-frame noise)
- Track closed after max_gap_sec with no matching detection
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from detector import Detection

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ActiveTrack:
    track_id: int
    bbox: tuple[int, int, int, int]     # (x1, y1, x2, y2) of most-recent detection
    species_votes: dict[str, float]     # species → cumulative confidence
    vote_counts: dict[str, int]         # species → frame count (for avg-conf calc)
    frame_count: int                    # total frames this track appeared in
    confirmed: bool                     # True once frame_count >= min_frames_confirm
    first_seen_wall: str                # ISO timestamp
    last_seen_mono: float               # monotonic clock (gap detection)
    last_seen_wall: str                 # ISO timestamp
    best_crop: np.ndarray | None        # crop from highest-confidence frame
    best_confidence: float              # highest single-frame species conf seen
    best_species: str                   # species at best_confidence
    best_yolo_conf: float               # YOLO conf at best detection


@dataclass
class ClosedTrack:
    track_id: int
    species_common: str | None          # winning species; None if no confident votes
    confidence: float                   # avg per-frame conf for winning species
    vote_summary: dict[str, float]      # full vote tally (for debug / MQTT payload)
    is_unknown: bool
    first_seen: str                     # ISO
    last_seen: str                      # ISO
    duration_sec: float
    frame_count: int
    crop: np.ndarray | None             # best crop for snapshot saving
    yolo_confidence: float              # YOLO conf of best detection


# ── IoU + greedy matching ────────────────────────────────────────────────────

def _iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_greedy(
    tracks: list[ActiveTrack],
    dets: list[tuple[Detection, str, float, np.ndarray]],
    iou_threshold: float,
) -> tuple[dict[int, int], set[int]]:
    """Greedy highest-IoU-first assignment.

    Returns:
        matched          — {track_idx → det_idx}
        unmatched_dets   — set of det indices that opened no existing track
    """
    if not tracks or not dets:
        return {}, set(range(len(dets)))

    # Collect all (iou, track_idx, det_idx) pairs above threshold
    pairs: list[tuple[float, int, int]] = []
    for t_idx, track in enumerate(tracks):
        for d_idx, (det, _, _, _) in enumerate(dets):
            iou = _iou(track.bbox, (det.x1, det.y1, det.x2, det.y2))
            if iou >= iou_threshold:
                pairs.append((iou, t_idx, d_idx))

    # Greedily assign highest-IoU pairs first (one-to-one)
    pairs.sort(reverse=True)
    matched: dict[int, int] = {}
    used_t: set[int] = set()
    used_d: set[int] = set()
    for _, t_idx, d_idx in pairs:
        if t_idx not in used_t and d_idx not in used_d:
            matched[t_idx] = d_idx
            used_t.add(t_idx)
            used_d.add(d_idx)

    return matched, set(range(len(dets))) - used_d


def _duration(first: str, last: str) -> float:
    try:
        return (
            datetime.fromisoformat(last) - datetime.fromisoformat(first)
        ).total_seconds()
    except Exception:
        return 0.0


# ── Tracker ───────────────────────────────────────────────────────────────────

class BirdTracker:
    """Per-frame IoU tracker.  Call update() each frame, flush() at stream end."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_gap_sec: float = 2.0,
        min_frames_confirm: int = 3,
        min_vote_confidence: float = 0.3,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_gap_sec = max_gap_sec
        self._min_frames_confirm = min_frames_confirm
        self._min_vote_conf = min_vote_confidence
        self._tracks: list[ActiveTrack] = []
        self._next_id: int = 1
        logger.info(
            "BirdTracker ready — iou_threshold=%.2f  max_gap=%.1fs  "
            "min_confirm=%d  vote_conf_min=%.2f",
            iou_threshold, max_gap_sec, min_frames_confirm, min_vote_confidence,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        dets: list[tuple[Detection, str, float, np.ndarray]],
        ts_wall: str,
    ) -> tuple[list[ClosedTrack], dict[int, int]]:
        """Process one frame's bird detections.

        Args:
            dets:     list of (Detection, species_name, species_conf, crop_bgr)
                      Only bird detections; predators are handled separately.
            ts_wall:  ISO-8601 wall-clock timestamp for this frame.

        Returns:
            closed_tracks    — tracks that timed out this frame, ready to log.
            det_idx_to_tid   — maps each detection index → track ID (for debug
                               overlay labelling).
        """
        now_mono = time.monotonic()
        matched, unmatched_dets = _match_greedy(self._tracks, dets, self._iou_threshold)
        det_to_tid: dict[int, int] = {}

        # 1. Update matched tracks ─────────────────────────────────────────
        for t_idx, d_idx in matched.items():
            track = self._tracks[t_idx]
            det, species, conf, crop = dets[d_idx]

            track.bbox = (det.x1, det.y1, det.x2, det.y2)
            track.last_seen_mono = now_mono
            track.last_seen_wall = ts_wall
            track.frame_count += 1

            if not track.confirmed and track.frame_count >= self._min_frames_confirm:
                track.confirmed = True
                logger.debug("Track #%d confirmed (%d frames)", track.track_id, track.frame_count)

            if conf >= self._min_vote_conf:
                track.species_votes[species] = track.species_votes.get(species, 0.0) + conf
                track.vote_counts[species] = track.vote_counts.get(species, 0) + 1

            if conf > track.best_confidence:
                track.best_confidence = conf
                track.best_species = species
                track.best_yolo_conf = det.confidence
                track.best_crop = crop.copy()

            det_to_tid[d_idx] = track.track_id

        # 2. Open new tracks for unmatched detections ─────────────────────
        for d_idx in unmatched_dets:
            det, species, conf, crop = dets[d_idx]
            has_vote = conf >= self._min_vote_conf
            track = ActiveTrack(
                track_id=self._next_id,
                bbox=(det.x1, det.y1, det.x2, det.y2),
                species_votes={species: conf} if has_vote else {},
                vote_counts={species: 1} if has_vote else {},
                frame_count=1,
                confirmed=False,
                first_seen_wall=ts_wall,
                last_seen_mono=now_mono,
                last_seen_wall=ts_wall,
                best_crop=crop.copy() if has_vote else None,
                best_confidence=conf if has_vote else 0.0,
                best_species=species,
                best_yolo_conf=det.confidence,
            )
            logger.debug("Track #%d opened: %s conf=%.2f", self._next_id, species, conf)
            det_to_tid[d_idx] = self._next_id
            self._next_id += 1
            self._tracks.append(track)

        # 3. Close stale tracks ───────────────────────────────────────────
        still_active: list[ActiveTrack] = []
        closed: list[ClosedTrack] = []
        for track in self._tracks:
            if now_mono - track.last_seen_mono > self._max_gap_sec:
                if track.confirmed:
                    closed.append(self._finalize(track))
                else:
                    logger.debug(
                        "Track #%d discarded (unconfirmed, %d frames)",
                        track.track_id, track.frame_count,
                    )
            else:
                still_active.append(track)
        self._tracks = still_active

        return closed, det_to_tid

    def flush(self) -> list[ClosedTrack]:
        """Finalize all remaining confirmed tracks.  Call at end of stream."""
        closed = [self._finalize(t) for t in self._tracks if t.confirmed]
        n_discarded = sum(1 for t in self._tracks if not t.confirmed)
        self._tracks = []
        if n_discarded:
            logger.debug("Flushed %d unconfirmed tracks", n_discarded)
        return closed

    def active_count(self) -> int:
        """Number of currently active tracks (for monitoring)."""
        return len(self._tracks)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _finalize(self, track: ActiveTrack) -> ClosedTrack:
        duration = _duration(track.first_seen_wall, track.last_seen_wall)

        if track.species_votes:
            winner = max(track.species_votes, key=lambda s: track.species_votes[s])
            # Average per-frame confidence for winner (interpretable: "how sure
            # was the classifier on average when it called it [winner]?")
            avg_conf = track.species_votes[winner] / track.vote_counts[winner]
            is_unknown = False
        else:
            winner = None
            avg_conf = 0.0
            is_unknown = True

        logger.info(
            "Track #%d closed: %-28s conf=%.2f  %d frames  %.1fs  "
            "votes=%s",
            track.track_id,
            winner or "unknown",
            avg_conf,
            track.frame_count,
            duration,
            {k: round(v, 2) for k, v in
             sorted(track.species_votes.items(), key=lambda x: -x[1])},
        )

        return ClosedTrack(
            track_id=track.track_id,
            species_common=winner,
            confidence=avg_conf,
            vote_summary=dict(track.species_votes),
            is_unknown=is_unknown,
            first_seen=track.first_seen_wall,
            last_seen=track.last_seen_wall,
            duration_sec=duration,
            frame_count=track.frame_count,
            crop=track.best_crop,
            yolo_confidence=track.best_yolo_conf,
        )
