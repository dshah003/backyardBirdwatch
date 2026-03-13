"""iNaturalist Computer Vision API client with rate limiting.

NOTE: The iNaturalist CV API (computervision/score_image) is not publicly
available — it requires approved/fee-based access.  Set INAT_API_TOKEN in
your .env with a JWT obtained from https://www.inaturalist.org/users/api_token
(requires an iNaturalist account and approved API access).
"""

import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests

from config import INAT_API_TOKEN, INAT_RATE_LIMIT_PER_MIN, LATITUDE, LONGITUDE

logger = logging.getLogger(__name__)

# Correct endpoint (singular "computervision", not "computervisions")
INAT_CV_URL = "https://api.inaturalist.org/v1/computervision/score_image"


@dataclass
class ClassificationResult:
    species_common: str
    species_scientific: str
    confidence: float
    taxon_id: int


class INatClient:
    """Rate-limited iNaturalist CV API client."""

    def __init__(
        self,
        rate_limit_per_min: int = INAT_RATE_LIMIT_PER_MIN,
        lat: float | None = LATITUDE,
        lng: float | None = LONGITUDE,
        api_token: str | None = INAT_API_TOKEN,
    ) -> None:
        self._min_interval = 60.0 / rate_limit_per_min
        self._last_call: float = 0.0
        self._lat = lat
        self._lng = lng
        self._session = requests.Session()
        if api_token:
            self._session.headers["Authorization"] = f"Bearer {api_token}"
        else:
            logger.warning(
                "INAT_API_TOKEN is not set — iNaturalist CV API calls will fail. "
                "Set it in .env (requires approved iNat API access)."
            )

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def classify(self, image_path: Path) -> list[ClassificationResult]:
        """Classify an image using the iNaturalist CV API.

        Returns top 5 results sorted by combined score.
        """
        self._wait_for_rate_limit()

        params: dict[str, object] = {}
        if self._lat is not None and self._lng is not None:
            params["lat"] = self._lat
            params["lng"] = self._lng
        params["observed_on"] = date.today().isoformat()

        try:
            with open(image_path, "rb") as f:
                response = self._session.post(
                    INAT_CV_URL,
                    files={"image": ("snapshot.jpg", f, "image/jpeg")},
                    params=params,
                    timeout=30,
                )
            self._last_call = time.monotonic()

            if response.status_code in (401, 403):
                logger.error(
                    "iNaturalist CV API auth failed (%d). "
                    "Set INAT_API_TOKEN in .env — access requires iNat approval.",
                    response.status_code,
                )
                return []
            if response.status_code == 404:
                logger.error(
                    "iNaturalist CV API returned 404. "
                    "The endpoint may have moved; check api.inaturalist.org/v1/docs."
                )
                return []

            response.raise_for_status()

            data = response.json()
            results: list[ClassificationResult] = []
            for r in data.get("results", [])[:5]:
                taxon = r.get("taxon", {})
                results.append(
                    ClassificationResult(
                        species_common=taxon.get(
                            "preferred_common_name", taxon.get("name", "Unknown")
                        ),
                        species_scientific=taxon.get("name", "Unknown"),
                        confidence=r.get("combined_score", 0.0),
                        taxon_id=taxon.get("id", 0),
                    )
                )
            return results

        except requests.RequestException:
            logger.exception("iNaturalist API request failed")
            return []
