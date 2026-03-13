"""Dual-write logger: CSV + SQLite for detection events."""

import csv
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from config import CSV_PATH, DB_PATH

log = logging.getLogger(__name__)

CSV_FIELDS = [
    "timestamp",
    "date",
    "time",
    "species_common",
    "species_scientific",
    "confidence",
    "source",
    "classifier",
    "duration_sec",
    "count",
    "weather",
    "temperature_f",
    "snapshot_path",
    "audio_path",
    "frigate_event_id",
    "reviewed",
    "corrected_species",
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    species_common TEXT,
    species_scientific TEXT,
    confidence REAL,
    source TEXT DEFAULT 'visual',
    classifier TEXT,
    duration_sec REAL,
    count INTEGER DEFAULT 1,
    temperature_f REAL,
    snapshot_path TEXT,
    audio_path TEXT,
    frigate_event_id TEXT,
    reviewed INTEGER DEFAULT 0,
    corrected_species TEXT
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_species ON detections(species_common);",
    "CREATE INDEX IF NOT EXISTS idx_date ON detections(date(timestamp));",
]

INSERT_SQL = """
INSERT INTO detections (
    timestamp, species_common, species_scientific, confidence, source,
    classifier, duration_sec, count, temperature_f, snapshot_path,
    audio_path, frigate_event_id, reviewed, corrected_species
) VALUES (
    :timestamp, :species_common, :species_scientific, :confidence, :source,
    :classifier, :duration_sec, :count, :temperature_f, :snapshot_path,
    :audio_path, :frigate_event_id, :reviewed, :corrected_species
)
"""


@dataclass
class DetectionRecord:
    timestamp: str
    species_common: str
    species_scientific: str
    confidence: float
    source: str = "visual"
    classifier: str = "inaturalist"
    duration_sec: float | None = None
    count: int = 1
    weather: str | None = None
    temperature_f: float | None = None
    snapshot_path: str | None = None
    audio_path: str | None = None
    frigate_event_id: str | None = None
    reviewed: int = 0
    corrected_species: str | None = None
    date: str = field(init=False)
    time: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            dt = datetime.fromisoformat(self.timestamp)
            self.date = dt.strftime("%Y-%m-%d")
            self.time = dt.strftime("%H:%M:%S")
        except ValueError:
            self.date = ""
            self.time = ""


class DetectionLogger:
    """Writes detection records to both CSV and SQLite."""

    def __init__(
        self,
        csv_path: Path = CSV_PATH,
        db_path: Path = DB_PATH,
    ) -> None:
        self._csv_path = csv_path
        self._db_path = db_path
        self._init_csv()
        self._init_db()

    def _init_csv(self) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
            log.info("Created CSV log at %s", self._csv_path)

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(CREATE_TABLE_SQL)
            for idx_sql in CREATE_INDEXES_SQL:
                conn.execute(idx_sql)
            conn.commit()
            log.info("Initialized SQLite database at %s", self._db_path)
        finally:
            conn.close()

    def log(self, record: DetectionRecord) -> None:
        """Write a detection record to both CSV and SQLite."""
        self._write_csv(record)
        self._write_db(record)
        log.info(
            "Logged detection: %s (%.2f) via %s",
            record.species_common,
            record.confidence,
            record.classifier,
        )

    def _write_csv(self, record: DetectionRecord) -> None:
        row = asdict(record)
        try:
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writerow({k: row.get(k) for k in CSV_FIELDS})
        except OSError:
            log.exception("Failed to write CSV")

    def _write_db(self, record: DetectionRecord) -> None:
        row = asdict(record)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(INSERT_SQL, row)
            conn.commit()
        except sqlite3.Error:
            log.exception("Failed to write to SQLite")
        finally:
            conn.close()
