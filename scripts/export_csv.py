#!/usr/bin/env python3
"""Export detections from SQLite to CSV.

Usage:
    python scripts/export_csv.py                     # Export all
    python scripts/export_csv.py --days 7            # Last 7 days
    python scripts/export_csv.py --output weekly.csv # Custom output file
"""

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "detections.db"


def export(db_path: Path, output_path: Path, days: int | None = None) -> int:
    """Export detections to CSV. Returns the number of rows exported."""
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM detections"
    params: list[str] = []

    if days is not None:
        query += " WHERE timestamp > datetime('now', ?)"
        params.append(f"-{days} days")

    query += " ORDER BY timestamp DESC"

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    if not rows:
        print("No detections found.", file=sys.stderr)
        return 0

    columns = rows[0].keys()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    conn.close()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export detections to CSV")
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("detections_export.csv"),
        help="Output CSV file path",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only export the last N days",
    )
    args = parser.parse_args()

    count = export(args.db, args.output, args.days)
    print(f"Exported {count} detections to {args.output}")


if __name__ == "__main__":
    main()
