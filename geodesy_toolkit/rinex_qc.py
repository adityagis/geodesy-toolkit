"""Quick quality check for RINEX observation files (versions 2.x and 3.x).

For each file it reports the station, receiver, antenna, observation interval,
first and last epoch, number of epochs, completeness against the expected
number of epochs, and any data gaps.

Usage:
    python -m geodesy_toolkit.rinex_qc path/to/file.25o
    python -m geodesy_toolkit.rinex_qc path/to/folder --csv report.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

OBS_SUFFIXES = (".rnx", ".obs", ".crx")
# RINEX 2 epoch line: " yy mm dd hh mi ss.sssssss  f nnn"
EPOCH_V2 = re.compile(r"^ [ \d]\d [ \d]\d [ \d]\d [ \d]\d [ \d]\d [ \d]\d\.\d{7}  [01][ \d]{3}")


@dataclass
class RinexReport:
    file: str
    version: str = ""
    marker: str = ""
    receiver: str = ""
    antenna: str = ""
    interval_s: float | None = None
    first_epoch: datetime | None = None
    last_epoch: datetime | None = None
    epochs: int = 0
    expected_epochs: int | None = None
    completeness_pct: float | None = None
    gaps: list[tuple[datetime, datetime]] = field(default_factory=list)

    @property
    def gap_count(self) -> int:
        return len(self.gaps)

    def as_row(self) -> dict:
        row = asdict(self)
        row["gaps"] = "; ".join(f"{a:%Y-%m-%d %H:%M:%S} -> {b:%H:%M:%S}" for a, b in self.gaps)
        row["gap_count"] = self.gap_count
        return row


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def _is_obs_file(path: Path) -> bool:
    name = path.name.lower().removesuffix(".gz")
    if name.endswith(OBS_SUFFIXES):
        return True
    # RINEX 2 short names: ssssdddf.yyo
    return len(name) >= 3 and name[-1] == "o" and name[-3:-1].isdigit()


def _epoch_v2(line: str) -> datetime | None:
    try:
        yy, mo, dd, hh, mi = (int(line[i:i + 3]) for i in range(0, 15, 3))
        sec = float(line[15:26])
        flag = line[26:29].strip()
    except ValueError:
        return None
    if flag not in ("", "0", "1"):
        return None
    year = yy + (2000 if yy < 80 else 1900)
    return datetime(year, mo, dd, hh, mi) + timedelta(seconds=sec)


def _epoch_v3(line: str) -> datetime | None:
    try:
        parts = line[1:].split()
        year, mo, dd, hh, mi = (int(p) for p in parts[:5])
        sec = float(parts[5])
        flag = parts[6] if len(parts) > 6 else "0"
    except (ValueError, IndexError):
        return None
    if flag not in ("0", "1"):
        return None
    return datetime(year, mo, dd, hh, mi) + timedelta(seconds=sec)


def check_file(path: str | Path, gap_factor: float = 1.5) -> RinexReport:
    """Read one RINEX observation file and return its quality report."""
    path = Path(path)
    rep = RinexReport(file=path.name)
    epochs: list[datetime] = []
    with _open(path) as fh:
        for line in fh:
            label = line[60:80].strip()
            if label == "RINEX VERSION / TYPE":
                rep.version = line[:9].strip()
            elif label == "MARKER NAME":
                rep.marker = line[:60].strip()
            elif label == "REC # / TYPE / VERS":
                rep.receiver = line[20:40].strip()
            elif label == "ANT # / TYPE":
                rep.antenna = line[20:40].strip()
            elif label == "INTERVAL":
                rep.interval_s = float(line[:10])
            elif label == "END OF HEADER":
                break
        v3 = rep.version.startswith("3") or rep.version.startswith("4")
        for line in fh:
            if v3:
                if line.startswith(">"):
                    t = _epoch_v3(line)
                    if t:
                        epochs.append(t)
            elif EPOCH_V2.match(line):
                t = _epoch_v2(line)
                if t:
                    epochs.append(t)

    rep.epochs = len(epochs)
    if not epochs:
        return rep
    epochs.sort()
    rep.first_epoch, rep.last_epoch = epochs[0], epochs[-1]
    if rep.interval_s is None and len(epochs) > 1:
        diffs = sorted((b - a).total_seconds() for a, b in zip(epochs, epochs[1:]))
        rep.interval_s = diffs[len(diffs) // 2]
    if rep.interval_s:
        span = (rep.last_epoch - rep.first_epoch).total_seconds()
        rep.expected_epochs = int(round(span / rep.interval_s)) + 1
        rep.completeness_pct = round(100 * rep.epochs / rep.expected_epochs, 2)
        limit = rep.interval_s * gap_factor
        rep.gaps = [(a, b) for a, b in zip(epochs, epochs[1:]) if (b - a).total_seconds() > limit]
    return rep


def check_path(path: str | Path) -> list[RinexReport]:
    """Check one file, or every RINEX observation file inside a folder."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and _is_obs_file(p))
    return [check_file(f) for f in files]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Quality check for RINEX observation files.")
    ap.add_argument("path", help="RINEX file or folder")
    ap.add_argument("--csv", help="write the report to this CSV file")
    args = ap.parse_args(argv)

    reports = check_path(args.path)
    if not reports:
        print("No RINEX observation files found.", file=sys.stderr)
        return 1
    for r in reports:
        comp = f"{r.completeness_pct:.1f}%" if r.completeness_pct is not None else "n/a"
        print(f"{r.file:28} {r.marker:10} epochs={r.epochs:<6} interval={r.interval_s or 0:g}s "
              f"completeness={comp:>7} gaps={r.gap_count}")
    if args.csv:
        rows = [r.as_row() for r in reports]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Report written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
