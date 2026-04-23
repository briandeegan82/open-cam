#!/usr/bin/env python3
"""Build a SQLite database from the Joensuu matte Munsell ``.mat`` file.

Schema (query-friendly, easy summaries):

- ``meta`` — key/value strings (source path, wavelength range, row counts).
- ``chip`` — one row per color chip: parsed Munsell hue / value / chroma, optional
  16 D65-derived scalars from matrix ``C`` in the MAT file (see ``README.txt``).
- ``spectrum`` — long format: ``(chip_id, wavelength_nm, reflectance)`` with
  composite primary key for ad-hoc SQL and tools like Datasette / sqlite-utils.

Example queries::

    sqlite3 spectra/munsell/munsell.sqlite
    SELECT hue, COUNT(*) AS n FROM chip GROUP BY hue ORDER BY n DESC LIMIT 10;
    SELECT c.label, s.reflectance FROM chip c JOIN spectrum s ON c.id=s.chip_id
      WHERE c.label LIKE '% 5R %' AND s.wavelength_nm=550;

Requires: scipy (see ``requirements.txt``).
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path

import numpy as np

from munsell_mat import load_joensuu_mat, parse_munsell_label, sanitize_filename


def _schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chip (
            id INTEGER PRIMARY KEY,
            mat_index INTEGER NOT NULL UNIQUE,
            label TEXT NOT NULL,
            slug TEXT NOT NULL,
            hue TEXT,
            value REAL,
            chroma REAL,
            c_0 REAL, c_1 REAL, c_2 REAL, c_3 REAL, c_4 REAL, c_5 REAL,
            c_6 REAL, c_7 REAL, c_8 REAL, c_9 REAL, c_10 REAL, c_11 REAL,
            c_12 REAL, c_13 REAL, c_14 REAL, c_15 REAL
        );

        CREATE TABLE IF NOT EXISTS spectrum (
            chip_id INTEGER NOT NULL REFERENCES chip(id) ON DELETE CASCADE,
            wavelength_nm REAL NOT NULL,
            reflectance REAL NOT NULL,
            PRIMARY KEY (chip_id, wavelength_nm)
        );

        CREATE INDEX IF NOT EXISTS idx_spectrum_wavelength ON spectrum(wavelength_nm);
        CREATE INDEX IF NOT EXISTS idx_chip_hue ON chip(hue);
        CREATE INDEX IF NOT EXISTS idx_chip_value ON chip(value);
        CREATE INDEX IF NOT EXISTS idx_chip_chroma ON chip(chroma);
        """
    )


def _print_summary(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    n_chip = cur.execute("SELECT COUNT(*) FROM chip").fetchone()[0]
    n_spec = cur.execute("SELECT COUNT(*) FROM spectrum").fetchone()[0]
    print(f"chips: {n_chip}  spectrum rows: {n_spec}", file=sys.stderr)
    print("top hues by count:", file=sys.stderr)
    for row in cur.execute(
        "SELECT hue, COUNT(*) AS n FROM chip WHERE hue IS NOT NULL "
        "GROUP BY hue ORDER BY n DESC LIMIT 12"
    ):
        print(f"  {row[0]!r}: {row[1]}", file=sys.stderr)
    wl0, wl1 = cur.execute(
        "SELECT MIN(wavelength_nm), MAX(wavelength_nm) FROM spectrum"
    ).fetchone()
    print(f"wavelength_nm range: {wl0} .. {wl1}", file=sys.stderr)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--mat",
        type=Path,
        default=None,
        help="Input .mat (default: spectra/munsell/munsell380_800_1.mat)",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Output SQLite path (default: spectra/munsell/munsell.sqlite)",
    )
    ap.add_argument(
        "--max-chips",
        type=int,
        default=None,
        help="Import only the first N chips (testing).",
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="After import, print counts and top hues.",
    )
    args = ap.parse_args()

    mat_path = (args.mat or (root / "spectra" / "munsell" / "munsell380_800_1.mat")).resolve()
    db_path = (args.db or (root / "spectra" / "munsell" / "munsell.sqlite")).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_joensuu_mat(mat_path)
    spec = bundle.reflectance
    wl = bundle.wavelength_nm
    labels = bundle.labels
    C = bundle.C_D65

    n = spec.shape[1]
    if args.max_chips is not None:
        n = min(n, max(0, args.max_chips))
        spec = spec[:, :n]
        labels = labels[:n]
        if C is not None:
            C = C[:, :n]

    if db_path.is_file():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        _schema(conn)
        h = hashlib.sha256(mat_path.read_bytes()).hexdigest()
        meta_rows = [
            ("source_mat", str(mat_path)),
            ("source_sha256", h),
            ("n_chips", str(n)),
            ("n_wavelengths", str(spec.shape[0])),
            ("wavelength_min_nm", str(float(wl[0]))),
            ("wavelength_max_nm", str(float(wl[-1]))),
        ]
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_rows)

        c_cols = ",".join([f"c_{i}" for i in range(16)])
        ins_chip = (
            f"INSERT INTO chip (mat_index,label,slug,hue,value,chroma,{c_cols}) "
            f"VALUES (?,?,?,?,?,?,{','.join(['?']*16)})"
        )

        chip_ids: list[int] = []
        for i in range(n):
            lab = labels[i]
            parsed = parse_munsell_label(lab)
            slug = sanitize_filename(lab)
            cvals: list[float | None] = [None] * 16
            if C is not None:
                for k in range(16):
                    cvals[k] = float(C[k, i])
            cur = conn.execute(
                ins_chip,
                (
                    i,
                    lab,
                    slug,
                    parsed["hue"],
                    parsed["value"],
                    parsed["chroma"],
                    *cvals,
                ),
            )
            chip_ids.append(int(cur.lastrowid))

        rows_spec: list[tuple[int, float, float]] = []
        for j, cid in enumerate(chip_ids):
            for k in range(spec.shape[0]):
                rows_spec.append((cid, float(wl[k]), float(spec[k, j])))
        CHUNK = 50_000
        for off in range(0, len(rows_spec), CHUNK):
            conn.executemany(
                "INSERT INTO spectrum (chip_id, wavelength_nm, reflectance) VALUES (?,?,?)",
                rows_spec[off : off + CHUNK],
            )

        conn.commit()
        if args.summary:
            _print_summary(conn)
    finally:
        conn.close()

    print(f"wrote {db_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
