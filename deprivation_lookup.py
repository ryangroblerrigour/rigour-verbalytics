"""
Postcode -> deprivation quintile lookup for the Rigour validate-postcode API.
RIG926002 HKR x Macmillan Cost of Cancer.

Data file: RIG926002_postcode_deprivation_api_lookup.csv.gz
  columns: postcode (upper case, no spaces), imd_quintile (1-5), imd_rank, imd_index
  2,690,877 UK postcodes (live + terminated), NSPL May 2025.

Uses SQLite (standard library) so the 2.7m rows sit on disk, not in memory.
Run build_db() once at deploy/startup, then call lookup_deprivation() per request.
"""
import csv
import gzip
import os
import re
import sqlite3

LOOKUP_GZ = os.environ.get("DEPRIVATION_LOOKUP_GZ", "RIG926002_postcode_deprivation_api_lookup.csv.gz")
DB_PATH = os.environ.get("DEPRIVATION_DB", "postcode_deprivation.sqlite")

EMPTY = {"imd_quintile": None, "imd_rank": None, "imd_index": None}


def build_db(gz_path=LOOKUP_GZ, db_path=DB_PATH):
    """Load the gzipped CSV into an indexed SQLite table (one-off, ~30s)."""
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS dep")
    con.execute(
        "CREATE TABLE dep (postcode TEXT PRIMARY KEY, imd_quintile INTEGER, "
        "imd_rank INTEGER, imd_index TEXT) WITHOUT ROWID"
    )
    with gzip.open(gz_path, "rt", newline="", encoding="utf-8") as fh:
        rows = csv.reader(fh)
        next(rows)  # header
        con.executemany(
            "INSERT INTO dep VALUES (?, ?, ?, ?)",
            ((p, int(q), int(r), i) for p, q, r, i in rows),
        )
    con.commit()
    con.close()


_con = None


def lookup_deprivation(postcode):
    """Return {"imd_quintile", "imd_rank", "imd_index"}; all None if not found."""
    global _con
    if _con is None:
        _con = sqlite3.connect(DB_PATH, check_same_thread=False)
    key = re.sub(r"[^A-Z0-9]", "", (postcode or "").upper())
    if not key:
        return dict(EMPTY)
    row = _con.execute(
        "SELECT imd_quintile, imd_rank, imd_index FROM dep WHERE postcode = ?", (key,)
    ).fetchone()
    if row is None:
        return dict(EMPTY)
    return {"imd_quintile": row[0], "imd_rank": row[1], "imd_index": row[2]}


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        build_db()
    for pc in ["SW1A 1AA", "l8 0rn", "CF10 1EP", "EH1 1YZ", "BT1 5GS", "ZZ9 9ZZ", ""]:
        print(repr(pc), lookup_deprivation(pc))
