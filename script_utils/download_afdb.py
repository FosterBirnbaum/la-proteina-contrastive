"""Download AlphaFold DB structures by UniProt accession.

Intended to be run on the remote training server. Reads a newline-delimited
list of UniProt accessions (the format used in the La-Proteina AFDB ID release,
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/la_proteina_afdb_ids.zip)
and fetches one PDB per accession from the EBI AFDB.

Usage:
    python script_utils/download_afdb.py \
        --ids   $DATA_PATH/laproteina_afdb_ids/<id_file> \
        --out   $DATA_PATH/afdb_344k/raw \
        --workers 32

The script is resumable: it skips IDs whose target file already exists with a
reasonable size. Failures are written to <out>/../failed.txt for inspection.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

EBI_URL_TEMPLATE = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v{version}.pdb"
MIN_VALID_BYTES = 1024  # anything smaller is almost certainly an error page

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_afdb")


def make_session(retries: int = 5, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open() as f:
        for line in f:
            tok = line.strip().split()[0] if line.strip() else ""
            if not tok or tok.startswith("#"):
                continue
            # accept either bare UniProt (Q9NRG9) or AF-<uniprot>-F1-model_v4 format
            if tok.startswith("AF-"):
                tok = tok.split("-")[1]
            ids.append(tok)
    return ids


def target_path(out_dir: Path, uniprot: str, version: int) -> Path:
    return out_dir / f"AF-{uniprot}-F1-model_v{version}.pdb"


def already_done(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size >= MIN_VALID_BYTES
    except OSError:
        return False


def fetch_one(session: requests.Session, uniprot: str, out_dir: Path, version: int,
              timeout: float = 30.0) -> tuple[str, bool, str]:
    dest = target_path(out_dir, uniprot, version)
    if already_done(dest):
        return (uniprot, True, "exists")
    url = EBI_URL_TEMPLATE.format(uniprot=uniprot, version=version)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, timeout=timeout, stream=True) as r:
            if r.status_code == 404:
                return (uniprot, False, "404")
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        if tmp.stat().st_size < MIN_VALID_BYTES:
            tmp.unlink(missing_ok=True)
            return (uniprot, False, f"too_small({tmp.stat().st_size if tmp.exists() else 0}b)")
        tmp.replace(dest)
        return (uniprot, True, "ok")
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return (uniprot, False, f"err:{type(e).__name__}:{e}")


def run(ids: Iterable[str], out_dir: Path, fail_log: Path, workers: int, version: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_log.parent.mkdir(parents=True, exist_ok=True)
    ids = list(ids)
    total = len(ids)
    log.info(f"Targeting {total} AFDB entries -> {out_dir} (workers={workers})")

    session = make_session()
    done_existing = 0
    done_fetched = 0
    failed: list[tuple[str, str]] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex, fail_log.open("a") as flog:
        futures = {ex.submit(fetch_one, session, uid, out_dir, version): uid for uid in ids}
        for i, fut in enumerate(as_completed(futures), 1):
            uid, ok, status = fut.result()
            if ok:
                if status == "exists":
                    done_existing += 1
                else:
                    done_fetched += 1
            else:
                failed.append((uid, status))
                flog.write(f"{uid}\t{status}\n")
                flog.flush()
            if i % 500 == 0 or i == total:
                elapsed = time.time() - started
                rate = i / elapsed if elapsed > 0 else 0
                log.info(
                    f"[{i}/{total}] fetched={done_fetched} skipped={done_existing} "
                    f"failed={len(failed)}  {rate:.1f}/s"
                )

    log.info(
        f"Done. fetched={done_fetched} skipped(existing)={done_existing} "
        f"failed={len(failed)}. Failures logged to {fail_log}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", type=Path, required=True,
                    help="Newline-delimited UniProt accession file (or AF-...-model_v4 names).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Destination directory for AF-<uniprot>-F1-model_v4.pdb files.")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--version", type=int, default=4,
                    help="AFDB model version (default 4).")
    ap.add_argument("--failed-log", type=Path, default=None,
                    help="Where to append (id, reason) for failures. "
                         "Defaults to <out>/../failed.txt")
    args = ap.parse_args()

    if not args.ids.exists():
        log.error(f"ID file not found: {args.ids}")
        return 2

    fail_log = args.failed_log or (args.out.parent / "failed.txt")
    ids = read_ids(args.ids)
    if not ids:
        log.error("No IDs parsed from input.")
        return 2

    run(ids, args.out, fail_log, args.workers, args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
