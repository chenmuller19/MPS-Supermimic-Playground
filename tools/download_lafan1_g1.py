"""Download and verify the 40 G1-retargeted LAFAN1 CSV clips from Hugging Face."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.request import Request, urlopen

DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
API_URL = f"https://huggingface.co/api/datasets/{DATASET}/tree/main/g1"
FILE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/g1"
EXPECTED_CLIPS = 40
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "origin_lafan1_g1"


def _open_with_retries(url: str, timeout: int):
    request = Request(url, headers={"User-Agent": "SuperMimic-LAFAN1-data-fetch/1.0"})
    for attempt in range(5):
        try:
            return urlopen(request, timeout=timeout)
        except Exception:
            if attempt == 4:
                raise
            delay = min(5 * (attempt + 1), 20)
            print(f"request failed; retrying in {delay}s: {url}", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def _git_blob_oid(content: bytes) -> str:
    header = b"blob " + str(len(content)).encode() + bytes([0])
    return hashlib.sha1(header + content).hexdigest()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with _open_with_retries(API_URL, timeout=90) as response:
        entries = json.load(response)
    files = [entry for entry in entries if entry.get("type") == "file" and entry["path"].endswith(".csv")]
    if len(files) != EXPECTED_CLIPS:
        raise RuntimeError(f"expected {EXPECTED_CLIPS} dataset clips, found {len(files)}")
    print(f"dataset lists {len(files)} clips; downloading to {DESTINATION}", flush=True)

    for index, entry in enumerate(files, 1):
        name = Path(entry["path"]).name
        destination = DESTINATION / name
        expected_size = int(entry["size"])
        expected_oid = entry["oid"].split(":")[-1]

        if destination.exists():
            content = destination.read_bytes()
            if len(content) == expected_size and _git_blob_oid(content) == expected_oid:
                print(f"verified existing [{index}/{len(files)}] {name}", flush=True)
                continue
            if name != "dance1_subject1.csv":
                raise FileExistsError(f"refusing to replace existing non-dataset file: {destination}")

        url = f"{FILE_URL}/{name}"
        print(f"downloading [{index}/{len(files)}] {name}", flush=True)
        with _open_with_retries(url, timeout=600) as response:
            content = response.read()
        actual_oid = _git_blob_oid(content)
        if len(content) != expected_size or actual_oid != expected_oid:
            raise RuntimeError(f"download verification failed for {name}")
        fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=DESTINATION)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(f"downloaded {name} ({expected_size} bytes)", flush=True)

    downloaded = sorted(DESTINATION.glob("*.csv"))
    if len(downloaded) != EXPECTED_CLIPS:
        raise RuntimeError(f"expected {EXPECTED_CLIPS} CSV files in {DESTINATION}, found {len(downloaded)}")
    print(f"verified {len(downloaded)} LAFAN1 G1 clips in {DESTINATION}")


if __name__ == "__main__":
    main()
