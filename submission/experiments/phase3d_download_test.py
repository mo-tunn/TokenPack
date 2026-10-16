"""Download a SINGLE original QASPER test parquet from immutable HF commit.

Opt-in, public data only; no model download, API or paid service.
No read of test questions, answers or evidence happens here.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import requests

COMMIT="96fd84d8f3c9d9129403f06ddc9ad72042d81a20"
NAME="test-00000-of-00001.parquet"
URL=f"https://huggingface.co/datasets/allenai/qasper/resolve/{COMMIT}/qasper/{NAME}"


def download(out:Path):
    out.mkdir(parents=True,exist_ok=True)
    dest=out/NAME
    if dest.exists():
        raise FileExistsError("Test data already present; refusing replacement")
    partial=out/(NAME+".partial")
    if partial.exists():
        raise FileExistsError("Partial test download exists; inspect manually")
    hasher=hashlib.sha256()
    count=0
    try:
        with requests.get(URL,stream=True,timeout=(15,120),allow_redirects=True) as response:
            response.raise_for_status()
            content_type=response.headers.get("Content-Type","")
            if "text/html" in content_type:
                raise ValueError("HTML returned rather than Parquet")
            with partial.open("xb") as f:
                for block in response.iter_content(chunk_size=262144):
                    if block:
                        f.write(block)
                        hasher.update(block)
                        count+=len(block)
        if count<100000 or partial.read_bytes()[:4]!=b"PAR1":
            raise ValueError("Incomplete or invalid Parquet; .partial retained")
        partial.rename(dest)
    except Exception:
        print("Download incomplete; previous data untouched",flush=True)
        raise
    print(f"DOWNLOADED pinned QASPER test: bytes={count} sha256={hasher.hexdigest()}",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",required=True,type=Path)
    args=ap.parse_args()
    download(args.output_dir)
