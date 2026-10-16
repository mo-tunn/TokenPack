"""Download ONLY publicly licensed QASPER train/validation parquet files.

This is the sole Phase 3B component that uses network access. The user
explicitly authorized proceeding with the complete QASPER train/validation
experiment. It never downloads models or contacts paid APIs.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import requests

# Fixed, reviewable release, avoiding revisions changing under the analysis.
COMMIT="96fd84d8f3c9d9129403f06ddc9ad72042d81a20"
BASE=f"https://huggingface.co/datasets/allenai/qasper/resolve/{COMMIT}/qasper"
SPLITS={
    "train":("train-00000-of-00001.parquet",14368661),
    "validation":("validation-00000-of-00001.parquet",4742094),
}


def download(out:Path):
    out.mkdir(parents=True,exist_ok=True)
    for split,(name,expected_size) in SPLITS.items():
        dest=out/name
        if dest.exists():
            if dest.stat().st_size != expected_size:
                raise ValueError(f"Unexpected existing {name} size. Not overwriting.")
            print(f"REUSED {split} {expected_size} bytes",flush=True)
            continue
        url=f"{BASE}/{name}"
        # Never replace an existing historical file on failure.
        temp=out/(name+".partial")
        if temp.exists():
            raise FileExistsError("Partial file needs manual review: "+str(temp))
        size=0
        sha=hashlib.sha256()
        with requests.get(url,stream=True,timeout=(15,90),allow_redirects=True) as response:
            response.raise_for_status()
            for block in response.iter_content(chunk_size=262144):
                if not block:continue
                with temp.open("ab") as f:f.write(block)
                sha.update(block)
                size+=len(block)
                if size%(262144*16)<262144:
                    print(f"{split}: {size}/{expected_size} bytes",flush=True)
        if size!=expected_size:
            raise ValueError(f"Unexpected {name} size {size}!={expected_size}; left .partial")
        magic=temp.open("rb").read(4)
        if magic!=b"PAR1":
            raise ValueError(f"{name} did not have parquet magic; left .partial")
        temp.rename(dest)
        print(f"DOWNLOADED {split}: {size} bytes sha256={sha.hexdigest()}",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()
    download(args.output_dir)
