"""Download only 2WikiMultiHopQA validation parquet at fixed repository commit."""
import hashlib,json,os
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"submission/data/phase12_2wiki_validation_20261012"
REPO="framolfese/2WikiMultihopQA"
REV="fe713bfbd1afbca1a65246741a75890405d56a3a"
FILE="data/validation-00000-of-00001.parquet"
def go():
    OUT.mkdir(parents=True,exist_ok=True)
    dest=hf_hub_download(repo_id=REPO,repo_type="dataset",revision=REV,
        filename=FILE,local_dir=str(OUT))
    f=Path(dest); digest=hashlib.sha256(f.read_bytes()).hexdigest()
    pf=pq.ParquetFile(f)
    print("VALIDATION",f.name,"n",pf.metadata.num_rows,"bytes",f.stat().st_size,
        "sha256",digest,"schema",str(pf.schema_arrow),flush=True)
    record={"repo":REPO,"revision":REV,"path":FILE,"sha256":digest,
        "bytes":f.stat().st_size,"rows":pf.metadata.num_rows,
        "only_validation_downloaded":True,
        "train_and_test_not_downloaded":True}
    metadata=OUT/"PROVENANCE.json"
    if metadata.exists():
        assert json.loads(metadata.read_text())==record
    else:
        with metadata.open("x",encoding="utf8") as v:
            json.dump(record,v,indent=2);v.write("\n")
if __name__=="__main__":go()
