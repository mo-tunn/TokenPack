"""Metadata-only probe for fresh 2WikiMultiHopQA dataset; never downloads test."""
from huggingface_hub import HfApi
repo="framolfese/2WikiMultihopQA"
info=HfApi().dataset_info(repo)
print("REPO",repo,"SHA",info.sha,flush=True)
for s in info.siblings:
    if "validation" in s.rfilename or s.rfilename.endswith("README.md"):
        print("CANDIDATE",s.rfilename,flush=True)
