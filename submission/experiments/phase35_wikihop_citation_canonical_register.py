"""Register a fresh WikiHop citation-ID canonicalization pilot before calls."""
from __future__ import annotations
import hashlib, json, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'.test-tmp/qangaroo_v1.1.zip'
OUT=ROOT/'submission/results/phase35_wikihop_citation_canonical_20261013'
SOURCE_NAME='qangaroo_v1.1/wikihop/dev.json'
SEED='TokenPack-phase35-WikiHop-citation-canonical-20261013-v1'
PILOT_N=48; VALIDATION_N=96

def sha_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest()
def main():
 if not ARCHIVE.exists(): raise FileNotFoundError(ARCHIVE)
 OUT.mkdir(parents=True,exist_ok=False)
 with zipfile.ZipFile(ARCHIVE) as z: rows=json.loads(z.read(SOURCE_NAME))
 assert len(rows)==5129
 prior_ids=set()
 for folder in ('phase31_wikihop_em_replication_20261013','phase32_wikihop_sentence_atoms_20261013','phase33_wikihop_candidate_reader_20261013','phase34_wikihop_answer_guided_20261013'):
  m=json.loads((ROOT/f'submission/results/{folder}/ID_MANIFEST_PRE_GOLD.json').read_text(encoding='utf8'))
  prior_ids |= {x['id'] for split in ('pilot','validation') for x in m[split]}
 eligible=[]
 for row in rows:
  assert set(row)=={'id','query','answer','candidates','supports','annotations'}
  if row['id'] in prior_ids: continue
  eligible.append({'id':row['id'],'query_sha256':digest(row['query']),'supports_sha256':digest(row['supports']),'candidates_sha256':digest(row['candidates']),'candidate_n':len(row['candidates']),'support_n':len(row['supports']),'annotation_n':len(row['annotations'])})
 eligible.sort(key=lambda x:digest(SEED+'|'+x['id']))
 pilot=eligible[:PILOT_N]; validation=eligible[PILOT_N:PILOT_N+VALIDATION_N]
 assert len(pilot)==PILOT_N and len(validation)==VALIDATION_N
 assert not ({x['id'] for x in pilot}&{x['id'] for x in validation})
 assert not ({x['id'] for x in pilot}|{x['id'] for x in validation})&prior_ids
 protocol={'study':'Phase35 WikiHop citation-ID canonicalization pilot','benchmark':'QAngaroo WikiHop v1.1 official dev split','source_archive_sha256':sha_file(ARCHIVE),'source_archive_bytes':ARCHIVE.stat().st_size,'source_member':SOURCE_NAME,'source_rows':len(rows),'source_license':'CC BY-SA 3.0','selection_seed':SEED,'excluded_prior_phase_ids':len(prior_ids),'pilot_n':PILOT_N,'validation_n':VALIDATION_N,'pilot_gate':{'minimum_net_EM':3,'token_cap_violations':0},'confirmatory_gate':{'paired_two_sided_exact_mcnemar_alpha':0.025,'positive_net_EM_required':True,'token_cap_violations':0},'primary_metric':'normalized paired answer Exact Match','support_metric':None,'support_metric_limitation':'Official WikiHop dev annotations expose relation patterns, not gold supporting source sentence indices.','control':'MiniLM sentence-atom rank with candidate-aware verifier and reader','treatment':'Phase34 answer-guided semantic selector with canonical citation IDs and candidate-aware reader','hypothesis':'Canonicalizing valid numeric citation IDs removes a mechanical fallback bottleneck and reveals whether answer-guided semantic selection improves EM.','canonicalization':'JSON citation_ids integer k or digit string k maps to S{k:02d} only before the existing whitelist, distinct-source, and token-cap checks; out-of-range IDs remain invalid.','same_reader_and_token_cap':True,'gold_answer_unopened_for_selection':True,'gold_sent_remote':False,'qasper_official_test_untouched':True,'prior_phase_ids_untouched':True,'phase34_not_rescored':True,'code_sha256':sha_file(Path(__file__))}
 (OUT/'PRE_GOLD_PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 manifest={'source_archive_sha256':protocol['source_archive_sha256'],'seed':SEED,'pilot':pilot,'validation':validation,'gold_fields_loaded':False,'selection_fields':['id','query','supports','candidates','annotations'],'excluded_from_selection':['answer'],'excluded_prior_ids':sorted(prior_ids)}
 (OUT/'ID_MANIFEST_PRE_GOLD.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 audit={'status':'REGISTERED_PRE_GOLD','archive_hash_valid':True,'rows':len(rows),'eligible_after_prior_exclusion':len(eligible),'pilot_n':len(pilot),'validation_n':len(validation),'pilot_validation_disjoint':True,'prior_disjoint':True,'answer_loaded_for_selection':False,'support_sentence_gold_available':False,'phase34_rescored':False,'decision':'EM_ONLY_CITATION_CANONICALIZATION_CANDIDATE'}
 (OUT/'DATASET_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(json.dumps({'status':'PHASE35_REGISTERED','out':str(OUT),'pilot_n':len(pilot),'validation_n':len(validation),'excluded_prior':len(prior_ids)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
