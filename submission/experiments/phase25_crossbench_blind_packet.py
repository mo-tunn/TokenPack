"""Generate source-and-model-only randomized ALL-64 blind evaluation packet.

This packet is prepared post-gold only for external INDEPENDENT raters;
reviewers receive NO gold, aliases, EM or error category. Reviewer judgments
are not simulated or fabricated and cannot be assumed to exist.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase25_crossbench_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows
FILE=OUT/"BLIND_REVIEW_ALL64.jsonl"
MAP=OUT/"PRIVATE_REVIEW_ID_MAP.json"
SPEC=OUT/"BLIND_REVIEW_PACKET_PROVENANCE.json"
GUIDE=OUT/"BLIND_REVIEW_GUIDELINES_TR.md"
SEED="TokenPack-phase25-all64-review-shuffle-no-gold-v1"
def digest(x):return hashlib.sha256((SEED+"|"+x).encode("utf8")).hexdigest()
def main():
    assert not FILE.exists() and not MAP.exists() and not GUIDE.exists()
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    frozen=json.loads((OUT/"CROSSBENCH_READER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    freeze=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(OUT/"crossbench_reader_responses.jsonl")==frozen["responses_sha256"]
    assert sha_file(OUT/"gold_blind_contexts.jsonl")==freeze["context_file_sha256"]
    c=read_rows(OUT/"gold_blind_contexts.jsonl")
    a=read_rows(OUT/"crossbench_reader_responses.jsonl")
    assert [x["id"] for x in c]==[x["id"] for x in a]==p["pilot_ids"]
    store={x["id"]:(x,y) for x,y in zip(c,a)}
    ordered=sorted(p["pilot_ids"],key=lambda x:(digest(x),x))
    private={};records=[]
    for j,ident in enumerate(ordered,1):
        anonymous=f"MC-{j:03d}"
        case,pred=store[ident]
        private[anonymous]=ident
        records.append({
           "review_case_id":anonymous,
           "question":case["question"],
           "selected_original_Wikipedia_source_evidence":case["final_text"],
           "model_answer":pred["answer"],
           "rubric_supported_by_supplied_evidence":"[supported | contradicted | cannot_determine]",
           "rubric_answer_granularity":"[appropriate | unnecessarily_specific | insufficiently_specific | indeterminate]",
           "rubric_answer_refers_to_requested_entity":"[yes | no | unclear]",
           "reviewer_short_rationale":"[fill in from source]",
           "reviewer_uncertainty":"[low | medium | high]"})
    with FILE.open("x",encoding="utf8") as f:
        for r in records:
            assert not any(k in r for k in (
               "id","answer","gold","aliases","EM","strict_EM","score","category",
               "is_supporting","answerable"))
            f.write(json.dumps(r,ensure_ascii=False)+"\n")
    new_json(MAP,private)
    guide="""# MuSiQue Faz25 — Bağımsız kör anlamsal değerlendirme

Her değerlendirici BLIND_REVIEW_ALL64.jsonl dosyasındaki **64 vakanın tamamını**
birbirinden bağımsız değerlendirmeli. Dosyada doğru/gold cevap, kabul edilen
alternatifler veya Exact Match puanı verilmemiştir. Bu dosya **geliştirme veya
referans cevaplarını öğretme materyali değildir**; ölçüm içindir.

Yalnız soruyu, modelin cevabını ve sağlanan özgün kaynak alıntılarını kullanın.
Dış arama, ek Wikipedia veya gold referans dosyaları kullanmayın.

Her vaka için:
1. **Kanıt desteği:** model cevabını ve gerekli ara ilişkileri verilen alıntılar
   gerçekten destekliyor mu? supported / contradicted / cannot_determine.
   Bir kaynaktan tek cümle görülmesi, tüm hop zincirini doğrulamayabilir.
2. **Soruya uygun ayrıntı:** modelin cevap ayrıntısı soru için uygun mu?
   appropriate / unnecessarily_specific / insufficiently_specific / indeterminate.
   Şehir-ülke, unvan-kişi ve tarih ayrıntısı konularında özen gösterin.
3. **Hedeflenen varlık:** model cevabı sorulan varlığa/olaya mı işaret ediyor?
   yes / no / unclear.
4. Yalnız sağlanan özgün kaynak cümlelerine atıf yapan kısa gerekçe ve
   belirsizlik derecesi yazın.

İki değerlendirici arasında ilk turda müzakere yapılmamalı. İki bağımsız
dosya alındıktan sonra uyuşma oranı/Cohen kappa hesaplanmalı ve çelişen
vakalar üçüncü kişiyle adjudicate edilmeli. Sınıflar otomatik EM'yi
değiştirmez; kesin anlamsal doğruluk iddiası için sağlam hakemlik gereklidir.

PRIVATE_REVIEW_ID_MAP.json yalnız koordinatör içindir. Hakemlere veya
değerlendirme paketiyle birlikte verilmemelidir.

NOT: Paket orijinal model çıktıları dondurulduktan sonra oluşturuldu,
ama bu çalışma bağımsız gizli test değildir. Şu an hiçbir insan hakem
görüşü toplanmış değildir.
"""
    GUIDE.write_text(guide,encoding="utf8")
    spec={"n":64,"benchmark":"MuSiQue public validation",
      "reader_log_sha256":frozen["responses_sha256"],
      "source_context_sha256":freeze["context_file_sha256"],
      "gold_or_score_columns_exported":False,
      "all_cases_included_no_selection_on_gold":True,
      "review_packet_sha256":sha_file(FILE),
      "guide_sha256":sha_file(GUIDE),
      "private_map_sha256":sha_file(MAP),
      "two_independent_human_raters_required":True,
      "human_ratings_collected":False,
      "output_generation_source_sha256":sha_file(Path(__file__)),
      "no_model_calls":True,"qasper_sealed_test_not_used":True}
    new_json(SPEC,spec)
    print("PHASE25_BLIND_REVIEW_PACKET_READY",
       json.dumps({"n":spec["n"],"packet_SHA256":spec["review_packet_sha256"],
                   "human_ratings_collected":False},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
