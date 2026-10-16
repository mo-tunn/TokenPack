"""Post-freeze, post-reader exploratory diagnostics. No tuning, no new inference."""
import hashlib,json,statistics
from pathlib import Path
from collections import defaultdict
import pyarrow.parquet as pq
from submission.experiments.phase10_musique_pilot import OUT,DATA,DATA_SHA256,sha_file

def main():
    manifest=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    frozen=json.loads((OUT/"pilot_CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    source=OUT/"pilot_gold_blind_contexts.jsonl"
    assert sha_file(source)==frozen["context_file_sha256"]
    assert sha_file(OUT/"PRE_GOLD_PROTOCOL.json")==frozen["manifest_sha256"]
    reader=json.loads((OUT/"pilot_READER_SCORED_POST_GOLD.json").read_text(encoding="utf8"))
    assert reader["n"]==32 and reader["decision"]=="STOP_NO_VALIDATION"
    assert sha_file(DATA)==DATA_SHA256
    cases=[json.loads(x) for x in source.open(encoding="utf8") if x.strip()]
    assert [x["id"] for x in cases]==manifest["samples"]["pilot"]
    ids={x["id"] for x in cases}
    labels={x["id"]:x for x in pq.ParquetFile(DATA).read(
        columns=["id","paragraphs"]).to_pylist() if x["id"] in ids}
    assert set(labels)==ids
    arms=("atomic-rank","crossdoc-bridge","bridge-neighbor")
    stats={a:{"budget_violations":0,"tokens":[],"sentences":[],
              "all_support":[],"support_recall":[],"selected_paragraphs":[]} for a in arms}
    pair={"neighbor_covered_gain":0,"neighbor_covered_loss":0,
          "neighbor_new_support_paragraphs":0,"neighbor_lost_support_paragraphs":0,
          "new_sentences":0,"lost_sentences":0,
          "neighbor_sentences_explicitly_inserted":0}
    for c in cases:
        gold={int(p["idx"]) for p in labels[c["id"]]["paragraphs"] if p["is_supporting"]}
        assert gold
        ctx=c["contexts"]
        assert set(ctx)==set(arms)
        for a in arms:
            v=ctx[a]
            assert v["budget"]==c["budget"]
            stats[a]["budget_violations"]+=int(v["tokens"]>c["budget"])
            stats[a]["tokens"].append(v["tokens"])
            stats[a]["sentences"].append(v["selected_sentence_count"])
            stats[a]["selected_paragraphs"].append(v["selected_paragraph_count"])
            selected=set(v["selected_source_idxs"])
            stats[a]["all_support"].append(int(gold<=selected))
            stats[a]["support_recall"].append(len(gold&selected)/len(gold))
        old=set(ctx["crossdoc-bridge"]["selected_source_idxs"])
        new=set(ctx["bridge-neighbor"]["selected_source_idxs"])
        pair["neighbor_covered_gain"]+=int(gold<=new and not gold<=old)
        pair["neighbor_covered_loss"]+=int(gold<=old and not gold<=new)
        pair["neighbor_new_support_paragraphs"]+=len((gold&new)-old)
        pair["neighbor_lost_support_paragraphs"]+=len((gold&old)-new)
        a=set(ctx["crossdoc-bridge"]["selected_sentence_indexes"])
        b=set(ctx["bridge-neighbor"]["selected_sentence_indexes"])
        pair["new_sentences"]+=len(b-a)
        pair["lost_sentences"]+=len(a-b)
        pair["neighbor_sentences_explicitly_inserted"]+=ctx["bridge-neighbor"]["neighbor_sentences_added"]
    summary={}
    for a in arms:
        v=stats[a]
        summary[a]={
            "cases":32,
            "support_all":sum(v["all_support"]),
            "support_all_rate":round(statistics.mean(v["all_support"]),6),
            "mean_support_recall":round(statistics.mean(v["support_recall"]),6),
            "mean_actual_cl100k_tokens":round(statistics.mean(v["tokens"]),2),
            "mean_selected_sentences":round(statistics.mean(v["sentences"]),2),
            "mean_selected_paragraphs":round(statistics.mean(v["selected_paragraphs"]),2),
            "min_used_tokens":min(v["tokens"]),
            "max_used_tokens":max(v["tokens"]),
            "budget_violations":v["budget_violations"],
        }
    result={"phase":"Phase10 pilot post-hoc (32 public validation IDs)",
        "label_access_stage":"Only after contexts and 64 reader responses frozen",
        "reference_sha256":{"contexts":frozen["context_file_sha256"],
                            "reader":sha_file(OUT/"pilot_READER_SCORED_POST_GOLD.json")},
        "statistical_status":"EXPLORATORY; no validation because preregistered gate failed",
        "arms":summary,"neighbor_minus_bridge_support_diff":pair,
        "reader_metrics":reader["arms"],
        "reader_pair":reader["neighbor_minus_bridge"],
        "blank_answer_incident":json.loads((OUT/"pilot_EMPTY_RESPONSE_AUDIT.json").read_text(encoding="utf8")),
        "actual_paid_remote_calls":reader["real_modal_calls"]+1,
        "paid_call_accounting_note":"64 task records (one literal empty response) + one additional retry; actual invoice not audited",
        "validation_ids_reserved_but_unused":len(manifest["samples"]["validation"]),
        "production_changed":False,"qasper_test_reused":False}
    outfile=OUT/"PILOT_POSTHOC_DIAGNOSTICS.json"
    if outfile.exists():raise FileExistsError("Preserve posthoc archive")
    outfile.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    old=summary["crossdoc-bridge"];new=summary["bridge-neighbor"]
    m=reader["neighbor_minus_bridge"]
    report=f"""# TokenPack — Faz 10 MuSiQue komşu cümle pilotu (SONUÇ)

**12 Ekim 2026 | Önceden kimliklerle sabitlenmiş 32 pilot sorusu.**
MuSiQue *public validation*; yeni gizli bağımsız benchmark testi değildir.

## Ön kayıt ve veri izolasyonu
Faz 9'un 256 kimliği dışarıda bırakıldı. Diğer ID'lerden deterministik SHA256 seçimle 32 pilot ve ayrı 96 doğrulama ID'si önceden donduruldu. Gold cevap ve supporting etiketleri yalnızca tüm bağlamlar ve okuyucu cevapları kaydedildikten sonra yerel olarak açıldı. Token üst sınırı, son render edilmiş cl100k_base metnine uygulandı. QASPER resmî testi veya üretim dosyaları kullanılmadı.

## Qwen 27B eşleştirilmiş gerçek cevap testi

| Yöntem | EM | Answer F1 |
|---|---:|---:|
| crossdoc-bridge | {reader['arms']['crossdoc-bridge']['correct']}/32 = %37,50 | %{100*reader['arms']['crossdoc-bridge']['F1']:.2f} |
| bridge-neighbor | {reader['arms']['bridge-neighbor']['correct']}/32 = %31,25 | %{100*reader['arms']['bridge-neighbor']['F1']:.2f} |

**Komşu - köprü EM = -6,25 yüzde puan**. Yalnızca komşu doğru: {m['wins']}; yalnızca köprü doğru: {m['losses']}; net doğru: {m['net_correct']}. Eşleştirilmiş 10.000 bootstrap %95 GA = [{100*m['EM_ci95'][0]:+.2f}; {100*m['EM_ci95'][1]:+.2f}] yüzde puan; iki yönlü exact McNemar p = {m['exact_mcnemar_two_sided_p']:.3f}. F1 farkı {100*m['F1_delta']:+.2f} yüzde puan, GA [{100*m['F1_ci95'][0]:+.2f}; {100*m['F1_ci95'][1]:+.2f}].

**Önceden tanımlanan maliyet kapısı geçilemedi:** +3 net doğru beklenirken -2 bulundu. 96 kimliklik doğrulama seçildi, fakat **ne bağlamları üretildi ne Qwen okuyucu çalıştırıldı**. Başarısızlık sonucu değiştirilmeden arşivlendi.

## Kanıt erişimi ve token bütçesi (sonradan gold-local)

| Ölçüm | crossdoc-bridge | bridge-neighbor |
|---|---:|---:|
| En az bir cümleyle tüm gold destek paragraflarına erişim | {old['support_all']}/32 | {new['support_all']}/32 |
| Ortalama gold destek paragraf recall | %{100*old['mean_support_recall']:.2f} | %{100*new['mean_support_recall']:.2f} |
| Ortalama gerçek kullanılan cl100k token | {old['mean_actual_cl100k_tokens']} | {new['mean_actual_cl100k_tokens']} |
| Ortalama seçilen kaynak paragraf | {old['mean_selected_paragraphs']} | {new['mean_selected_paragraphs']} |
| Token kap ihlali | {old['budget_violations']} | {new['budget_violations']} |

Seçici {pair['neighbor_sentences_explicitly_inserted']} adet komşu ekleme işlemi gerçekleştirdi. İki yöntemin final cümle kümeleri arasında {pair['new_sentences']} yeni seçilen, {pair['lost_sentences']} çıkarılan cümle oluştu. Gold destek paragrafına erişim yalnızca **zayıf yapısal proxy**; doğru ara-gerekçe veya gerçek EM değildir.

## Güvenilirlik notu

Bir Qwen istemi iki kez boş metin döndürdü. Soru çıkarılmadı, yeniden etiketlenmedi, cevaba 'unknown' yazılmadı; boş tahmin olarak kaydedildi ve iki denemenin logları saklandı. 64 cevap kaydı vardır; tekrarlı boş istem nedeniyle toplam **65 gerçek Modal çağrısı** yapıldı. Gerçek faturayı sorgulamadık, parasal süre tahmini kesin maliyet değildir. Bu pilot çok küçük; CI sıfırı içerir ve genellenebilir negatif etki kanıtı değildir.

**Sonuç:** Sabit 74% anchor / komşu ±1 algoritması için okuyucu kazanımı gösterilmedi; doğrulama koşulu başarısız. Daha fazla paragraf seçilmesi veya cümle komşuluğu, her zaman cevap için gerekli doğru ilişkinin korunması anlamına gelmez.
"""
    (OUT/"PHASE10_PILOT_TURKCE_RAPOR.md").write_text(report,encoding="utf8")
    print("PHASE10_POSTHOC",json.dumps({"arms":summary,"difference":m,
        "support":pair,"calls":65},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
