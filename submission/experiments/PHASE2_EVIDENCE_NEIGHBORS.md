# Faz 2 — Evidence-neighbor augmentation + source-grouped rendering

**Durum:** Deneysel. Mevcut TokenPack ve RAG varsayilanlari degismedi.

## Hipotez

Ayni token *tavani* altinda, TokenPack'in kaynakta hemen bitisik
tamamlayici parcalari sinirli sayida geri almasi ve bunlari kaynak
sirasinda gruplamasi yanit kalitesini iyilestirebilir.

Bu, SEMANTIK kanit zinciri kuran bir sistem degil: yalnizca belge
kimlikleri, yapisal konum ve fiziksel bitisiklik kullaniliyor.

## Eklenen kod

- src/tokenpack/evidence_neighbors.py:
  - Ilk %85 butceyle degismemis budget-top-k secimi.
  - En cok %15 ayrilan butcede kaynak komsusu tamamlama.
  - Komsular top-300 aday havuzunun disindan gelebilir.
  - Kullanilmayan token butcesi eski secicinin normal doldurmasina gider.
- src/tokenpack/evidence_grouping.py:
  - Bitişik kaynak parcalarinin ayrik gruplar halinde sunumu.
  - Kaynak metin kelimesi kelimesine degismez.
  - Grup basliklariyla birlikte gercek tokenizer butcesi kontrol edilir.
  - Sigmayan en dusuk puanli TAM parca cikarilir; icerik kesilmez.
- submission/longbench_eval/client.py:
  - Varsayilanlar dokunulmadi, sadece --phase2-ablation aktifken ek kollari olusturur.
  - --run-modal olmadan API/GPU cagrisi yapmaz.

## Sekiz deney kolu

1. full-context (orijinal, degismedi)
2. production-rag-50 (orijinal, degismedi)
3. tokenpack-50 (orijinal, degismedi)
4. production-rag-50-p2-budgeted (baslik maliyeti dahil tam butce kontrolu)
5. tokenpack-50-p2-control (mevcut secici + tam butce kontrolu)
6. tokenpack-50-p2-grouped (mevcut secici + grup render)
7. tokenpack-50-p2-neighbor (komsu secimi + mevcut skor sirali render)
8. tokenpack-50-p2-neighbor-grouped (komsu secimi + grup render)

Bu sayede secim, gruplama ve basliklarin butce maliyeti farklari
olculebilir. Giris tokenlari tam esit degildir; ortak ust limit esittir.

## Yerel olarak yeniden calistirma

Proje koku:

    C:\Users\metehan\Desktop\TokenPack

Mevcut tamamlanmis arsivden iki gercek LongBench metniyle yerel CPU
kontrolu (model cevabi yok, GPU/API yok, HF offline):

    set HF_HUB_OFFLINE=1
    set TRANSFORMERS_OFFLINE=1
    C:\Python313\python.exe submission\experiments\phase2_offline_smoke.py --cases 2

Bu komut, halihazirda olusturulmus ayni rapor dosyasini tekrar
yazmayi reddeder. Yeniden calistirilacaksa farkli cikti klasoru
kullanilmali; onceki rapor silinmemeli.

Testler:

    C:\Python313\python.exe -m pytest -q tests\test_evidence_phase2.py tests\test_longbench_eval.py tests\test_longbench_fast_pilot.py tests\test_evidence_failure_audit.py

Gecerli LongBench JSON/JSONL data dosyasiyla yalnizca yeni prompt
hazirlama, cagrisiz:

    C:\Python313\python.exe submission\longbench_eval\client.py --data-file DATASET.json --output-dir NEW_OUTPUT_DIR --skip-compression --phase2-ablation --limit 2 --dry-run

Dikkat: --run-modal verme. phase2-ablation icin skip-compression,
budget-ratio=0.50, selection-strategy=budget-top-k,
context-order=score ve cakisan deney bayraklarinin KAPALI olmasi gerekir.

## Gercek CPU kontrolu — iki eski kisa LongBench sorusu

1) Case 66ebed525a08c7b9b35e1cb4:
- Kaynak: 23.719 token, butce tavani: 11.859.
- Butun Faz2 ciktilari bu tavanin icinde.

2) Case 66f39aa7821e116aacb2da76:
- Kaynak: 15.410 token, butce tavani: 7.705.
- Eski TP-50: 8.303 (eski renderer basliklariyla tavan asiliyor).
- Eski RAG-50: 8.596 (ayni eski baslik maliyeti problemi).
- Butceli RAG-50: 7.241.
- TP Faz2 kontrol: 7.358.
- TP Faz2 grouped: 7.218.
- TP Faz2 neighbor: 7.640.
- TP Faz2 neighbor+grouped: 7.514.

Rapor:

    submission/results/phase2_offline_smoke_v2_20261011/token_budget_report.json

Bu CPU kontrolu cevap dogrulugunu HIC OLCMEDI.

## Bilinen sinirlamalar ve sonraki adim

- LongBench adaptorunde tum orijinal uzun metin tek sentetik kaynak
  kimliginde tutuluyor. Kaynak ofsetinde bitisik olmak, ayni mantiksal
  belgeden gelmek veya bir nedensel bag kurmak anlamina gelmez.
- Su anki moduller sadece kaynak-provenansli/yapisal komsuluk saglar.
  Anlamsal tamamlilik ve cok-adimli kanit zinciri sonraki Faz 3'tur.
- Header maliyeti icin dusuk puanli chunklarin dusurulmesi
  baseline secime gore ayrica raporlanir.
- Phase2 production pack_source varsayilanina DAHIL EDILMEDI.
- Bu deneyler gelistirmede kullanilan sorularla kuruldugundan,
  uyarlamadan sonra yeni hold-out sorularda dogrulanmalidir.
- Ayni prompt, model, seed, maksimum cikti ve token tavaniyla
  kucuk bir eslestirilmis inference testi yapmadan
  yeni yaklasimin daha dogru cevap verdigi ileri surulemez.
