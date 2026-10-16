# Sprint 03 — Metrik, gecikme ve istatistik doğruluğu

Önce ortak sözleşmeyi, Sprint 01–02 sonuçlarını ve audit bölüm 4'ü oku. Önceki sprintlerin config/token sözleşmesini kullan. Bu sprint yalnız araştırma önerisi değildir: metric kodunu, rapor üretimini ve regresyonlarını uygula; mevcut ham sonuçları yeni sürümlü analizle değerlendir.

## Hedef

“Doğru cevap”, “kanıt koruma”, “toplam süre” ve “exact/timeout” ifadeleri kodda ölçülen şeyi doğru anlatsın. Tarihsel sonuçları koruyarak yeni analiz hattı kur.

## Uygula

### A. Denominator ve koşu bütünlüğü

- LongBench ve QASPER generation raporlarını incele. Birincil answer accuracy planlanan vaka manifestine göre hesaplanmalı; parse/API failure doğru paydadan çıkarılmamalı.
- Başarılı cevap koşullu doğruluğu ancak ayrıca adlandırılmış ikincil metrik olarak bırak.
- Planned/running/completed/failed/missing ve parse failure'ı ayır. Eksik sonucu üretim başarısı gibi etiketleme.
- Duplicate task/case ID, aynı yöntemin tekrarları, farklı vaka kümeleri ve bilinmeyen yöntemleri açık doğrula. Tekrar run'ları seed/run ID ile ayır.
- Pairwise karşılaştırma için ortak planlı vaka kümesi ve eksik yanıt politikası belirle; sessiz inner join ile vakaları atma.
- Çalıştırılmamış yöntemler 0 doğruluk yerine N/A veya “not run” taşısın.

### B. Tam süre ve token ölçümü

- Extraction, chunking, document/query embedding, scoring/reranking, selection, compressor preparation, compression, rendering, generation ve I/O'yu ölçülebilir stage'lere ayır.
- Timer'ları gerçek işlem sınırlarına yerleştir. Ortak preprocessing maliyetini ablation yöntemlerine iki kere yükleme; yöntem attribution politikasını açıkla.
- Cold-document, warm-index/new-query ve prepared-context modlarını ayır. Model load/cold start dahil mi her kayıtta belli olsun.
- Batch ortalamasını per-request latency diye raporlama; batch size, sıra, warm-up ve repeat kimlikleri kaydedilsin.
- p50/p90/p95, başarısız istek süresi, başarısızlık oranı ve örnek sayısını ver. Eski veride ölçülmeyen stage'i 0 ile doldurma; “unknown/not measured” kullan.
- Aynı hedef tokenizer'ıyla gerçek render/prompt tokenları üzerinden saving hesapla. Ortalama vaka saving'i ve toplam token ağırlıklı saving'i ayır.
- Yeni ölçüm kodunu fake clock/backend ile doğrula; bu sprint yeni GPU benchmark çalıştırmayacak.

### C. QASPER evidence tanımı

- Mevcut kelime örtüşmesini sürümlü `lexical_evidence_coverage` olarak koru. %80 eşikli metriğin adını gerçek anlamına uygun yap.
- Anotasyonun izin verdiği yerde paragraph/span evidence recall ve bütün gerekli evidence'ın birlikte korunması için ayrı evaluator ekle. Temsillerin uygun olmadığı compressed metinde token kelime kümesini span kanıtı yerine geçirme; ölçümün uygulanabilirlik sınırını kaydet.
- Çoklu anotasyon alternatiflerini birleştirme/alternatif olarak değerlendirme politikasını resmi dataset yapısına uygun seç ve belgele.
- Unanswerable/evidence bulunamayan vakaları dahil etme/dışlama sayıları görünür olsun; abstention metriği generation çıktısı varsa ayrı hesaplansın.
- Negation/sıra bozulması, aynı kelimelerle farklı cümle, tek eksik evidence ve çoklu anotasyon için küçük elle doğrulanabilir fixtures ekle.
- İnsan değerlendirmesi yapılmış gibi davranma; rubrik ve gelecekte etiketlenecek audit örneklerini hazırla.

### D. Knapsack sonuç semantiği

- `skipped_by_state_limit`, gerçek `timeout`, memory/error ve completed durumlarını ayır. State sınırında hiç çalışmayan DP'yi timeout diye raporlama.
- Üretimdeki DP için candidate pruning ve effective granularity'nin exact garantiye etkisini metadata'ya yaz.
- Exact garanti yalnız gerçek orijinal ağırlıklarda çözülmüş aday problemine ait olsun; corpus genel optimum diye sunma.
- Küçük exhaustive oracle ile objective/feasibility kontrolü; yaklaşık modun bütçeyi aşmadığını kontrol et.

### E. Mevcut pilotun sürümlü yeniden analizi

- `docs/audit/reanalyze_pilot.py` başlangıç kanıtıdır; gerekiyorsa ortak evaluator'a taşı ama çalıştırma yolunu koru.
- Ana 83 vaka için tüm vaka doğruluğu, paired fark, exact McNemar ve seed'li bootstrap üret.
- Bağımsız vaka/ortak belge varsayımını açık yaz; cluster bilgisi yoksa cluster-corrected iddiası üretme.
- Equal accuracy'yi equivalence sayma; cascade'in vaka kazanım/kayıplarını göster.
- Yeni analiz çıktısını ayrı bir metric-v2/audit dizinine yaz; tarihi raw/summary dosyalarını değiştirme. Kaynak hash ve evaluator sürümü olsun.

## Kabul kriterleri

1. Parse failure, missing result, duplicate ID ve sıfır run senaryoları için rapor doğru denominator/statü üretiyor.
2. Tarihsel ana pilot full=32/83, dense=34/83, TP=37/83, compression-only=30/83, cascade=37/83 değerlerine izlenebilir biçimde ulaşıyor; farklıysa ham veride nedenini kanıtla.
3. Paired analiz yaklaşık p=0,302 ve p=0,581 sonuçlarını kaynak hash/seed ile yeniden üretiyor; aralıklar keşifsel olarak etiketli.
4. Ölçülmemiş preprocessing sıfır maliyet sayılmıyor; yeni stage toplama ve amortizasyon testleri geçiyor.
5. Lexical proxy ve gerçek evidence metriği isim/şema/testte ayrılmış.
6. Skip/timeout ve exact/approx ayrımı JSON/CSV/tabloda tutarlı.
7. Tarihsel ham dosyalar korunmuş, yeni metric-v2 üretimi deterministik ve ilgili genel testler yeşil.

## Teslim

Çalışan evaluator/report değişiklikleri, küçük fixtures, yeniden analiz çıktıları, metrik migration notu ve `docs/sprints/results/sprint-03.md`. Sprint 04'e metric schema, missing-data politikası, stage timing schema ve tekrar üretim komutlarını devret. `STATUS.md` güncelle.
