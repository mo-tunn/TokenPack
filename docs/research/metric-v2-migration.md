# Metric-v2 geçişi

## Arayüzler

- `submission.evaluation.core.summarize(rows, expected=None, methods=None)` ve `compare(rows, method, baseline, expected=None, ...)` model bağımsızdır. Manifest satırları en az string `case_id` ve `method` taşır; doğrulanmış belge kimliği `cluster_id` olabilir.
- LongBench `summarize_rows`, `summarize_grounded_rows`, `pairwise_rows`, `pairwise_grounded_rows` aynı `expected` parametresini kabul eder. Ana runner task listesini geçirir; yerel özetleyici yanındaki task JSONL'yi kullanır. Manifest yoksa all-planned metriği üretmez.
- Yeni özetlerde eski `accuracy` ve `grounded_accuracy` yoktur. `accuracy_all_planned`, `accuracy_all_observed`, `accuracy_answered_only`, `verified_correct_supported_all_planned` kullanılır. Null CSV'de boş hücre, JSON'da null'dur.
- `total_latency_seconds` tarihsel reader'da desteklenir. Yeni üretimde `recorded_stage_total_seconds` kullanılır; özetler `avg_recorded_stage_total_seconds` ve `recorded_stage_speedup_vs_full` içerir.
- QASPER scalar çıktı adları lexical proxy anlamını yansıtır. Eski CSV'ler `rename_legacy_metrics` ile okunabilir. `_evidence_recall` yardımcı fonksiyonu tarihsel aritmetiği korumak için değiştirilmedi.
- QASPER compression runner `qasper_evaluation_records.jsonl` ve `evaluation_manifest.json` üretir. Metinler, segmentler, anotasyon grupları, güvenilir kaynak aralıkları, config hash'i, tokenizer ve timing burada saklanır. Kimlik eşleştirmesi paper+question+setting üzerinden yapılır.

## Veri sınırları

Canonical LongBench: 415 sonuç, 83 vaka, 5 yöntem. Compression-only doğruluk 30/82 yerine 30/83'tür. Canonical QASPER: iki hybrid-greedy raw CSV'den toplam beş setting, her birinde aynı 200 soru ve 77 belge. Diğer tarihsel pilotlar otomatik olarak yeni benchmark sonucuna dönüştürülmez.

Yeni verbatim/span metrikleri eski scalar CSV'den hesaplanamaz. Tarihsel prompt tokenları, uçtan uca zaman, model/tokenizer revision'ları veya insan yargıları uydurulmaz. Model backend'i çağrılmadan çalışan fixture testleri gerçek model kalitesi kanıtı değildir.

## Artifact kullanımı

Canonical tabloların kaynağı `submission/results/metric_v2/canonical/` dizinidir. `manifest.json` kaynak ve evaluator hash'lerini, `integrity.json` veri kontrollerini, `unavailable.json` eksikleri içerir. Tekrar koşuda sayısal çıktılar deterministiktir; commit/dirty durumunu içeren provenance kaydı çalışma ağacına göre değişebilir.

LaTeX kaynak tabloları güncellenmiştir; eski PDF ve eski görseller tarihsel artifact olarak korunur. README eski headline görsellerini artık göstermez. Eski PDF'nin yeni metriklerle yeniden derlenmiş olduğu iddia edilmez.
