# TokenPack değerlendirme protokolü v1

Durum: ölçüm sözleşmesi. Tarihsel pilotların yeniden analizi keşifseldir; bağımsız test veya ön kayıt değildir. Yeni nihai deney henüz tasarlanmamış ve çalıştırılmamıştır.

## Araştırma soruları

Ana soru: Sorguya duyarlı hibrit kanıt seçimi, aynı token bütçesindeki retrieval yöntemlerine kıyasla cevap kalitesini ne ölçüde korur veya artırır?

İkincil soru: Seçimden sonra sıkıştırma, cevap kalitesi ile gerçekleşen token tasarrufu ilişkisini nasıl değiştirir?

Ana karşılaştırma TokenPack ile bütçeli dense retrieval'dır. Full context, yalnız sıkıştırma ve cascade ikincil karşılaştırmalardır. Knapsack araştırmanın ana katkısı veya zorunlu deney kolu değildir; mevcut kod tarihsel karşılaştırmalar için korunur. Üretim skorları/seçicisi bu paket tarafından değiştirilmez.

## Vaka ve doğruluk sözleşmesi

- Beklenen vaka–yöntem çiftlerinin kaynağı task manifestidir. Boş/eksik kimlik, duplicate ve manifest dışı sonuç açık hatadır. Çiftlerin farklı vaka kümeleriyle kıyaslanması reddedilir.
- `accuracy_all_planned = correct / planned`. Parse failure, execution failure ve missing result doğru sayılmaz; her biri ayrıca raporlanır. Eksik/başarısız yürütme bulunan koşu `partial` olarak işaretlenir.
- `accuracy_answered_only` ikincil, koşullu metriktir. Manifest yoksa all-planned değeri null olur; `accuracy_all_observed` ve `observed_only_exploratory` etiketi kullanılır.
- Planlanmamış yöntem `not_run` ve null doğrulukla gösterilir. Planlanan fakat sonucu olmayan yöntem partial'dır; sıfır başarısı tüm koşunun tamamlandığı anlamına gelmez.
- Judge değerlendirmesinde dört destek bayrağı geçerli olmalıdır. Eksik veya parse edilemeyen judge sonucu unknown'dur; hallucination veya desteklenmeme olarak etiketlenmez. Judge kapsamı ve değerlendirilebilir cevaplar üzerindeki oran ayrı gösterilir.
- `verified_correct_supported_all_planned` yalnız doğruluğu ve desteği doğrulanmış cevapları sayar. Unknown judge bu sayaca katkı yapmaz; bu, unknown cevapların gerçekten desteksiz olduğu iddiası değildir.

## Kanıt ölçümleri

Tarihsel lexical algoritma değiştirilmeden korunur: normalize edilmiş tam metin eşleşmesi yoksa kanıt kelime kümesinin bağlam kelime kümesindeki kapsamı hesaplanır. `lexical_evidence_coverage`, `lexical_coverage_ge_080` ve `answer_context_token_overlap_f1` yardımcı proxy'lerdir. Sonuncusu üretilmiş cevap F1 değildir. Lexical proxy karşılaştırılabilirlik için tarihsel ilk geçerli anotasyonu kullanır; yeni metrikler tüm geçerli alternatifleri saklar.

Yeni verbatim ölçüm yalnız NFC Unicode ve boşluk normalizasyonu kullanır. Sayılar, olumsuzluklar, harf büyüklüğü ve kelime sırası değiştirilmez. Bir kanıt parçası tek segmentte kesintisiz bulunmalıdır; ayrı seçilmiş chunk'lar üzerinden sahte span kurulmaz.

Her answerable anotasyon ayrı bir alternatif kanıt kümesidir. Ana recall alternatifler arasındaki en yüksek recall'dır. Complete-set başarısı en az bir alternatifin bütün parçalarının korunmasını gerektirir. Anotasyon bazlı değerler JSONL'de korunur. Unanswerable, eksik anotasyon, kanıtsız veya cevapsız sorular dışlama nedeni ile sayılır; bu sayaç taranan belgelerde incelenen anotasyonları kapsar, çalıştırılan soru limitiyle karıştırılmaz.

Source-span coverage, normalize edilmiş kaynak üzerindeki yarı açık karakter aralıklarının birleşim kapsamıdır; tekrar eden aralıklar çift sayılmaz. Kaynak ve seçilen metin için tekil tam eşleşme yoksa eşleme güvenilir kabul edilmez. Sıkıştırma kaynak eşlemesi sağlamıyorsa span metriği null olur. Verbatim kaybı anlamsal kayıp demek değildir.

Eski scalar CSV metni geri getirmez. Böyle kayıtlardan verbatim/span sonucu üretilmez; `unavailable_missing_text` yazılır. İnsan değerlendirmesi veya semantic judge bu pakette çalıştırılmaz.

## Bütçe ve zaman

Tarihsel %50 seçimi kaynak-token oranıdır. Seçili içerik, render edilmiş context ve tam generator prompt tokenları farklı alanlardır. Gelecek koşular context tokenizer'ını, tahmini regex fallback'i ve prompt tokenizer'ını belirtir. Revision bilinmiyorsa unknown kalır. Mevcut model istemcisi ham prompt encode sayısını kaydeder; chat-template kullanan gelecekteki profiller kendi gerçek prompt kapsamını ayrıca tanımlamalıdır.

Belge hazırlama (chunking ve belge embedding'i dahil), sorgu embedding'i, scoring, selection, render, compression ve generation ayrı aşamalardır. Belge maliyeti belge başına bir kez sahiplenilir; birçok sonuç satırında referans verilmesi maliyetin tekrar toplanmasına izin vermez. Gelecek çoklu-soru amortizasyonu bu kimliğe göre yapılmalıdır.

Ölçülmeyen süre null, çalıştırılmayan aşama skipped'dir. Sıkıştırıcının içindeki rendering `included_in_compression` olur; tekrar eklenmez. Aşama toplamı istek wall-time'ı değildir. Task-build ile uzak generation ayrı yürüdüğünden dış istek süresi bu akışta ölçülmemiştir. Batch ortalaması bireysel istek gecikmesi değildir.

Tarihsel 3.90× oranı yalnız kaydedilmiş selection + compression + generation sürelerini karşılaştırır. Chunking, embeddings, scoring ve model/cold-start maliyetlerini içermez. Ürünün uçtan uca hızlanması olarak sunulamaz.

## İstatistik

Ana tarihsel LongBench karşılaştırmaları TP–full, TP–dense ve cascade–TP'dir. Mutlak/göreli fark, kazanım/kayıp/beraberlik, 20.000 eşleştirilmiş percentile bootstrap (%95; seed 20260916) ve iki taraflı exact McNemar raporlanır. Baseline sıfırsa göreli fark null olur.

Doğrulanmış cluster bilgisi varsa bootstrap belge düzeyinde bütün soruları birlikte yeniden örnekler; istatistik soru ağırlıklı ortalama farktır. Tek cluster için aralık üretilmez. Cluster bağımlılığı varken bağımsız-vaka McNemar p-değeri üretilmez. LongBench'te doğrulanmış cluster kimliği yoktur; vaka düzeyi varsayımı açıkça yazılır.

Tarihsel karşılaştırmalar yöntem seçimi/çoklu deneme etkileri için düzeltilmiş değildir. P-değerleri doğrulayıcı üstünlük iddiasına dönüştürülmez. Eşit toplamlar equivalence/non-inferiority kanıtı değildir; bunun için önceden belirlenmiş marj ve ayrı deney gerekir.

## Yeniden üretim ve sonraki kapı

`python -m submission.evaluation.reanalyze` yerelde model yüklemeden canonical çıktı üretir. `--output-dir` ayrı çıktı dizini seçer. `--update-paper-tables` dört ilgili LaTeX tablosunu günceller. Kaynak JSONL/CSV'ler değişmez. Manifest hash'leri, evaluator sürümü, seed, etkin tarihsel ayarlar ve bilinmeyen alanlar raporlanır.

Bağımsız final değerlendirmeden önce ortak ürün/deney pipeline profili, belge düzeyinde dev/test ayrımı, sabit baseline'lar ve güç analiziyle örneklem seçimi yeni protokol sürümünde tamamlanmalıdır. Bu aşamada model, örneklem sayısı, non-inferiority marjı veya yeni başarı sonucu sabitlenmiş değildir.
