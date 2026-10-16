# TokenPack: repo sağlığı ve araştırma geliştirme planı

Tarih: 16 Eylül 2026. İncelenen HEAD: `0061578`. Paket sürümü: `0.1.2`.

## 1. Yönetici özeti

TokenPack'in sağlam bir uygulama çekirdeği ve kapsamlı deney birikimi var. Ortak `pack_source` servisi, CLI/MCP ayrımı, yapı farkındalıklı parçalama, açıklanabilir hibrit skorlar ve ham deney çıktıları önemli güçlü yönler. Yakın commitlerdeki atomik indeks yazımı, eşzamanlı embedding cache birleştirmesi ve manifest düzeltmeleri de doğru yönde.

En büyük ilerleme fırsatı yeni seçici eklemekten önce **ürün davranışı, deney ayarları ve ölçüm tanımlarını aynı sözleşmede buluşturmak**. Mevcut test başarısı bazı kritik sınırları korumuyor; araştırma sonuçlarının olumlu sinyali ise örneklem ve ölçüm sınırlamaları nedeniyle henüz genel bir üstünlük iddiasını desteklemiyor.

Önerilen sıra:

1. MCP dosya sınırı, gerçek çıktı token bütçesi ve sıkıştırma profillerini düzelt.
2. Yanıt doğruluğu paydası, gecikme kapsamı ve kanıt koruma metriklerini düzelt; mevcut ham çıktılardan sonuçları yeniden üret.
3. Sürüm ve konfigürasyonları sabitle; makale/README tablolarını tek kaynaktan üret.
4. Ayrı geliştirme ve değerlendirme kümeleriyle güçlü taban çizgileri kur.
5. Yeni skor/çeşitlilik/bağımlılık fikirlerini kontrollü deneylerle değerlendir.

### İncelemenin kapsamı ve sınırları

- Paket, testler, CI, paketleme, agent rehberi, LaTeX makalesi, deney harness'leri ve sonuç dosyaları incelendi.
- Windows / Python 3.13.1 üzerinde mevcut 120 test çalıştırıldı; tamamı geçti.
- Ayrı coverage dosyasıyla ölçüm: 2.883 statement, 47 eksik, yaklaşık **%98,37 satır kapsamı**. Araştırma kodu bu kapsamın dışında.
- Testlerde 368 `pathspec` deprecation uyarısı üretildi.
- 83 örnekli ana LongBench pilotunun 415 ham sonuç satırı yeniden analiz edildi.
- Sentetik dosyalar ve sahte embedding ile iki davranış hatası çalıştırılarak doğrulandı; gerçek model kalitesi bu deneylerle ölçülmedi.
- GPU/Modal deneyi, model indirme, ücretli servis çağrısı, yayınlama veya kaynak kod düzeltmesi yapılmadı.
- Yereldeki beş izlenmeyen Word dosyası mevcut kullanıcı çalışması olarak korundu; tez dosyalarının içerik karşılaştırması bu raporun kapsamı dışında. Makale değerlendirmesi `submission/paper/main.tex` üzerinden yapıldı.
- Uzak CI geçmişi, branch protection ve yayımlanmış wheel'in içeriği doğrulanmadı. Yerel checkout'tan bunların açık/kapalı olduğu sonucu çıkarılmadı.
- Bulgular **çalıştırılarak doğrulandı**, **koddan doğrulandı** veya **ayrıca doğrulanmalı** biçiminde ayrıştırılmıştır.

## 2. Mimari ve mevcut durum

Akış: dosya çıkarımı → bloklar → yapısal/semantik parçalama → embedding → evidence-hybrid skor → bütçeli seçim → isteğe bağlı sıkıştırma → Markdown.

| Katman | Mevcut durum | Değerlendirme |
|---|---|---|
| Giriş | CLI ve yerel stdio MCP, ortak packing servisi | Tek servis üzerinden ilerlemek doğru; deney harness'i aynı profili kullanmıyor |
| Çıkarım | PDF, Office, kod, metin, yapılandırılmış dosyalar | Geniş kapsam; gerçek dosya fixture'ları ve hata raporlama kritik |
| İndeks | JSON indeks, dosya hash'li incremental manifest | Küçük/orta veri için anlaşılır; büyük corpus'ta bellek ve yazma maliyeti ölçülmeli |
| Embedding | MiniLM, model+metin hash cache | Model revision/tokenizer kimliği sözleşmede eksik |
| Skor | Cosine/BM25/query coverage/structure/coherence | Şeffaf; ağırlıklar elle seçilmiş, sistematik kalibrasyon yok |
| Seçici | Üretimde budget-top-k; araştırmada DP/MMR/varyantlar | Pratik varsayılan makul; optimizer ile skor katkısı ayrılmalı |
| Sıkıştırma | LLMLingua entegrasyonu | Model/algoritma bayrakları ve makale profili uyumsuz |
| Test | 120 test, yüksek satır kapsamı | Çok sayıda mock; entegrasyon güvencesiyle eş anlamlı değil |
| CI | Ubuntu, Python 3.10–3.12, coverage tabanı %70 | Platform, paket kurulum ve opsiyonel gerçek backend kontrolleri eksik |
| Deney arşivi | Çok sayıda ham JSONL/CSV ve readout | Yeniden analiz edilebilirlik güçlü; canonical koşu/provenance yönetimi zayıf |

İzlenen 458 dosyanın mevcut çalışma ağacı toplamı **268.150.824 bayt (~255,7 MiB)**. Bu `.git` geçmiş boyutu değildir. En büyük tek task JSONL yaklaşık 19,5 MB; birçok koşu aynı kaynak bağlamları tekrar saklıyor. Araştırma çıktıları ana kod checkout'unu büyütüyor.

## 3. Repo sağlığı: öncelikli bulgular

### R1 — MCP kök klasör paketlemesi workspace dışına çıktı yazıyor — P0

**Durum: çalıştırılarak doğrulandı.** `mcp_server.py::pack_context_tool`, kullanıcıdan gelen `source` ve açık `out` yolunu doğruluyor. Ancak `out` verilmezse `_infer_pack_output_path` ile üretilen yolu tekrar workspace kontrolünden geçirmiyor. Kaynak workspace'in kendisiyse çıktı kardeş yola, örneğin `parent/workspace-tp.md` konumuna gidiyor.

Sentetik geçici çalışma alanında `source='.'` çağrısı ile `MCP_OUTPUT_OUTSIDE=True` elde edildi. Bu, README'nin varsayılan okuma/yazma sınırı sözünü ihlal ediyor.

**Düzeltme:** nihai çıktı, indeks, seçim, cache ve manifest yollarını oluşturma öncesinde aynı path politikasından geçir. Workspace kökü için varsayılan çıktı workspace içinde olmalı veya açık hata vermeli. `allow_any_path` istisnasını merkezi tut.

**Kabul:** `source='.'`, alt klasör, mutlak/bağıl yol, `..`, symlink hedefi ve açık `out` için sınır testleri; varsayılan modda dışarıda hiçbir dosya oluşmaması.

**İlişkili risk:** `loaders.py::iter_supported_files` içindeki dosya adaylarına resolve+containment uygulanmıyor. `os.walk` dizin linklerini normalde izlemese de dosya symlink'i başka yerden veri okuyabilir. Yerel Windows symlink oluşturma denemesi yetki/ortam nedeniyle çalışmadı; bu alt bulgu kod incelemesi düzeyinde ve Linux entegrasyon testi gerektiriyor. Cache klasörünün link olması da aynı politikaya dahil edilmeli.

### R2 — “Strict token budget” yalnızca chunk toplamına uygulanıyor — P0

**Durum: çalıştırılarak doğrulandı.** `packing.py` bütçeyi `result.used_tokens` üzerinden denetliyor, sonra sorgu, başlık, kaynak yolları ve ayırıcıları ekliyor. Son Markdown yeniden sayılmıyor.

Kontrollü örnek: bütçe **50**, seçilen içerik **7**, yazılan Markdown **202 token**. Uzun sorgu kullanılarak eksik muhasebe görünür kılındı. Normal kısa sorgularda da başlık maliyeti sıfır değil.

`TokenCounter` ayrıca varsayılan olarak `cl100k_base` kullanıyor; yüklenemezse sessizce regex sayımına dönüyor. Regex tahmini hedef modelin token sayısı için güvenli üst sınır değil. Qwen deneyinde bu sayaç Qwen tokenizer'ıyla aynı kabul edilemez.

**Düzeltme:** `source_tokens`, `selected_content_tokens`, `rendered_context_tokens`, `prompt_tokens`, `reserved_output_tokens` ayrımı. Hedef tokenizer kimliği/revision'ı metadata'ya yazılmalı. Tam prompt garanti edilecekse sistem mesajı ve chat template maliyeti de gerekli; bunlar bilinmiyorsa garanti yalnızca tanımlı artifact bütçesiyle sınırlanmalı. Render sonrası sayım ve gerekirse son chunk çıkarma/yeniden seçme döngüsü. Strict modda tahmini sayaçla sessiz devam edilmemeli.

**Kabul:** temiz/debug/none çıktı, uzun sorgu/yol, Unicode/Türkçe/kod, küçük bütçe ve compression dahil tüm akışlarda tanımlanan nihai sınır aşılmıyor; metadata gerçek ölçüm yöntemini söylüyor.

### R3 — README'nin sıkıştırma komutu makale koşulunu temsil etmiyor — P0

**Durum: konfigürasyon farkı koddan doğrulandı; gerçek backend çağrısının hata biçimi ayrıca doğrulanmalı.** Üretim varsayılanı LLMLingua-2 BERT modeli; `llmlingua2=False`. README `--longllmlingua` öneriyor fakat nedensel model seçmiyor. Microsoft örneğinde LLMLingua-2 modeli `use_llmlingua2=True` ile kullanılıyor; LongLLMLingua ayrı akış.

Daha önemlisi, `longbench_eval/client.py::_compress` **context-level filtre açık, token-level filtre kapalı**, `llmlingua2=False` kullanıyor; istemci model varsayılanı `gpt2`. Ürün `pack_source` varsayılanında context-level filtre kapalı, token-level açık. Makale bu sınırlamayı yazıyor; README'deki “best combination” doğrudan aynı koşulmuş gibi sunuluyor.

**Düzeltme:** adlandırılmış, doğrulanan profiller: `selection-only`, `llmlingua2`, `longllmlingua-context`, `longllmlingua-token`. Her profil model ailesi, filtreler, yeniden sıralama, rate/target, device ve revision'ı belirlemeli. Çelişkili bayraklar pahalı indeksleme başlamadan reddedilmeli. Makale profilini hem CLI hem harness aynı nesneden almalı.

**Kabul:** her desteklenen profil için gerçek küçük model smoke testi; README komutunun etkin konfigürasyonu canonical deney manifestiyle eşleşiyor. Tarihsel sonuçlar sessizce yeni koşul etiketiyle yeniden sunulmuyor.

### R4 — Ürün/deney parçalama ve sıralama farkları — P1

Ürün varsayılanı `low-budget` 250/40/320 token; LongBench istemci varsayılanları 650/120/900. `select_chunks` ve `render_context` kaynak sırasına dönüyor; 83 örneklik task raporu `score-then-source` bildiriyor. Deney render'ı chunk id, skor ve density de ekliyor; kullanıcı çıktısı temiz başlıklar taşıyor.

Bu farklar ölçülen sistemi değiştiriyor. **Çözüm:** ortak `PipelineConfig` ve renderer; `paper-v1` değişmez tarihsel profil, `production-v1` ayrı profil. Eşit token bütçesinde ordering ablation; varsayılanı ancak değerlendirmeden sonra değiştir. Kabul: aynı fixture ve profil CLI/API/harness üzerinden aynı chunk listesi ve bağlam hash'ini üretiyor.

### R5 — Yazma ve overwrite sözleşmesi tamamlanmamış — P1

İndeks/cache/manifest atomik yazmaya ilerlemiş; `packing.py` seçim JSON'u ve son Markdown için doğrudan `write_text` kullanıyor. `exists()` ile yazma arasındaki pencere eşzamanlı iki çağrının `overwrite=False` korumasını aşmasına izin verebilir. Açık artifact yollarında kaynakla çakışma da önceden reddedilmeli.

**Plan:** kaynak/çıktı/index/selection için alias kontrolü; geçici dosya+atomik commit; overwrite yasağını atomik claim/lock ile koruma; başarılı/başarısız koşu durum kaydı. Kabul: iki işlem aynı çıktıya koştuğunda yalnızca biri kazanır; yarıda kesilen yazım geçerli önceki çıktıyı bozmaz; `out=source` güvenli biçimde reddedilir.

### R6 — Cache doğruluğu ve ölçek — P1/P2

`EmbeddingCache` anahtarı model adı+metin hash'i. Manifest konfigürasyonu tokenizer kimliğini, model revision'ını ve extractor sürümünü taşımıyor. Aynı model adı altında model değişimi veya regex→tiktoken geçişi eski embedding/token ağırlıklarının kullanılmasına neden olabilir.

Cache bütünü JSON olarak okunup yazılıyor; büyük dosya ağacında her yeni dosyanın embedding yazımı büyüyen cache'i tekrar serileştiriyor. Bu ölçülmüş bir performans regresyonu değil; ölçülmesi gereken ölçek riski.

**Plan:** sürümlenmiş provenance anahtarı; invalidation testleri; tekrar eden missing metinleri tekilleştirme; 1k/10k/100k chunk için soğuk/sıcak indeksleme, RSS ve yazılan bayt ölçümü. Sonuç gerekçelendirirse SQLite veya parça bazlı cache. Büyük yeniden yazımdan önce profil çıkar.

### R7 — Test kapsamı yüksek, davranış güvencesi eksik — P1

README rozeti %99; bu ortamda ölçülen satır kapsamı %98,37. CI eşiği %70. Bu fark tek başına kötü kalite göstergesi değil; sabit rozet güncel koşuyu temsil etmiyor. `scoring_experimental.py` kapsam hesabına dahil, `submission/` değerlendirme kodu dahil değil. `test_coverage_boost.py` içinde çok sayıda sahte Office/PDF/LLMLingua/SentenceTransformer backend var.

**Plan:**

- Hızlı birim testlerini koru; güvenlik, bütçe, gerçek minimal dosya fixture'ları, CLI ve MCP entegrasyonunu ayrı katman yap.
- Seçiciler için küçük problemde brute-force oracle: kapasiteye uyma, benzersiz seçim, deterministik tie-break, optimality yalnızca exact modda.
- Gerçek eşzamanlı process testleri; tokenizer/cache invalidation testleri.
- Araştırma metrikleri için parse failure, eksik koşu, duplicate ID, farklı vaka kümeleri ve rapor yeniden üretim testleri.
- Coverage XML'den rozet veya CI sonucu; ölçülen tabanın altına düşüşü kontrollü engelle. Branch coverage ayrı raporlansın; %100 uğruna anlamsız test eklenmesin.
- `pathspec` API'sini sabitlenmiş uyumlu sürümde güncelle, nested ignore davranışını koru.

### R8 — Paketleme, CI ve bakım altyapısı — P1/P2

Mevcut CI sadece Linux ve Python 3.10–3.12. Yerel 3.13 test başarısı gerçek ML bağımlılıklarıyla tam destek kanıtı değil. Dev kurulumunda araştırma scriptlerinin tüm bağımlılıkları tanımlı değil; Modal image'larında alt sınırlar var fakat tam kilitli ortam yok. İncelenen iki Modal image listesinde `pathspec` açıkça yok; paket kurulumu ve import smoke kontrolüyle temiz image kurulabilirliği doğrulanmalı.

**Plan:**

1. Runtime bağımlılık aralıkları ve araştırma lock dosyasını ayır; platforma uygun kilitli CPU/GPU profilleri oluştur.
2. Wheel/sdist build ve temiz venv'de wheel'den CLI/import smoke; editable kuruluma bağımlılığı yakala.
3. Windows+Linux hızlı test; desteklenecek Python sürümlerini gerçek dependency smoke ile belirle.
4. Ruff ve kademeli tip kontrolü; zorunlu olmayacak pahalı model smoke'larını manuel/gecelik job olarak ayır.
5. Dependency/secret taraması ekle; bir tarama yapılmadan “zafiyet yok” sonucuna varma.
6. `CONTRIBUTING`, `CHANGELOG`, `SECURITY`, deney rehberi ve `CITATION.cff`; agent rehberini standart `AGENTS.md` keşfine uygun hale getir.
7. Release checklist: paket kurulumu, README örneği, profile manifesti, sonuç tablosu hash'leri, lisans dosyasının artifact içinde bulunması.

### R9 — Artifact yönetimi — P2

Ana Git'te kod, küçük fixtures, özetler ve yeniden üretim manifestleri; büyük dataset/task içerikleri sürümlü artifact deposunda tutulabilir. İlk adım taşıma/silme değil, run registry ve referans bütünlüğü kurmaktır.

Her koşu için: `run_id`, amaç, canonical/exploratory/smoke statüsü, commit+dirty diff hash, dataset split/revision/hash, vaka listesi, model/tokenizer revision, tam config, environment lock, donanım, prompt/parser sürümü, ham çıktı hash'i. İndirme komutu ve checksum olmadan büyük dosyaları Git'ten çıkarmamak gerekir. Lisans/yeniden dağıtım izinlerini veri kaynağı bazında kayıt altına almak ayrıca gerekli; mevcut depolamanın hukuki uygunluğu bu incelemede değerlendirilmedi.

## 4. Araştırma sonuçlarının yeniden değerlendirilmesi

### 4.1 Ana 83 örnek: sinyal var, üstünlük henüz kanıtlanmış değil

Kaynak: `submission/results/longbench_v2_modal_hybrid_greedy_83_latency/longbench_generation_results.jsonl`.

| Yöntem | Doğru / tüm vakalar | Doğruluk |
|---|---:|---:|
| Full context | 32/83 | %38,55 |
| Dense production-RAG-50 | 34/83 | %40,96 |
| TokenPack-50 | 37/83 | %44,58 |
| Yalnız LongLLMLingua | 30/83 | %36,14 |
| TokenPack + LongLLMLingua | 37/83 | %44,58 |

Mevcut CSV yalnız LongLLMLingua için 30/82 = %36,59 raporluyor; bir parse failure dışlanmış. `eval_utils.py::summarize_rows` doğruluğu yalnız prediction bulunan satırlarda hesaplıyor. **Birincil metrik planlanan tüm vakalar üzerinde olmalı**, parse/API failure ayrıca raporlanmalı. Başarılı cevap koşullu doğruluk ancak ikincil metrik olabilir. Yöntemler arasında eksik vaka silerek avantaj üretmemek için ortak case manifesti gerekir.

| Eşleştirilmiş karşılaştırma | Kazanç / kayıp / beraberlik | Mutlak fark | %95 bootstrap aralığı | Exact McNemar p |
|---|---:|---:|---:|---:|
| TokenPack − full context | 10 / 5 / 68 | +6,02 yüzde puan | −2,41 … +15,66 pp | 0,302 |
| TokenPack − dense RAG | 8 / 5 / 70 | +3,61 yüzde puan | −4,82 … +12,05 pp | 0,581 |
| Cascade − TokenPack | 8 / 8 / 67 | 0,00 yüzde puan | −9,64 … +9,64 pp | 1,000 |

Hesap: 20.000 vaka düzeyinde eşleştirilmiş percentile bootstrap, seed `20260916`; iki taraflı exact McNemar. Analiz keşifseldir; tekrar tekrar denenen konfigürasyonlar için seçim düzeltmesi veya ortak belge kümeleri için cluster düzeltmesi uygulanmadı. Sayılar bağımsız doğrulama deneyi yerine geçmez.

**Önemli yorum:** +%15,6 göreli artış aritmetik olarak doğru: 5/32. Okuyucunun etkiyi anlayabilmesi için yanında +6,02 yüzde puan ve 37/83–32/83 yazılmalı. Aynı toplam doğruluk, aynı vakaların korunduğu veya kalite eşdeğerliği anlamına gelmiyor: cascade 8 cevabı kazanırken 8 cevabı kaybediyor. “Kalite kaybı yok” iddiası önceden belirlenmiş eşdeğerlik/non-inferiority sınırı gerektirir.

### 4.2 Örneklem genel LongBench performansını temsil etmiyor

Task raporu 493 taranan, 410 fazla uzun olduğu için elenen, 83 kullanılan vaka gösteriyor. Token penceresi 8k–24k. Ham sonuçların **83/83'ü `length='short'`**.

Alan dağılımı: 49 single-document QA, 25 multi-document QA, 4 dialogue, 3 in-context learning, 1 structured data, 1 code repository. Kod/çok dilli/Office kullanım kalitesi buradan çıkarılamaz. Makaledeki 30 ve 8 vakalık uzun bağlam tanılamaları yararlı, fakat alan genellemesini çözmüyor.

**Plan:** uzunluk×alan katmanlarıyla vaka manifesti; tüm dışlama nedenleri; geliştirme/test ayrımı; aynı belge ailesinin iki tarafa düşmesini engelleme. Önceden incelenen pilotu geliştirme verisi say. Nihai örneklem büyüklüğünü gözlenen discordance ve hedeflenen pratik farkla güç simülasyonundan türet; “200 yeter” gibi keyfi sayı belirleme. Veri yetmiyorsa daha geniş güven aralıklarını ve iddia sınırını kabul et.

### 4.3 3,90× gecikme ölçümü tam kullanıcı yolunu kapsamıyor

CSV'de TokenPack preprocessing ortalaması yaklaşık 0,0000215 saniye: seçici süresi. `client.py` parçalama, block/chunk embedding, query embedding ve skorlamayı bundan önce yapıyor. `app.py` toplamı seçim+sıkıştırma+üretim olarak topluyor. `_compress` içindeki hazırlık/parçalara ayırma da timer başlamadan yapılıyor. Cold start/model load dışlanmış; README bunu belirtiyor, ancak diğer hazırlık maliyetleri de kapsam dışında.

**Düzeltme:** `extract`, `chunk`, `document_embed`, `query_embed`, `score`, `select`, `compress_prepare`, `compress`, `render`, `prefill/decode` veya toplam generation ve I/O zamanlarını ayrı kaydet. Üç kullanım modu raporla:

- İlk kez açılan belge: tam uçtan uca maliyet.
- İndeksi hazır belge, yeni sorgu: yalnız amortize edilemeyen maliyet.
- Hazır packed context: generation-only maliyet.

Warm-up, yöntem sırası randomizasyonu, tekrarlı ölçüm, aynı donanım/model/decoding, batch=1 latency ve ayrı throughput testi. p50/p90/p95 ve RAM/VRAM. API maliyetinde giriş+çıkış+kompresör/local compute ayrımı. Kaynak token tasarrufunun ortalaması ile toplam token tasarrufu oranını ayrı ver; 1M-token maliyet örneğini hangi ağırlıklandırmanın ürettiğini açıkla. Eski 3,90× sonucunu silmeden “önceden hazırlanmış koşulda ölçülen kısmi toplam” şeklinde etiketle.

### 4.4 QASPER kanıt metriği anlamsal sadakat ölçmüyor

`qasper_selector_eval.py::_evidence_recall` önce normalize substring, sonra evidence kelime kümesinin seçili metnin kelime kümesiyle örtüşmesini kullanıyor. Kelime sırası, bağlam, olumsuzlama ve hangi span'in korunduğu kaybolabiliyor. `complete_evidence_rate` eşiği **0,80**; isim tam korumayı çağrıştırıyor ama %100 span retention ölçmüyor. Compression özellikle kelime düzeyinde çalıştığı için bu proxy yöntem sıralamasını etkileyebilir; yanlılığın yönü ölçülmeden varsayılmamalı.

`_choose_answer` tek anotasyonu seçiyor; çoklu kabul edilebilir cevap/evidence alternatifleriyle uyum kontrol edilmeli. Kanıtı bulunmayan/unanswerable vakaların atılması kapsamı daraltıyor.

**Plan:** mevcut metriği `lexical_evidence_coverage` olarak açık tanımla ve tarihsel seriyi koru. Ek metrikler: anotasyonlu paragraph/span recall, tüm gerekli kanıtların birlikte korunması, answer EM/F1, cevaplanamayan sorularda abstention, citation precision/recall. 0,80 eşikli metriği `evidence_coverage_ge_0_8` gibi dürüst adla raporla. 50–100 vaka üzerinde kör insan kontrolüyle proxy uyumunu ölç; bu sayı metrik hata teşhisi içindir, üstünlük testine güç garantisi değildir. Makale düzeyinde kümelenmiş bootstrap kullan.

### 4.5 Sentetik knapsack çalışmasındaki “timeout” gerçekte önceden atlama

`knapsack_performance.py::dp_knapsack` `n*capacity > max_cells` durumunda algoritmayı hiç çalıştırmadan `(None,None,0.0)` dönüyor. Summary bunu `timeout_rate`, çizim “DP timeout” diye adlandırıyor. Bu gerçek zaman aşımı ölçümü değil. Makalenin bazı yerlerinde “skipped” doğru açıklaması mevcut, ancak tablo/özet terminolojisi tutarsız.

**Plan:** `skipped_by_state_limit`, `timeout`, `memory_limit`, `completed` ayrı statüler; gerçek timeout deneyi yapılacaksa subprocess, zaman/bellek sınırı ve gözlenen süre. Skip kayıtları latency ortalamasına sıfır olarak girmemeli. Mevcut uniform bağımsız value/weight örneklerine correlated, strongly correlated, subset-sum ve zor adversarial aileler ekle. `n` ve kapasiteyi ayrı tarayarak etkileri ayır; SA'nın güçlü greedy başlangıcına göre ek getirisi ve ek compute maliyetini göster. Küçük örneklerde exhaustive oracle ile exact çözümü doğrula.

Üretimdeki `_knapsack` ayrıca 10 milyon state üstünde granularity'yi otomatik büyütüyor. Böylece bütçe korunuyor fakat orijinal token probleminde exact optimum garantisi kalkıyor. Metadata `effective_granularity`, candidate sayısı, approx/exact durumu taşımalı. Candidate pruning de global optimum iddiasını zaten sınırlar. İki farklı DP implementasyonunun ayrı sözleşmeleri belgelenmeli.

## 5. Araştırmayı geliştirme programı

### 5.1 Araştırma sorularını ayır

1. **Skor sorusu:** hibrit skor dense-only/BM25'e göre aynı bütçede daha yararlı kanıt seçiyor mu?
2. **Optimizer sorusu:** aynı skor ve aday havuzunda DP, greedy/MMR'a göre ne kazandırıyor?
3. **Sıralama sorusu:** aynı seçili chunk kümesinin sırası cevap kalitesini değiştiriyor mu?
4. **Sıkıştırma sorusu:** selection→compression, eşit gerçekleşen prompt token sayısında compression-only'den daha iyi mi?
5. **Sistem sorusu:** kullanıcı yolunun tamamında kalite/latency/maliyet bakımından hangi yöntem Pareto sınırında?

Araştırmanın savunulabilir ana hipotezi: **“Hedef modelin gerçek prompt bütçesi altında, seçimi ve sıkıştırmayı ayrı kontrol etmek daha iyi kalite–maliyet çalışma noktaları sağlayabilir.”** Knapsack uygulaması faydalı matematiksel çerçeve; tek başına yeni algoritmik katkı veya genel üstünlük değil.

### 5.2 Kontrol edilmiş deney matrisi

| Deney | Sabit tutulacaklar | Değişken | Ana ölçüm |
|---|---|---|---|
| Skor ablation | Chunk, aday havuzu, greedy, sıralama, token bütçesi | Dense / BM25 / hibrit / bileşen çıkarma | Kanıt + cevap kalitesi |
| Seçici ablation | Aynı skor ve aynı aday ID listesi | Greedy-value / density / MMR / exact DP | Objective gap + kalite + seçim süresi |
| Chunking | Embedder, skor, seçici, bütçe | Fixed-token / paragraph / structure / structure+semantic | Span retention + kalite + toplam süre |
| Ordering | Seçilmiş küme birebir aynı | Source / score / score-then-source | Paired accuracy farkı |
| Compression | Aynı generator/prompt, eşleşen gerçek token kullanımı | Compression-only / selection-only / cascade | Kalite–token–latency Pareto eğrisi |
| Genelleme | Dondurulmuş konfigürasyon | Generator ailesi, alan, uzunluk, dil | Katmanlı performans ve failure rate |

Tam Kartezyen ürün çok pahalıdır. Önce CPU/retrieval proxy ile ele, sonra sabit geliştirme kümesinde generation smoke, en son önceden seçilmiş az sayıda yöntemle kapalı değerlendirme.

**Taban çizgileri:** full-context yalnız gerçekten sığan vakalarda; prefix; random feasible; dense retrieval+budget fill; BM25+budget fill; BM25+dense rank fusion; fusion+cross-encoder; MMR; LLMLingua-2; LongLLMLingua context-level ve token-level ayrı. `production-rag` etiketi yerine `dense-cosine-budget-fill` daha somut; güçlü üretim RAG sistemlerini temsil ettiği varsayılmamalı.

**Aday adaleti:** `_knapsack_candidate_pool` value ve density havuzlarını birleştirdiği için en fazla 2× aday görebiliyor; greedy tek havuz görüyor. Selector ablation'da birebir aynı aday listesi gerekli. Ayrı end-to-end karşılaştırmada aday üretimi yöntem bileşeni olabilir; maliyeti raporlanmalı.

### 5.3 Model ve tokenizasyon duyarlılığı

MiniLM model kartı 256 word-piece sonrası kesmeyi bildiriyor. Projedeki 320/900 token chunk sınırları farklı tokenizer biriminde; semantik embedding'e giren uzun bloklar daha da büyük olabilir. Bu yüzden her chunk'ın tamamının temsil edildiği varsayımı güvenli değil. [Resmi model kartı](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

Önce gerçek embedding tokenizer'ıyla `truncated_chunk_rate`, son bölgedeki kanıt kaybı ve block truncation ölç. Sonra tokenizer'a uyan chunk boyu, sliding-window pooling ve daha uzun bağlamlı embedder karşılaştır. Yalnız `max_seq_length` değerini büyütmek yeterli çözüm kabul edilmemeli. Türkçe ve kod alt kümeleriyle genelleme ayrıca ölçülmeli.

### 5.4 Ağırlık kalibrasyonu ve yenilik adayları

Mevcut 0,35/0,25/0,20/0,15/0,05 ağırlıklarını dev kümesinde sınırlı random search veya basit öğrenilen ranker ile karşılaştır. Belge bazlı ayrım, aynı tuning bütçesi, sensitivity grafiği ve kapalı test. Min-max normalizasyonunun alakasız/aykırı chunk eklenmesine duyarlılığını test et; metadata bulunmayan belgelerde structural prior katkısını ayrıca raporla.

Sonraki adaylar, önem sırasıyla:

1. **Dinamik yenilik/kapsama:** mevcut üretim `pack` redundancy penalty=0 ile çalışıyor; benzer kanıtın tekrar seçilmesi için MMR veya marginal coverage yaklaşımı. README'deki redundancy azaltma iddiasını ölç.
2. **Bağımlılık farkındalığı:** kod fonksiyonu+import, tablo+başlık, iddia+kanıt komşuluğu için group/precedence constraint. Daha iyi cevap üretip üretmediğini göster; sadece objective artışı yeterli değil.
3. **Sorguya göre bütçe politikası:** dev kümesinde marjinal kanıt kazanımı üzerinden stop kuralı; testte değişmeyen politika. %50 varsayımını 10/25/50/75% ve mutlak token bütçelerinde kıyasla.
4. **Seçim+sıkıştırma ortak optimizasyonu:** chunk başına farklı sıkıştırma seçeneklerini multiple-choice knapsack olarak modellemek. Önce seçilen rate'lerin kalite tahmini güvenilir olmalı; aksi halde optimizer yanlış hedefi daha iyi optimize eder.

Bu fikirler mevcut metrik/sözleşme problemleri çözülmeden ana geliştirme hattına alınmamalı.

### 5.5 İnsan değerlendirmesi ve istatistik protokolü

- Birincil metrik ve iki ana karşılaştırmayı önceden yaz; ek deneyleri keşifsel etiketle.
- QASPER'de belge, LongBench'te paylaşılan kaynak varsa kaynak kümesi düzeyinde yeniden örnekleme.
- Accuracy için eşleştirilmiş fark ve CI; discordant çiftlerde McNemar; çoklu doğrulayıcı karşılaştırmada Holm gibi önceden seçilmiş düzeltme.
- Eşdeğerlik hedefleniyorsa ürünün tolere ettiği kalite kaybını önceden belirle; “anlamlı fark yok” eşdeğerlik kanıtı değil.
- İnsan etiketlemesinde yöntem adı gizli, kaynak görünür, rubrik sabit, en az iki değerlendirici; anlaşmazlık ve adjudication kaydı.
- LLM judge tanısal araç olarak kullanılabilir; aynı generator/judge ailesinin yanlılığı ve insanla uyumu ölçülmeli.
- Error taxonomy: extraction, chunk boundary, retrieval miss, duplicate context, compression fact/negation loss, ordering, model reasoning, parse failure.

## 6. Uygulama planı ve kabul kapıları

Eforlar tek geliştirici için yaklaşık aktif iş günüdür; GPU kuyrukları, veri etiketleme ve model indirme süreleri dahil değildir. Toplam süre bir teslim taahhüdü değil, planlama aralığıdır.

| İş | Öncelik | Efor | Bağımlılık | Somut teslim ve kabul |
|---|---|---:|---|---|
| MCP containment ve path alias koruması | P0 | 1–2 | Yok | Kök/alt klasör/dış yol testleri, varsayılanda dış write yok |
| Nihai token bütçesi sözleşmesi | P0 | 2–4 | Yok | Hedef tokenizer, rendered count, strict mod invariants |
| Compression profilleri | P0 | 2–3 | Yok | Gerçek backend smoke, çelişkili model/flag erken hata |
| Doğruluk paydası ve ham sonuç reanalysis | P0 | 1–2 | Yok | Tüm vakalar denominator, raw→table deterministik |
| Canonical run manifesti | P1 | 2–3 | Profiller | Commit/model/data/tokenizer/config/hash kayıtları |
| Ortak ürün–harness pipeline | P1 | 3–5 | Bütçe+profiller | Aynı fixture/profil aynı seçili ID ve bağlam hash'i |
| Tam latency/cost ölçümü | P1 | 2–4 | Ortak pipeline | Soğuk/sıcak/hazır bağlam ayrı rapor |
| QASPER metric v2 ve insan kalibrasyonu | P1 | 3–5 | Vaka manifesti | Span ve lexical ayrımı, annotation protokolü |
| Atomic output/cache invalidation | P1 | 2–4 | Path politikası | Concurrent write ve revision invalidation testleri |
| CI/wheel/platform/araştırma smoke | P1 | 2–4 | Temel düzeltmeler | Temiz wheel kurulumu, Linux+Windows, metric testleri |
| Sentetik benchmark semantiği | P1 | 1–3 | Yok | Skip/timeout ayrımı, zor instance aileleri |
| Dev/holdout, baseline, güç analizi | P1 | 3–5 | Metric v2+manifest | Önceden dondurulmuş değerlendirme protokolü |
| Kontrollü generation benchmark | P2 | 3–7 + compute | Önceki kabul kapıları | Güçlü baseline, ≥2 model ailesi, paired CI |
| Skor/çeşitlilik ve embedding ablation | P2 | 3–6 + compute | Dev/holdout | Kazanç+overhead, negatif sonuçlar dahil |
| Artifact registry ve dokümantasyon | P2 | 2–3 | Manifest | Checksum ile indirilebilir koşular, claim–table eşleşmesi |

### Aşama A — İlk 1–2 hafta: güvenilir temel

R1/R2/R3, accuracy denominator, timeout adlandırması ve README iddia kapsamı. Kapı: güvenlik/bütçe/profile regresyonları yeşil; pilot tabloları ham veriden yeniden üretilebiliyor. Bu kapı geçmeden pahalı benchmark başlatma.

### Aşama B — Sonraki 2–3 hafta: deney altyapısı

Ortak profiller, manifest, locked environment, latency instrumentation, QASPER metric v2, wheel/CI ve split protokolü. Kapı: yeni bir makinede küçük frozen fixture koşusunun chunk IDs, prompt hash ve metrikleri tekrarlanıyor; nondeterministik backend ölçümleri ayrıca tanımlı toleransla değerlendiriliyor.

### Aşama C — Sonraki 3–5 hafta: kanıt üretimi

Güçlü baseline'lar, 2 model ailesi, tabakalı veri ve sınırlı ablation; insan audit'i. Kapı: birincil hipotez için paired CI ve failure analizi, tüm negatif sonuçlar arşivde. Üstünlük çıkmazsa daha az maliyetle kabul edilen kalite kaybı hedefi değerlendirilir; başarı tanımı sonuç görüldükten sonra değiştirilmez.

### Aşama D — Yayına hazırlık

README, görseller, LaTeX tabloları ve sonuç özetlerini canonical JSON'dan üret. Makale için araştırma sorularını netleştir: algoritma analizi/tez hedefi korunacaksa sentetik ve uygulama katkıları ayrı; RAG sistem makalesi hedeflenecekse kalite–maliyet protokolü merkezde olmalı. İki hedef tek “knapsack her şeyi iyileştiriyor” anlatısına sıkıştırılmamalı.

## 7. Hemen açılabilecek iş sırası

1. `fix/mcp-workspace-boundary`: R1 ve bağlantılı containment testleri.
2. `fix/rendered-token-budget`: R2, tokenizer metadata ve bütçe invariants.
3. `fix/compression-profiles`: R3 ve gerçek backend smoke.
4. `fix/eval-denominators`: tüm vaka doğruluğu, failed/parse ayrımı, boş yönteme N/A.
5. `research/pilot-reanalysis`: paired analiz, değişmez eski artifact, yeni metrik sürümü.
6. `refactor/shared-pipeline-profile`: ürün/deney sıralama ve render eşleşmesi.
7. `research/timing-and-provenance`: aşama süreleri ve kilitli manifest.
8. `research/frozen-evaluation`: güçlü baseline, split ve önceden kayıtlı protokol.

Bunlar iş başlıklarıdır; branch veya PR oluşturulmadı.

## 8. Dış kaynaklarla kontrol edilen noktalar

Literatür kontrolü hedefli bir birincil-kaynak kontrolüdür; Eylül 2026'ya kadar eksiksiz sistematik tarama iddiası taşımaz.

- [Microsoft LLMLingua kodu ve kullanım örnekleri](https://github.com/microsoft/LLMLingua): model ailesi/algoritma ayrımı ve gerçek entegrasyon profili için referans.
- [LongLLMLingua, ACL 2024](https://aclanthology.org/2024.acl-long.91/): bağlam filtresiyle tam token sıkıştırmasını ayrı değerlendirmek için yöntem kaynağı.
- [LLMLingua-2 makalesi](https://arxiv.org/abs/2403.12968): token classification tabanlı kompresörü ayrı taban çizgisi olarak ele alma.
- [LongBench v2, ACL 2025](https://aclanthology.org/2025.acl-long.183/): çok alanlı uzun bağlam değerlendirmesi; mevcut kısa alt kümeyi tüm benchmark diye genellememek için referans. BibTeX'te yalnız preprint kaydı yerine yayımlanmış kayıt eklenebilir.
- [QASPER, NAACL 2021](https://aclanthology.org/2021.naacl-main.365/): soru–cevap ve kanıt anotasyonunu ayrı değerlendirme.
- [MiniLM resmi model kartı](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): embedding girdi kesilmesi kontrolü.
- [Lost in the Middle, TACL 2024](https://aclanthology.org/2024.tacl-1.9/): sıralamayı seçici etkisinden ayıran deney gereksinimi.

Fiyatlar bu raporda güncel ürün tavsiyesi olarak yeniden kullanılmadı. README'deki dolar hesabı tarihsel fiyat varsayımı; güncel tasarruf garantisi değil. Yeni raporda sağlayıcı fiyatı tarihli konfigürasyon olarak saklanmalı.

## 9. Yeniden çalıştırma

Repo kökünden:

```powershell
python -m pytest -q -p no:cacheprovider
python -m coverage run --data-file=.test-tmp/audit.coverage -m pytest -q -p no:cacheprovider
python -m coverage report --data-file=.test-tmp/audit.coverage
python docs/audit/reanalyze_pilot.py
```

Son komut model yüklemez, ağa çıkmaz; mevcut ham pilotu kontrol eder ve kaynak SHA-256, tüm vaka doğruluğu, eşleştirilmiş farklar, bootstrap aralıkları, McNemar p-değerleri ve alan dağılımını JSON olarak verir. Rastgele seed sabittir. Güven aralıkları bu pilotun sınırlamalarını ortadan kaldırmaz.

**Son karar:** TokenPack'i ileri taşıyacak ilk yatırım daha fazla heuristic değil; güvenilir dosya/bütçe sınırları, doğru metrikler ve aynı sistemin ürün ile deneyde çalışmasını sağlayan ortak konfigürasyondur. Bu temel kurulduğunda mevcut seçim-önce hipotezi daha güçlü ve savunulabilir biçimde sınanabilir.
