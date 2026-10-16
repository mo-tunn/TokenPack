# Sprint 02 — Ortak pipeline, compression profilleri ve cache doğruluğu

Önce `docs/sprints/00-calisma-sozlesmesi.md`, Sprint 01 sonuçları ve audit R3/R4/R6'yı oku. Sprint 01'in path ve bütçe sözleşmesinin güncel kodda bulunduğunu doğrula. Bu sprinti uygulama ve testleriyle tamamla; sıradakine geçme.

## Hedef

CLI, Python API ve araştırma harness'i aynı konfigürasyonun aynı pipeline'ı ifade ettiği bir yapıya getir. Compression model/bayrak hatalarını kaldır; cache'in bu konfigürasyonu doğru temsil etmesini sağla.

## İncele

`packing.py`, `compression.py`, `chunk_profiles.py`, `embeddings.py`, `pipeline.py`, `export.py`, `cli.py`, `mcp_server.py`; iki değerlendirme dizininin client/app dosyaları; `task_report.json`, README ve agent rehberi.

## Uygula

### A. Sürümlü ve açık konfigürasyon

- Mevcut yapıya uygun, doğrulanabilir ve serileştirilebilir ortak pipeline config oluştur. Genel framework kurma.
- Chunker/boyutlar, scoring, selector, candidate pool, sıralama, renderer, tokenizer, output reserve ve compression ayarları görünür olsun.
- Varsayılanların çözülmüş hali saklanabilsin; bir config hash'i yorumlanan etkin ayarlardan üretilebilsin.
- Üretim ve tarihsel makale ayarlarını ayrı adlandır. Tarihsel profil yalnız kanıtlanan ayarları içersin; kayıtlarda olmayan model revision veya parametre uydurma. Eksik provenance'ı açık işaretle.
- CLI override önceliğini tanımla; bilinmeyen veya çelişen değerler erken hata versin. Eski komutlar gerekiyorsa belgeli deprecation ile taşınsın.

### B. Compression profilleri

- `selection-only`, LLMLingua-2, LongLLMLingua context-level ve LongLLMLingua token-level gibi açık modlar tanımla; kesin isimlerde mevcut CLI uyumluluğunu gözet.
- LLMLingua-2 encoder modelini nedensel LongLLMLingua backend'ine yanlışlıkla verme. Karşılıklı dışlayan flag'leri reddet.
- Model, model revision, filtreler, rate/target, device ve yeniden sıralama davranışını config'te görünür yap.
- Kullanılan kurulu LLMLingua API'sini/resmi örnekleri doğrula. Gerçek backend smoke için yereldeki cache'i kullan; büyük model indirme veya Modal işi başlatma.
- Model yoksa config/dispatch sözleşmesini bağımsız test et; gerçek backend kontrolünü çalıştırılmadı olarak kaydet.

### C. Ortak pipeline ve renderer

- Üretimle deney arasındaki farklılıkları önce envanterle: chunk boyu, sıralama, header, query oluşturma, tokenizer ve reserve.
- Ortak davranışı paylaş; araştırmaya özgü multiple-choice prompt oluşturmayı ürün katmanına taşıma.
- Aynı fixture, fake embedder ve etkin config için CLI/API/harness seçilen ID ve render hash'i eşleşsin. Sıralama ablation'ı açık değişken olarak kalsın.
- Sadece geçmiş headline'a benzemesi için üretim varsayılanını değiştirme; farklı profiller açıkça farklı sonuç üretmeli.
- Compression öncesi ve sonrası metin/sıralama/token metadata'sı karışmasın.

### D. Cache/manifest kimliği

- Cache anahtarı ve incremental manifest geçerliliğine model kimliği+revision, tokenizer kimliği+revision ve gerekli pipeline/extractor/schema sürümünü dahil et.
- Ayrı embedding cache ile chunk/token manifestinin hangi değişiklikte geçersizleşeceğini açık tanımla; alakasız ayar değişiminde tüm embedding'i gereksiz hesaplama.
- Eski cache şemasını güvenli migrate veya invalidate et; kullanıcı verisini silme.
- Aynı batch'teki tekrar eden missing metinleri bir kez embed et ve özgün sırayı koru.
- Cache hit/miss ve invalidation nedenleri ölçülebilir olsun. Atomik/eşzamanlı yazımı Sprint 01 ile uyumlu koru.

## Kabul kriterleri

1. Desteklenen profile/override kombinasyonları etkin config'e deterministik çözülüyor; çelişkiler erken hata.
2. LLMLingua-2 ve LongLLMLingua farklı backend koşullarıyla doğru dispatch ediliyor.
3. Aynı profile CLI/API/harness üzerinden aynı chunk seçimi ve bağlam hash'i üretiyor.
4. Source/score/score-then-source davranışları ayrı testlerle belgelenmiş; profile farkı gizli değil.
5. Model/tokenizer revision veya ilgili schema değişince doğru cache katmanı invalid oluyor; değişmeyen içerik tekrar kullanılıyor.
6. Eski cache, eşzamanlı yazım ve repeated text senaryoları veri bozmadan işleniyor.
7. Sprint 01 güvenlik ve bütçe regresyonları geçiyor.
8. README'nin önerdiği komut ve makale profilinin farkı açık; gerçek backend smoke'unun durumu dürüstçe raporlanmış.

## Teslim

Ortak config/pipeline kodu, profil fixture'ları, migration/uyumluluk notu, ilgili testler ve `docs/sprints/results/sprint-02.md`. Sprint 03'e etkin config ve token ölçüm API'si, hangi tarihsel ayarların bilinmediği ve gerçek backend kontrol durumu devredilsin. `STATUS.md` güncelle.
