# Sprint 01 — Güvenli dosya işlemleri ve gerçek token bütçesi

TokenPack reposunda bu sprinti uygula. Önce `docs/sprints/00-calisma-sozlesmesi.md` dosyasını oku ve uygula. Hedef model 5.6 Sol; görevi analiz, uygulama, test ve devir kaydı dahil uçtan uca tamamla. Bu ilk sprintin önceki sprint bağımlılığı yoktur.

## Hedef

Ürünün iki temel sözünü kodla güvenceye al: MCP izin verilen dosya alanı içinde kalmalı ve bütçe garantisi gerçekten yazılan context artifact'ına uygulanmalı. Aynı değişiklik grubunda çıktı bütünlüğü ve overwrite davranışını da sağlamlaştır.

## İncele

`src/tokenpack/mcp_server.py`, `packing.py`, `export.py`, `tokenization.py`, `loaders.py`, `index.py`, `pipeline.py`, `cli.py`, ilgili testler ve audit R1/R2/R5.

## Uygula

### A. Merkezi yol ve workspace politikası

- `source='.'` ile workspace kökü paketlenince varsayılan çıktı üst klasöre çıkıyor. Bunu sentetik geçici dosyalarla yeniden üret ve düzelt.
- Kullanıcıdan gelen yol kadar hesaplanan nihai yol da denetlensin: çıktı, index, selection, cache, manifest ve geçici dosyalar.
- Workspace kökü için güvenli, içeride kalan varsayılan çıktı seç. Bunun sonraki klasör taramasında tekrar kaynak olarak alınmamasını sağla.
- İçerideki symlink/reparse point üzerinden dış kaynak okuma veya dış artifact yazma durumlarını kapsa. Dosya adayı çözümlemesini dosya okuma sınırına yakın uygula. Platformun sağlayamadığı yarış güvenliği için mutlak iddia üretme; gerçek garantiyi belgele.
- Kaynak–çıktı ve artifact–artifact alias/çakışmalarını pahalı işten önce reddet. Klasör içi meşru çıktıların tekrar ingest edilmesini engelleyen açık politika kur.
- `allow_any_path` davranışını açık ve merkezi tut; genel CLI'ya istemeden MCP workspace sınırlaması dayatma.

### B. Çıktı bütünlüğü ve overwrite

- Seçim JSON'u ve Markdown yarım dosya bırakmayacak şekilde yazılsın.
- `overwrite=False` için `exists()` kontrolü tek başına yeterli değildir: aynı hedefe iki process koştuğunda yalnız biri yazabilsin. Atomik replace'in tek başına no-clobber garantisi vermediğini dikkate al.
- Kısmi hata durumunda hangi artifact'ın geçerli olduğunu anlaşılır kaydet. Kaynak dosya veya önceki başarılı çıktı zarar görmesin.
- Mevcut cache/index/manifest atomik yazımını bozma; tüm sistemi gereksiz büyük transaction altyapısına dönüştürme.

### C. Nihai token muhasebesi

- `budget`, output reserve ve context artifact sınırını tek sözleşmede tanımla. Tam model prompt'u verilmediği sürece sistem/chat-template tokenları için garanti verme.
- `selected_content_tokens` ile `rendered_context_tokens` ayrımını yap; eski alanları gerekiyorsa belgelenmiş uyumlulukla koru.
- Sorgu, kaynak işareti, ayırıcı ve başlıklar dahil son render'ı say. Sınır aşımında son chunk'ları çıkarıp yeniden render et veya gerekçeli deterministik yeniden seçim uygula. Sadece metadata sayısını değiştirerek aşımı gizleme.
- Header'ın kendisi sığmıyorsa açık hata veya belgeli minimal çıktı politikası kullan. Sessizce anlamlı içeriği kırpma.
- TokenCounter'a tokenizer kimliği/gerçek-tahmini sayım ayrımını ekle. Strict garantide regex fallback kullanma; tahmini modun sınırını kullanıcıya ve metadata'ya bildir.
- Hedef tokenizer adaptörünü en küçük gerekli arayüzle kur; sonradan Sprint 02/03 aynı arayüzü kullanabilsin.
- Bütçe/reserve/ratio gibi parametrelerde negatif, NaN/inf ve imkânsız kombinasyonları erken doğrula.

## Kabul kriterleri

1. MCP kök ve alt klasör paketlemesi varsayılanda workspace dışına dosya yazmıyor.
2. Dış yolu gösteren dosya ve çıktı linkleri desteklenen test platformunda reddediliyor; desteklenmeyen platform kontrolü açık skip.
3. Kaynak/çıktı çakışması kaynağa dokunmadan hata veriyor.
4. Aynı çıktıya iki bağımsız process ile `overwrite=False` çalıştırıldığında tek başarılı yazım var; diğeri anlaşılır hata alıyor.
5. Yarım yazım senaryosunda mevcut başarılı çıktı korunuyor.
6. Clean/debug/none, kısa/uzun sorgu, Unicode/Türkçe/kod, küçük bütçe ve kompresör test double'ı için render edilmiş artifact tanımlı token sınırını aşmıyor.
7. Sayaç fallback'i strict modda görünmez davranış değişikliği üretmiyor.
8. Mevcut ilgili testler ve genel suite geçiyor; README/API bütçe açıklaması yeni sözleşmeyle uyumlu.

## Kapsam ve teslim

Bu sprintte skor algoritmasını veya makale sonuçlarını değiştirme; compression backend profil düzeltmesi Sprint 02'de. Ancak mevcut compression çıktısı da son render bütçe denetiminden geçsin. Gerçek modele gerek olmayan kritik sınırları fake embedder ile test et; bunu model entegrasyonu diye sunma.

Kod, regresyonlar, gereken kısa kullanım güncellemesi ve `docs/sprints/results/sprint-01.md` teslim et. Sonuç kaydında Sprint 02 için tokenizer/path/write API'larını açıkla; `STATUS.md` güncelle ve dur.
