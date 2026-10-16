# Sprint 04 — Tekrarlanabilir deneyler, paketleme ve CI

Önce ortak sözleşmeyi, Sprint 01–03 raporlarını ve audit R7/R8/R9'u oku. Önceki değişikliklerin mevcut çalışma ağacında bulunduğunu kontrol et. Bu sprintte altyapıyı gerçekten kur ve yerelde erişilebilir kontrolleri çalıştır.

## Hedef

Bir sonucun hangi kod/veri/model/konfigürasyondan üretildiği izlenebilsin; paket temiz ortamda kurulabilsin; önemli davranışlar yalnız yerel editable kurulumda çalışıyor olmasın.

## Uygula

### A. Koşu manifesti ve registry

- Sürüm kontrollü JSON şeması veya repo ölçeğine uygun eşdeğer validasyon ekle.
- Run ID, amaç/canonical-exploratory-smoke etiketi, commit ve ilgili dirty diff hash, dataset revision/split/hash, planned case IDs, model/tokenizer revision, resolved config, prompt/parser/metric sürümü, environment lock, hardware ve artifact hash'lerini kaydet.
- Kaydedilemeyen değeri unknown yap; eski çalışmaya bugünkü commit/model revision'ını mal etme.
- Aynı config/data hash'inden farklı koşular üretilebilsin; run ID ile config ID'yi karıştırma. Yeniden başlatma ve eksik shard birleştirmesi duplicate üretmesin.
- Eski sonuçlar için küçük registry oluştur: kaynak yolları, bilinen/eksik metadata, hangi tabloya temel olduğu. Büyük raw dosyaları taşıma/silme.
- Manifest içine API anahtarı, ortamın tüm değişkenleri veya kullanıcı belge içeriği dökme.

### B. Ortam ve paketleme

- Runtime bağımlılıkları ile research/dev/compression extras'ını gerçekten kullanılan importlara göre kontrol et.
- CPU doğrulama ile GPU/Modal deney ortamını ayır. Doğrulanabilir constraints/lock çözümü kullan; rastgele üst sürüm sınırları uydurma.
- İki Modal image tanımının doğrudan import gereksinimlerini, özellikle `pathspec` ve token sayım bağımlılıklarını kontrol et. Image tanımını düzelt; uzak job başlatma.
- Wheel ve sdist üret; temiz izole ortamda wheel'den import ve CLI smoke çalıştır. Testin cwd/PYTHONPATH sayesinde yanlışlıkla source tree import etmediğini kanıtla.
- Optional extra eksikliğinin anlaşılır hata verdiğini ve normal selection kullanımını gereksiz bozmadığını doğrula.

### C. CI katmanları

- Hızlı unit/contract, güvenlik+bütçe regresyonu, araştırma metric-fixture testleri ve paket smoke'u belirgin katmanlara ayır.
- Linux+Windows matrisi kur; Python sürümlerini bağımlılık desteğine göre belirle. Yerelde geçmeyen/çalışmayan platform için tamamlandı iddiası üretme.
- Gerçek model backend smoke'u ayrı manuel/gecelik yol olsun; normal PR job'ı büyük indirme veya ücretli servis başlatmasın.
- Ruff ve kademeli tip kontrolünü mevcut kodun durumuna göre uygula. İlgisiz bütün kodu formatlayıp diff'i gereksiz büyütme; seçici baseline/ kapsam belgeli olsun.
- Coverage'ı runtime ve research için ayrı görünür kıl; sabit %99 rozetini canlı kanıt veya açık tarihli ölçümle değiştir. Davranış testlerini coverage yüzdesine kurban etme.
- `pathspec` deprecation uyarılarını nested ignore davranışını koruyarak çöz.
- Uygun dependency/secret kontrolü ekle. Araç kurulumu ve job tanımı ile gerçekten çalıştırılmış tarama sonucunu ayrı raporla; dış servise doküman/secret yükleme.

### D. Bakım belgeleri

- CONTRIBUTING: ortam kurulumu, test katmanları, profile/metric değiştirme kuralları, yeni fixture standardı.
- Agent rehberinin standart `AGENTS.md` keşfini mevcut `AGENT.md` içeriğini çeliştirmeden sağla.
- Release doğrulama checklist'i ekle; hiçbir release/push işlemi yapma.
- Güvenlik bildirim yolunu mevcut gerçek iletişim bilgilerine göre belirt; hayali ekip/e-posta oluşturma.

## Kabul kriterleri

1. Fixture koşusu manifestten aynı config/case listesiyle yeniden kurulabiliyor; output hash ve metric sürümü izlenebilir.
2. Eksik revision/provenance “unknown” olarak kalıyor; historical registry varsayım üretmiyor.
3. Temiz wheel kurulumundan import+CLI smoke geçiyor, source tree'ye bağımlı değil.
4. Linux/Windows CI konfigürasyonu ve lightweight metric/contract işleri mevcut; yalnız gerçekten çalıştırılanlar başarılı diye raporlanmış.
5. No-cache/cache-hit/config-changed davranışı ve sprint 01–03 regresyonları geçiyor.
6. Model smoke ve pahalı işler kontrollü ayrı tetikleyicide; normal testler ücretli servis gerektirmiyor.
7. Artifact registry validasyonu yollar/hash'ler üzerinden çalışıyor; büyük ham dosyalar ve kullanıcı değişiklikleri korunuyor.

## Teslim

Manifest/registry kodu ve fixture'ları, ortam dosyaları, paket/CI güncellemeleri, bakım belgeleri ve `docs/sprints/results/sprint-04.md`. Sonuç raporunda her ortam için “çalıştırıldı / yalnız yapılandırıldı” ayrımı yap. Sprint 05'e run başlatma/yeniden üretme/rapor üretme komutlarını devret; `STATUS.md` güncelle.
