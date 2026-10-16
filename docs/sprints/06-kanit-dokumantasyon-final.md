# Sprint 06 — Kanıt zinciri, dokümantasyon ve bütünleşik doğrulama

Önce ortak sözleşmeyi ve Sprint 01–05 sonuçlarını oku. Durum tablosunu kanıt sayma; kritik dosya ve testleri güncel çalışma ağacında kontrol et. Bu sprint uygulama serisini kapatır, yayın veya ücretli deney başlatmaz.

## Hedef

README, makale, CLI, agent rehberi ve deney raporları aynı davranış ve kanıtı anlatsın. Düzeltmelerin birlikte çalıştığını doğrula; yeni sonuç bekleyen araştırma işlerini görünür bırak.

## Uygula

### A. İddia–kanıt kaydı

- Sayısal ve davranışsal headline'ları envanterle: strict budget, workspace sınırı, coverage, accuracy lift, latency speedup, saving/cost, evidence retention ve exact/timeout.
- Her iddia için kaynak run/artifact hash, metric sürümü, denominator, profile, örnek kapsamı ve sınırlama kaydı oluştur.
- Desteksiz iddiayı yeni sayı uydurarak düzeltme; daralt, tarihsel/koşullu etiketle veya ana anlatımdan çıkar.
- %15,6 göreli artış yanında 37/83–32/83 ve mutlak farkı belirt. Aynı aggregate accuracy'nin equivalence veya her vakada korunma olmadığını açıkla.
- Eski kısmi latency'yi yeni full end-to-end ölçümü gibi sunma. Yeni tam benchmark yoksa ilgili sayısal iddiayı genişletme.

### B. Tek kaynaktan rapor üretimi

- Mevcut report üreticisini genişleterek canonical metric-v2 sonuçtan Markdown/CSV/LaTeX tablolarını üret. Ayrı kopya sayılar elle tutulmasın.
- Boş/eksik/ölçülmemiş durumlar tüm formatlarda aynı anlamı taşısın.
- Golden küçük fixture ile ham sonuç → özet → tablo zincirini doğrula; generation modeli gerektirmesin.
- Tarihsel raw ve summary'yi koru. Güncel sunumun hangi metric sürümüne geçtiğini migration notunda belirt.
- Görseller eski iddia içeriyorsa mevcut yerel figure üretim akışıyla güncelle; araç yoksa yanıltıcı görseli ana README'den çıkarıp durumu belgele. Eski PDF'yi yeniden üretilmiş gibi gösterme.

### C. Dokümantasyon bütünlüğü

- README kurulum/quickstart/compression/reproduce komutlarını gerçek parser ve profillere karşı doğrula.
- `AGENT.md`/`AGENTS.md`, submission README, source-code manifest, Python API örnekleri ve limitations birbiriyle uyumlu olsun.
- Kullanıcıya “best combination” diye tek evrensel profil önermek yerine hangi amaca/ölçüme dayandığını açıkla.
- LaTeX kaynakta yöntem, skor/selector iddiası, DP sınırı, evidence tanımı ve latency kapsamını güncelle; paper'ın algoritma analizi ve uygulama çalışması ayrımını koru.
- Kaynakça metadata'sında doğrulanabilir yayımlanmış kayıtları kullan; doğrulanmayan DOI/yazar/tarih değişimini tahmin etme.
- CITATION.cff/CHANGELOG ve bakım dokümanlarını mevcut gerçek proje bilgileriyle tamamla. Lisans türünü değiştirme ve hukuki uygunluk sonucu üretme.
- LaTeX ortamı varsa PDF'yi ayrı yeni çıktı olarak oluşturup hata/taşma kontrolü yap. Yoksa kaynak güncellemesi ve PDF'nin eski kaldığını açık raporla; bu sınırı gizleme.

### D. Bütünleşik kullanıcı yolu ve son inceleme

- Küçük gerçek text/code/structured fixture klasöründe source → pack → Markdown → read_packed_context yolunu çalıştır.
- Yerelde uygun optional paket varsa küçük gerçek PDF/Office fixture'larıyla çıkarımı doğrula; sahte modül testiyle karıştırma.
- Workspace root, aynı dosyaya tekrar yazım, düşük bütçe, cached tekrar, config/tokenizer değişimi ve failure recovery senaryolarını çapraz kontrol et.
- Temiz wheel smoke, ilgili genel testler, metric rapor üretimi, manifest/registry bütünlüğü ve doküman komutları çalışsın.
- Önceki sprintlerin birleşiminden doğan kusuru bu sprintte düzelt ve regresyon ekle; ilgisiz yeni özellik ekleme.
- Büyük raw dosyaları Git'ten silmeden artifact düzenleme planını ve checksum'lı erişim yolunu belgele. Hayali harici storage URL'si oluşturma.

## Kabul kriterleri

1. Kamuya dönük her ana sayısal iddianın kapsamı ve kaynak kanıtı izlenebilir; desteksiz headline kalmamış.
2. README/agent rehberi/CLI/profile örnekleri aynı davranışı anlatıyor.
3. Ham fixture'dan tablo üretimi deterministik; denominator/missing/metric sürümü formatlar arasında tutarlı.
4. Üretim güvenlik+bütçe sözleşmeleri ve shared pipeline eşleşmesi güncel kodda geçiyor.
5. End-to-end yerel smoke ve temiz paket smoke sonucu açık; Linux/Windows/gerçek model/LaTeX gibi dış ortam doğrulaması ayrıca belirtilmiş.
6. Makale kaynak/PDF/görsellerden hangisinin güncel olduğu açık; eski artifact sessizce yeni sonucu temsil etmiyor.
7. Her sprintin kalan işi, dış doğrulaması ve araştırma iddiası sınırı final readiness raporunda listelenmiş.

## Son teslim

`docs/sprints/results/sprint-06.md` ve `docs/sprints/results/final-readiness.md` oluştur; `STATUS.md` güncelle.

Final readiness en az şunları içersin:

- Repo sağlığı: çözülen kritik kusurlar ve kanıtları.
- Araştırma: düzeltilen ölçümler, yeniden analiz edilen tarihsel sonuçlar ve bunların sınırları.
- Doğrulama matrisi: local unit/integration, wheel, gerçek model, Windows/Linux, GPU, insan etiketi ve PDF build için ayrı durum.
- Sonraki çalıştırmalar: ihtiyaç duyulan veri/model, planlı vaka×yöntem×model sayısı, ön kontrol komutu ve beklenen artifact yolu; ölçülmemiş maliyet tahmini uydurma.
- Yayın öncesi gerçek engeller ve engel olmayan bakım işleri.

Kodun hazır olması, yeni bilimsel üstünlüğün kanıtlanması ve yayına hazır olma durumlarını üç ayrı sonuç olarak değerlendir. Commit/push/release/Modal run yapma. Kullanıcıya kısa sonuç ve dosya bağlantılarıyla teslim et.
