# Ölçüm güvenilirliği paketi — sonuç ve devir

## Durum

İlk çalışma paketi uygulandı ve yerel doğrulaması tamamlandı. Yeni model/GPU/Modal/API deneyi, insan değerlendirmesi, hız optimizasyonu veya bağımsız test kümesi değerlendirmesi yapılmadı. Bunlar bu paketin tamamlanma koşulu değildir.

Başlangıç HEAD: `0061578c3f650836f6ca3cb98665bd08fee15d2d`. Başlangıçta beş Word dosyası ve `docs/` izlenmiyordu. Kullanıcı Word dosyaları korunmuştur. Başlangıç çalışma ağacı ve 361 tarihsel sonuç dosyasının hash'leri `measurement-baseline.json` içinde kayıtlıdır.

## Teslim edilen davranışlar

- Ortak, model bağımsız evaluation katmanı; şema v2 / metrik 2.0; task-manifest paydası ve eksik veri politikası.
- LongBench özet/ikili karşılaştırma, yerel istemci, Modal raporlama ve hata analizi tüketicilerinde açık metrik adları. Bilinmeyen judge sonuçları null; değerlendirme kapsamı ayrıca raporlu.
- QASPER'ın tarihsel lexical proxy adları düzeltildi. Alternatif anotasyonlar, verbatim kanıt, complete-set ve güvenilir normalize-source aralıkları ayrı tutuluyor. Compression runner gelecekte metinli JSONL ve konfigürasyon manifesti yazıyor.
- Sürüm ve hash içeren canonical offline yeniden analiz; QASPER karşılaştırmalarında 77 belge üzerinden cluster bootstrap.
- Aşama süreleri, render edilmiş context tokenları, prompt tokenizer bilgisi ve bilinmeyen alanlar için kayıt desteği. Uçtan uca istek süresiyle aşama toplamı ayrıldı.
- README, LaTeX iddiaları ve dört ilgili LaTeX tablosu düzeltildi. Eski headline/sonuç görselleri README'den, eski birleşik sonuç görseli makale kaynağından çıkarıldı; dosyaları silinmedi. Eski PDF yeniden derlenmedi ve README'de tarihsel olarak etiketlendi.

## Kabul matrisi

| Kabul alanı | Kanıt | Sonuç |
|---|---|---|
| Tüm planlanan vakalar üzerinden doğruluk | Failure/parse/missing/manifest testleri | Geçti |
| Duplicate, manifest dışı kayıt, farklı vaka kümeleri | Açık hata regresyon testleri | Geçti |
| Unknown judge hallucination değildir | Judge failure fixture'ı ve parser sözleşmesi | Geçti |
| 83 vaka sayıları ve eşleştirilmiş sonuçlar | Canonical pilot regresyon testi + JSON/CSV | Geçti |
| Lexical/verbatim/span ayrımı | Sıra, sayı, olumsuzluk, alternatif anotasyon, segment sınırı testleri | Geçti |
| Gelecek QASPER kayıtları | Tiny fixture ile gerçek yerel runner; fake embedder | Geçti; model kalitesi ölçümü değildir |
| Cluster/bootstrap ve deterministik analiz | Sabit seed testleri; CLI iki tekrar | Geçti |
| Eksik süre/çift sayım | Fake-clock ve included-in-compression testleri | Geçti |
| Tarihsel artifact korunması | Başlangıçtaki 361 hash'in tümü karşılaştırıldı | Değişiklik yok |
| README/tablo/rapor uyumu | Dört tablo canonical üreticiden; migration/readout incelemesi | Geçti |
| GPU/Modal gerçek entegrasyonu | Çalıştırılmadı | Sonraki deney öncesi doğrulanmalı |
| PDF görsel doğrulaması | PDF yeniden üretilmedi | Tarihsel PDF aktif v2 çıktı sayılmıyor |

## Doğrulama

Ortam: Windows, Python 3.13.1; yeni bağımlılık kurulmadı.

```powershell
python -m pytest -q -p no:cacheprovider
python -m compileall -q submission/evaluation submission/longbench_eval submission/experiments
python -m submission.evaluation.reanalyze --update-paper-tables
python -m submission.evaluation.reanalyze
git diff --check
```

Başlangıç: 120 test geçti. Son genel kontrol: **134 test geçti**, önceden mevcut 368 pathspec deprecation uyarısı devam ediyor. Ara kontrolde bir eski cache `os.replace` testi Windows erişim hatası verdi; aynı testin hedefli tekrarı ve sonraki genel takım geçti. Bu pakette cache altyapısı değiştirilmedi.

İki ardışık CLI koşusunda sekiz sayısal/istatistik/tablo çıktısının byte hash'leri aynı bulundu. Manifestin commit/dirty provenance alanları çalışma ağacına göre değişebilir; sayısal sonuçların deterministikliği bundan ayrıdır. Tarihsel 361 sonuç dosyası değişmedi. Yerel fixture ve compile kontrolleri uzak GPU entegrasyonu başarı iddiası değildir.

## Sonuçların yorumu

Full context 32/83, dense RAG 34/83, TokenPack 37/83, yalnız LongLLMLingua 30/83, cascade 37/83. TP–full farkı +6.02 yüzde puan; %95 eşleştirilmiş aralık −2.41…+15.66 pp, p=0.302. TP–dense p=0.581. Cascade, TP'ye göre sekiz kazanım ve sekiz kayıp içeriyor; eşdeğerlik gösterilmiş değildir.

QASPER lexical değerleri yeniden hesaplanan özetlerle korunuyor, fakat eski scalar CSV'ler metin içermediğinden verbatim/span ölçümleri üretilemiyor. Başarı veya başarısızlık olarak sıfır doldurulmadı. Kaydedilmiş 3.90× hız oranı uçtan uca ürün hızı değildir.

## Sonraki pakete devir

1. CLI/API/deney akışını ortak, sürümlenmiş pipeline profiline bağla. Mevcut üretim varsayılanları bu pakette değiştirilmedi.
2. Yeni koşuda task manifesti ve v2 kayıtlarını sakla; eski scalar çıktılar yeni alanları geri getiremez. `metric-v2-migration.md` arayüz değişikliklerinin kaynağıdır.
3. Belge bazlı dev/test ayrımı, güçlü baseline ayarları ve güç analiziyle bağımsız final protokolü oluştur. Mevcut pilotları geliştirme verisi say.
4. Gerçek küçük model/Modal smoke doğrulaması ve kör insan değerlendirmesi ayrı yapılmalı. Bu paket bunlar için sonuç uydurmaz.
5. Hız optimizasyonuna ancak ortak pipeline ve bağımsız kalite değerlendirmesi sonrasında geç.

Eski altı sprintin tümü tamamlanmış değildir. Bu çalışma Sprint 03'ün metrik/istatistik bölümlerini ve Sprint 05'in protokol tanımının bir bölümünü karşılar; knapsack analizi, güvenli I/O, paketleme, ortak pipeline ve bağımsız deneyler için ilgili eski maddeler açık kalır.
