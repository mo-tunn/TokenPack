# Sprint 05 — Araştırma protokolü, güçlü baseline'lar ve ölçek ölçümü

Önce ortak sözleşmeyi, Sprint 01–04 sonuçlarını ve audit bölüm 5'i oku. Doğru metric, ortak pipeline ve manifest altyapısı olmadan bu sprintin bağımlı deneylerini başlatma. Amaç yeni büyük benchmark sonucu iddia etmek değil, güvenilir araştırmanın çalışan altyapısını ve yerel doğrulamasını tamamlamaktır.

## Hedef

Skor, seçici, parçalama, sıralama ve compression katkılarını ayrı sınayabilen; veri sızıntısını ve adaletsiz bütçeyi kontrol eden bir araştırma hattı kur. Ölçek sorunlarını varsaymak yerine ölç.

## Uygula

### A. Frozen değerlendirme protokolü

- Araştırma soruları, birincil metrik, ana karşılaştırmalar, bütçe rejimi, failure politikası ve model/dataset kapsamını yazılı protokole bağla.
- Daha önce incelenen pilotları geliştirme verisi say. Mevcut veri kimlikleri uygunsa belge/kaynak bazlı dev/holdout manifesti üret; değilse split üreticisi, fixture ve veri hazır olduğunda çalışacak komutu teslim et. Olmayan holdout'u oluşturulmuş sayma.
- LongBench'te alan×uzunluk dağılımı, dışlama nedenleri ve prompt'a sığma kontrolleri; QASPER'de aynı paper'ın iki split'e düşmemesi.
- Pilot discordance üzerinden paired binary güç simülasyonu aracı oluştur. Hedef pratik fark/alpha/power parametre olsun; yeterli örnek bulunmazsa iddia sınırını raporla.
- Frozen split/config hash değiştiğinde bunun yeni protokol sürümü olduğunu işaretle. Holdout tuning döngüsüne girmesin.

### B. Kontrollü baseline ve ablation runner

- Önce mevcut kodu kullanarak dense budget-fill, BM25 budget-fill, dense+BM25 fusion, MMR, value/density greedy ve DP baseline'larını aynı interface'e bağla; eksik basit baseline'ı uygula.
- Cross-encoder ve compression profillerini kurulu backend varsa çalışabilir, yoksa açık missing-dependency/model sonucu veren isteğe bağlı dallar yap. Dummy çıktıyı gerçek baseline sonucu olarak kaydetme.
- Seçici ablation'ında tüm yöntemlere birebir aynı candidate ID listesi, ağırlıklar ve skorlar ver. Knapsack'ın birleşik havuz nedeniyle iki kat aday görmesini kontrollü deneyde engelle.
- Ordering ablation'ında seçili küme aynı kalsın. Chunking/scoring ablation'ında hangi bileşenin değiştiği resolved config diff'inden görülsün.
- Compression karşılaştırmasında aynı rate ile eşit gerçekleşen token bütçesini karıştırma; gerçek token kullanımına göre karşılaştırma ve tolerans raporu kur.
- Tam kombinasyon patlamasını engelleyen aşamalı runner: retrieval smoke → dev generation → frozen az sayıda final konfigürasyon. Dry-run vaka×yöntem×model çağrı sayısını ve kaynak gereksinimini listelesin.

### C. Skor ve embedding tanılaması

- Mevcut bileşen çıkarma ve sınırlı ağırlık duyarlılığı deneylerini dev-only olacak şekilde config ile çalıştırılabilir yap. Tuning sonucu production varsayılanını otomatik değiştirmesin.
- Embedding tokenizer'ı ve gerçek max sequence length üzerinden block/chunk truncation oranını ölç; sadece cl100k token sayısından truncation tahmin etme.
- Aykırı/alakasız chunk eklenmesinde min-max skor ve seçimin değişimini gösteren diagnostic fixture ekle.
- Türkçe/kod/uzun chunk sentetik fixture'ları sözleşme testi olarak kullanılabilir; bu fixture sonuçlarını gerçek alan başarısı sayma.
- Yeni öğrenilmiş ranker veya bağımlılık farkındalıklı optimizer geliştirmeye bu sprintte yayılma. Önce mevcut yaklaşımın nerede kaybettiğini ölç.

### D. Ölçek ve sentetik solver ölçümü

- 1k/10k/100k chunk ölçeğini destekleyen parametrik local benchmark aracı oluştur. Hepsini varsayılan çalıştırma; mevcut makine için güvenli küçük/orta smoke çalıştır ve boyutu raporla.
- Cache warm/cold, dedup, incremental edit, JSON yazılan bayt, wall time ve mümkünse peak memory ölç. Fake embedder kullanıldığında bunun storage/pipeline testi olduğunu açıkla.
- Ölçüm kanıtı varsa düşük riskli gereksiz cache serileştirme/batch iyileştirmesi yap; SQLite migrasyonu gibi büyük değişimi otomatik şart sayma.
- Sentetik knapsack'a correlated/strongly-correlated/subset-sum/adversarial aileler ve bağımsız n/capacity taraması ekle. Küçük exact oracle ve gerçek zaman/state sınırı politikasını kullan.
- SA'nın greedy başlangıç üstüne artı objective ve ek runtime'ını ayrı raporla.

### E. İnsan değerlendirmesi paketi

- Kör yöntem etiketleri, kaynak görünürlüğü, correctness/faithfulness/citation rubriği, iki değerlendirici alanı ve adjudication formatı hazırla.
- Cascade'in kazanım/kayıp örneklerini ve farklı hata tiplerini seçen deterministik örnekleyici kur.
- Gerçek insan etiketi yoksa kappa/uyum/insan kalite puanı üretme; boş şablon ve yapılacak iş olarak tut.

## Kabul kriterleri

1. Split leakage, duplicate case ve frozen config değişimini yakalayan testler var.
2. Selector runner aynı candidate listesiyle çalışıyor; ordering runner aynı seçili kümeyi koruyor.
3. BM25/dense/fusion gibi lightweight yollar fixture üzerinde gerçek implementasyonla çalışıyor; unavailable model dalları uydurma sonuç üretmiyor.
4. Dry-run çağrı sayısını, model/data eksiklerini ve uygulanacak profilleri gösteriyor; ücretli servis başlatmıyor.
5. Küçük scale/solver smoke'u ölçüm artifact'ı üretiyor, fake/real ayrımı ve manifesti mevcut.
6. Protokol ve güç aracı tekrar çalışabilir; sonuçlar yeni benchmark üstünlüğü gibi sunulmuyor.
7. İnsan değerlendirme paketi hazır, insan tarafından doldurulmuş gibi raporlanmıyor.
8. Önceki sprintlerin kritik regresyonları geçiyor.

## Teslim

Araştırma protokolü, baseline/ablation/split/power/scale araçları, fixtures, küçük smoke çıktıları ve büyük deney için ön kontrolü yapılmış komutlar. `docs/sprints/results/sprint-05.md` içinde **altyapısı hazır / yerelde çalıştı / dış compute bekliyor / insan değerlendirmesi bekliyor** ayrımını yap. `STATUS.md` güncelle; büyük koşuları başlatma.
