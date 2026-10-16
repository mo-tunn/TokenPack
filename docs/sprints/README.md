# TokenPack — Codex / 5.6 Sol sprint promptları

Bu paket, `2026-09-16` tarihli repo ve araştırma incelemesini uygulamaya dönüştürür. Sprintler insan iş günlerine göre küçük ticket'lar değil, Codex'te tek bir kapsamlı çalışma görevi olarak yürütülecek sonuç kümeleridir. Hedef model **5.6 Sol**; özel veya doğrulanmamış modele özgü komut sözdizimi kullanılmaz.

## Kullanım

1. Codex'te TokenPack projesini aç ve model olarak 5.6 Sol'u seç.
2. Aşağıdaki başlatma mesajlarından yalnızca sıradakini gönder. Alternatif olarak ilgili sprint Markdown dosyasının tamamını yapıştır.
3. Sprintin sonuç raporunu ve `STATUS.md` içindeki kabul durumunu kontrol et.
4. Önceki sprintin gerekli kabul kapıları geçtikten sonra sonraki mesajı ver. Aynı task içinde ilerleyebilir veya aynı çalışma ağacını kullanan yeni bir task açabilirsin.
5. Yeni task farklı worktree kullanıyorsa önceki sprint değişikliklerinin orada bulunduğunu doğrula; dosyada “tamamlandı” yazması kodun taşındığı anlamına gelmez.

Her prompt ortak çalışma sözleşmesini okutur; önceki konuşmanın hatırlanmasını gerektirmez. Sprintler tamamlandığında agent'ın otomatik olarak sonrakine geçmesi istenmez. Her sprint kendi kapsamını bütünüyle bitirir.

## Sıra ve kapsam

| Sprint | Sonuç | Başlıca bağımlılık |
|---|---|---|
| [01 — Güvenli I/O ve gerçek bütçe](01-guvenli-io-ve-token-butcesi.md) | Workspace sınırı, overwrite/atomik yazım, son çıktı token garantisi | Yok |
| [02 — Ortak pipeline ve profiller](02-ortak-pipeline-profiller-cache.md) | Compression profilleri, ürün/deney eşleşmesi, cache kimliği | 01 |
| [03 — Ölçüm ve istatistik doğruluğu](03-metrikler-gecikme-istatistik.md) | Doğru denominator, süreler, evidence metrikleri, DP semantiği | 01–02 |
| [04 — Tekrarlanabilir deney ve CI](04-tekrarlanabilirlik-paketleme-ci.md) | Manifest, ortam profilleri, temiz wheel ve CI güvence katmanları | 01–03 |
| [05 — Araştırma ve ölçek altyapısı](05-arastirma-protokolu-ve-olcek.md) | Frozen değerlendirme, güçlü baseline'lar, ablation ve ölçek araçları | 01–04 |
| [06 — Kanıt, dokümantasyon ve son doğrulama](06-kanit-dokumantasyon-final.md) | Claim registry, artifact haritası, tutarlı README/makale, readiness raporu | 01–05 |

## Sırayla gönderilecek kısa mesajlar

### Sprint 01

```text
docs/sprints/01-guvenli-io-ve-token-butcesi.md dosyasını görev talimatı olarak oku ve uygula. Önce docs/sprints/00-calisma-sozlesmesi.md dosyasını oku. Yalnız plan hazırlama: sprintin uygulamasını, regresyon testlerini ve sonuç raporunu tamamla. Bu sprint bittiğinde dur; sonraki sprinte geçme.
```

### Sprint 02

```text
docs/sprints/02-ortak-pipeline-profiller-cache.md dosyasını görev talimatı olarak oku ve uygula. Önce ortak sözleşmeyi ve Sprint 01 sonuçlarını oku, mevcut kodla doğrula. Uygulamayı ve testleri tamamla; Sprint 02 sonuç raporunu yaz. Sonraki sprinte geçme.
```

### Sprint 03

```text
docs/sprints/03-metrikler-gecikme-istatistik.md dosyasını görev talimatı olarak oku ve uygula. Önce ortak sözleşmeyi ve önceki sprint sonuçlarını oku. Ölçüm düzeltmelerini uygula, tarihsel ham veriyi koruyarak yeniden analiz et, testleri ve Sprint 03 raporunu tamamla. Sonraki sprinte geçme.
```

### Sprint 04

```text
docs/sprints/04-tekrarlanabilirlik-paketleme-ci.md dosyasını görev talimatı olarak oku ve uygula. Önce ortak sözleşmeyi ve önceki sprint sonuçlarını oku. Tekrarlanabilirlik, paketleme ve CI değişikliklerini uygula, yerelde mümkün olan kontrolleri çalıştır ve Sprint 04 raporunu tamamla. Sonraki sprinte geçme.
```

### Sprint 05

```text
docs/sprints/05-arastirma-protokolu-ve-olcek.md dosyasını görev talimatı olarak oku ve uygula. Önce ortak sözleşmeyi ve önceki sprint sonuçlarını oku. Araştırma protokolünü çalışan baseline/ablation/ölçek araçlarıyla birlikte tamamla. Ücretli veya büyük model deneyleri başlatma; bunları ön kontrolü yapılmış çalıştırma komutları olarak hazırla. Sonuç raporunu yaz ve sonraki sprinte geçme.
```

### Sprint 06

```text
docs/sprints/06-kanit-dokumantasyon-final.md dosyasını görev talimatı olarak oku ve uygula. Önce ortak sözleşmeyi ve tüm sprint sonuçlarını oku. Kanıt–iddia eşleşmesini, dokümantasyonu ve bütünleşik doğrulamayı tamamla. Gerçekten doğrulanmış sonuçları, hazırlanmış ama çalıştırılmamış deneylerden ayır. Nihai readiness raporunu yaz; yayınlama veya ücretli deney başlatma.
```

## Beklenen kayıtlar

- [Ortak çalışma sözleşmesi](00-calisma-sozlesmesi.md)
- [Sprint durum tablosu](STATUS.md)
- `results/sprint-01.md` … `results/sprint-06.md`: ilgili sprint çalıştırıldığında oluşturulacak sonuç kayıtları.
- `results/final-readiness.md`: son sprintte hazırlanacak bütünleşik değerlendirme.

**Önemli:** Bu dosyaların hazırlanması sprintlerin uygulanmış olduğu anlamına gelmez. Başlangıçta tüm sprintler bekliyor. Yeni benchmark üstünlüğü, insan değerlendirmesi veya platform başarısı çalıştırılmadan tamamlandı sayılmaz.
