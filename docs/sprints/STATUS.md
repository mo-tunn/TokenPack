# Sprint durumları

Başlangıç: 16 Eylül 2026. Bu tablo prompt paketinin oluşturulduğu anda başlatılmıştır; uygulama yapılmış değildir.

| Sprint | Uygulama | Kabul/entegrasyon doğrulaması | Sonuç kaydı |
|---|---|---|---|
| 01 — Güvenli I/O ve token bütçesi | Bekliyor | Çalıştırılmadı | Henüz yok |
| 02 — Pipeline, profiller, cache | Bekliyor | Çalıştırılmadı | Henüz yok |
| 03 — Metrikler ve istatistik | Bekliyor | Çalıştırılmadı | Henüz yok |
| 04 — Tekrarlanabilirlik ve CI | Bekliyor | Çalıştırılmadı | Henüz yok |
| 05 — Araştırma ve ölçek | Bekliyor | Çalıştırılmadı | Henüz yok |
| 06 — Kanıt ve bütünleşik doğrulama | Bekliyor | Çalıştırılmadı | Henüz yok |

## Güncelleme ilkesi

Her sprint yalnız kendi satırını ve gerekiyorsa önceki sprintte düzelttiği somut kusurun kaydını günceller. “Tamamlandı” yalnız ilgili kabul kriterlerinin kanıtına dayanır. Çalıştırılmamış gerçek model/platform/insan değerlendirmeleri açıkça ayrı tutulur. Dış doğrulama bekleyen bir işin sonraki sprintin hangi bölümlerini etkilediği sonuç kaydında açıklanır.


## 23 Eylül 2026 — Ölçüm güvenilirliği paketi

Kullanıcının yeniden önceliklendirdiği ilk paket tamamlandı: [sonuç ve devir](../research/measurement-v2-handoff.md). Bu teslim altı sprintin bütünü değildir; yukarıdaki eski sprint satırları topluca tamamlandı olarak değiştirilmedi.

- Sprint 03: LongBench paydaları/istatistikleri, QASPER metrik ayrımı, aşama zaman kaydı ve ayrı v2 yeniden analiz uygulandı. Knapsack skip/timeout ve exact/approx işleri bu paketin dışında.
- Sprint 05: ölçüm protokolü yazıldı. Frozen bağımsız split, güç analizi, yeni baseline runner'ları, insan değerlendirmesi ve ölçek deneyleri henüz yapılmadı.
- Son genel doğrulama: 134 test geçti; gerçek model/GPU koşusu yapılmadı. Tarihsel 361 sonuç dosyası hash doğrulamasıyla korundu.
