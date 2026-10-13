# TokenPack Araştırma Kapsam Kararı

**Tarih:** 13 Ekim 2026  
**Karar durumu:** WikiHop selector hattı kapatıldı; Faz 28 HotpotQA sonucu ana doğrulanmış bulgu olarak korunuyor.

## Ana doğrulanmış sonuç: Faz 28 HotpotQA

Önceden ayrılmış 96 soruluk public validation üzerinde aynı okuyucu ve token bütçesiyle:

| Ölçüm | Kontrol | `semantic-joint-citations` | Net | Exact McNemar p |
|---|---:|---:|---:|---:|
| Exact Match | 48/96 (%50,00) | **61/96 (%63,54)** | **+13** | **0,000244** |
| Tam anotasyonlu destek kapsaması | 47/96 (%48,96) | **73/96 (%76,04)** | **+26** | **2,98×10⁻⁸** |

Önceden belirlenmiş Bonferroni düzeltilmiş eşik her iki birincil ölçütte de sağlandı. EM bootstrap %95 aralığı net fark için yaklaşık **[+7, +20]** soru; tam destek kapsaması için **[+18, +35]** soru oldu. Bu sonuç public validation setine aittir; QASPER resmi gizli testi kullanılmadı.

## Genelleme denemeleri

| Faz | Benchmark / değişiklik | Kontrol | Semantic | Net | p | Karar |
|---|---|---:|---:|---:|---:|---|
| 29 | MuSiQue frozen replication | 15/48 | 17/48 | +2 | 0,50 | `STOP_NO_VALIDATION` |
| 31 | WikiHop EM-only | 8/48 | 7/48 | -1 | 1,0 | durdu |
| 32 | WikiHop cümle atomları | 9/48 | 9/48 | 0 | 1,0 | durdu |
| 33 | WikiHop aday cevaplı reader | 19/48 | 20/48 | +1 | 1,0 | durdu |
| 34 | Aday cevaplı verifier + reader | 22/48 | 24/48 | +2 | 0,5 | durdu |
| 35 | Citation ID canonicalization | 21/48 | 22/48 | +1 | 1,0 | durdu |

Faz 31–35 için pilot kapısı net **+3 EM** idi. Hiçbir faz bu kapıyı geçmedi; ayrılmış 96 soruluk doğrulama setlerine geçilmedi. WikiHop'ta resmi cümle düzeyi gold destek indeksleri bulunmadığı için bu fazlarda yalnız EM raporlandı.

## Mekanizma yorumu

Faz 30A, selector'ın destek kapsamını iki arşivde artırdığını, fakat bunun EM'e dönüşmesinin heterojen olduğunu gösterdi. Faz 30B ve 30C okuyucu istemi değişiklikleri kazanımı artırmadı. Faz 31–35, farklı WikiHop yapısal ve okuyucu düzeltmelerinin semantic selector için tekrarlanabilir bir EM üstünlüğü üretmediğini gösterdi.

Bu nedenle genel iddia şu kapsamla sınırlandırıldı:

> `semantic-joint-citations`, TokenPack'in HotpotQA public validation deneyinde cevap doğruluğu ve anotasyonlu destek kapsamını artırdı. WikiHop ve MuSiQue sonuçları bu kazanımın veri kümeleri arasında otomatik olarak genellenmediğini gösteriyor.

## Kapanış ve arşiv durumu

- WikiHop selector hattı Faz 35 itibarıyla **kapatıldı**.
- Faz 31–35 arşivleri korunuyor; sonuçlar yeniden skorlanmayacak.
- Faz 28 HotpotQA arşivi ve doğrulama kayıtları korunuyor.
- QASPER resmi gizli testi ve üretim kodu değiştirilmedi.
- Yeni deney ancak farklı bir araştırma hipotezi ve yeni ön kayıtla başlatılabilir.
