# Faz 35 — WikiHop Citation ID Canonicalization Pilot

**Tarih:** 13 Ekim 2026  
**Benchmark:** QAngaroo WikiHop v1.1 resmi `dev` bölümü  
**Pilot:** 48 yeni soru; Faz 31–34 kimlikleri dışlandı

## Hipotez

Faz 34'te aday cevapları verifier'a vermek net +2 EM sağladı, ancak verifier çıktılarının bir bölümü `S00` yerine `0` biçimindeydi. Faz 35'te yalnızca bu mekanik arayüz sorunu önceden kaydedildi: geçerli tamsayı veya rakam dizisi `k`, whitelist ve kaynak çeşitliliği kontrollerinden önce `S{k:02d}` biçimine çevrildi. Out-of-range ve diğer biçimler reddedilmeye devam etti. Faz 34 yeniden skorlanmadı.

## Sonuç

| Kol | EM | Ortalama context tokenı |
|---|---:|---:|
| `atomic-sentence-rank` | **21/48 (%43,75)** | 384,54 |
| `semantic-joint-citations` | **22/48 (%45,83)** | 385,23 |

Eşleştirilmiş sonuç:

- Semantic kazanım: **2**
- Semantic kaybı: **1**
- Net: **+1**
- İki yönlü exact McNemar p: **1,0**
- Pilot eşiği: **en az +3 net EM**
- Karar: **`PILOT_STOP_NO_VALIDATION`**

Selector kaynak kümesi 13/48 soruda değişti. Token-cap ihlali olmadı.

## Canonicalization etkisi

- Verifier çıktılarının 13'ünde numeric citation ID görüldü.
- 23/48 çıktı geçerli cross-article citation olarak işlendi.
- 23/48 çıktı çekimser kaldı.
- 1 çıktı tek kaynak makaleye düştüğü için reddedildi.
- 1 çıktı JSON biçim hatasıydı.
- Semantic fallback sayısı 26/48 oldu.

Canonicalization, Faz 34'teki numeric-ID biçim hatası sınıfını teknik olarak çözdü; ancak geçerli citation oranındaki artış EM'de ön kayıtlı eşiği karşılamadı. Kalan ana sınırlama çekimserlik ve token-cap uyumsuzluğu.

## Eşleştirilmiş farklar

- `WH_dev_3364`: kontrol doğru (`robert mckimson`), semantic `unknown` verdi; kayıp.
- `WH_dev_3447`: kontrol `chinese`, semantic `cantonese` verdi; semantic kazandı.
- `WH_dev_4979`: kontrol `unknown`, semantic `ithaca` verdi; semantic kazandı.

## Deney bütünlüğü

- Verifier: 48 başarılı ücretli çağrı.
- Reader: 96 yanıt; 61 gerçek çağrı, 35 yeniden kullanım.
- Gold reader freeze sonrasında açıldı.
- Gold cevapları uzak modele gönderilmedi.
- İlk iki verifier timeout denemesi `verifier_error_attempt1/2.jsonl` ve `verifier_inflight_attempt1/2.json` olarak korundu; üçüncü kayıtlı denemede 48/48 tamamlandı.
- Ayrılmış 96 soruluk doğrulama seti kullanılmadı.
- Faz 34 sonucu yeniden hesaplanmadı.

## Karar

Citation ID mekanik onarımı gerekli bir teknik düzeltmeydi, fakat WikiHop'ta semantic selector üstünlüğünü doğrulamadı. Faz 31–35 boyunca hiçbir WikiHop pilotu ön kayıtlı EM kapısını geçmedi. Bu selector hattında yeni prompt varyasyonları denemek yerine WikiHop çalışması kapatılmalı; Faz 28 HotpotQA sonucu kendi benchmark sınırları içinde raporlanmalı.
