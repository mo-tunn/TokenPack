# Faz 31 — WikiHop EM Dış Tekrarlama Pilot Raporu

**Tarih:** 13 Ekim 2026  
**Benchmark:** QAngaroo WikiHop v1.1 resmi `dev` bölümü  
**Arşiv:** `qangaroo_v1.1.zip`  
**Çalışma türü:** Faz 28 politikasının yeni benchmark üzerinde EM-only dış tekrarlaması

## Ön kayıt ve kapsam

- Pilot: 48 çakışmayan soru.
- Ayrılmış doğrulama: 96 soru; pilot kapısı geçmediği için kullanılmadı.
- Kontrol: `atomic-sentence-rank`.
- Uygulama: `semantic-joint-citations`.
- Aynı okuyucu ve aynı token bütçesi iki kolda da kullanıldı.
- WikiHop resmi anotasyonları cümle düzeyinde gold destek indeksleri vermediği için destek kapsaması metriği raporlanmadı.
- Faz 28'in HotpotQA destek kapsaması sonucu bu benchmarkta yeniden ölçülmüş sayılmaz.

## Pilot sonucu

| Ölçüm | Kontrol | `semantic-joint-citations` |
|---|---:|---:|
| Exact Match | 8/48 (%16,67) | 7/48 (%14,58) |
| Ortalama çıktı tokenı | 433,65 | 432,44 |

Eşleştirilmiş sonuç: 0 kazanım, 1 kayıp, 47 beraberlik; net etki **-1 EM**. İki yönlü exact McNemar p değeri **1,0**. Ön kayıtlı pilot kapısı net EM en az +3 istediği için karar:

**`PILOT_STOP_NO_VALIDATION`**

## Deney bütünlüğü

- Token-cap ihlali: **0**.
- Verifier çağrıları: **48** başarılı ücretli çağrı.
- Reader kayıtları: **96** yanıt; **60** gerçek çağrı ve **36** yeniden kullanım.
- Gold etiketleri okuyucu yanıtları dondurulduktan sonra açıldı.
- Gold cevapları uzak modele gönderilmedi.
- Pilot ve ayrılmış doğrulama kimlikleri çakışmıyor.
- 48 örneğin 12'sinde seçilen kaynak kümesi değişti; 19 örnekte yöntem kontrollü geri dönüş kullandı.

Tek eşleştirilmiş kayıp `WH_dev_1228` oldu: kontrol `Hong Kong` cevabını tam eşleştirirken uygulama `Yuen Long District` üretti. Hiçbir örnekte uygulama kolu kontrolü geçemedi.

## Yorum ve sonraki aşama

Bu pilot, Faz 28 HotpotQA kazanımının WikiHop'a otomatik olarak taşındığını desteklemiyor. Aynı zamanda Faz 28 HotpotQA sonucunu geçersiz kılmıyor; iki sonuç farklı benchmark ve farklı ön kayıt kapsamlarına ait.

Faz 31 burada kapatılmalı; 96 soruluk doğrulama çalıştırılmamalı. Sonraki çalışma yeni bir okuyucu istemi denemek yerine 12 değişen örnek ve 19 geri dönüş örneği üzerinde **selector hata analizi** olmalı: aday kapsaması, kaynak kümesi değişiminin cevabı bozduğu durumlar ve WikiHop ilişki biçimlerinin Faz 28 selector varsayımlarıyla uyumu ayrıştırılmalı. Yeni bir yöntem ancak bu analizden sonra önceden kaydedilmiş yeni bir pilot kapısıyla denenmeli.
