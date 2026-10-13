# Faz 32 — WikiHop Cümle Atomları Pilot Raporu

- Pilot: 48 yeni ve önceki fazlarla çakışmayan soru
- Değişiklik: WikiHop destek pasajlarını kaynak makale kimliğini koruyarak cümle atomlarına bölme
- Reader: Faz 31 ile aynı
- Token bütçesi: aynı %25 kuralı

## Sonuç

| Kol | EM |
|---|---:|
| `atomic-sentence-rank` | 9/48 (%18,75) |
| `semantic-joint-citations` | 9/48 (%18,75) |

Eşleştirilmiş fark: 2 kazanım, 2 kayıp, net **0**; exact McNemar p = **1,0**. Ön kayıtlı pilot kapısı net +3 istediği için karar **`PILOT_STOP_NO_VALIDATION`** oldu.

- Kaynak kümesi değişen soru: 14/48
- Geri dönüş: 20/48
- Token-cap ihlali: 0
- Reader yanıtları: 96; 62 gerçek çağrı, 34 yeniden kullanım
- Gold reader freeze sonrasına kadar açılmadı ve uzak modele gönderilmedi

Cümle atomizasyonu yapısal sorunu azalttı, fakat selector EM'de pozitif net kazanım üretmedi. Ayrılmış 96 soruluk doğrulama çalıştırılmadı.
