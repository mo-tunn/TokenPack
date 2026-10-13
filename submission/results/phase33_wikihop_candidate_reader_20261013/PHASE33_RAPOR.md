# Faz 33 — WikiHop Aday Cevaplı Reader Ablation Raporu

- Pilot: 48 yeni ve önceki fazlarla çakışmayan soru
- Selector: Cümle atomları üzerinde aynı MiniLM kontrolü ve semantic-joint-citations uygulaması
- Reader değişikliği: WikiHop'un benchmark girdisindeki aday cevaplar iki kola da verildi; hedef ilişkiyi doğrudan cevaplama istendi
- Gold cevap veya annotation reader'a gönderilmedi

## Sonuç

| Kol | EM |
|---|---:|
| `atomic-sentence-rank` | 19/48 (%39,58) |
| `semantic-joint-citations` | 20/48 (%41,67) |

Eşleştirilmiş fark: 2 kazanım, 1 kayıp, net **+1**; exact McNemar p = **1,0**. Ön kayıtlı pilot kapısı net +3 istediği için karar **`PILOT_STOP_NO_VALIDATION`** oldu.

- Kaynak kümesi değişen soru: 11/48
- Geri dönüş: 25/48
- Token-cap ihlali: 0
- Reader yanıtları: 96; 59 gerçek çağrı, 37 yeniden kullanım
- Verifier'da ilk çağrı bir `TimeoutError` ile sonuçlandı; hata/inflight kayıtları `verifier_error_attempt1.jsonl` ve `verifier_inflight_attempt1.json` olarak korundu. Sonraki kayıtlı denemede 48/48 verifier yanıtı tamamlandı.
- Gold reader freeze sonrasına kadar açılmadı ve uzak modele gönderilmedi

Aday cevap listesi reader'ın toplam EM seviyesini yükseltmiş görünüyor, fakat yeni örneklem kullanıldığı için Faz 32 ile doğrudan paired artış iddiası kurulamaz. Selector kolları arasındaki ön kayıtlı net fark yine eşik altında kaldı. Ayrılmış 96 soruluk doğrulama çalıştırılmadı.

## Genel karar

Faz 31, Faz 32 ve Faz 33 birlikte şu sınırı gösteriyor: Faz 28 HotpotQA selector kazanımı WikiHop'a doğrudan taşınmıyor. WikiHop'ta cümle atomizasyonu ve aday cevaplı reader ayrı ayrı denendi; hiçbiri pilot kapısını geçmedi. Yeni bir Faz 34 ancak yeni bir hipotez ve ön kayıtlı kapıyla başlatılmalı; mevcut sonuçlarla doğrulama veya genel üstünlük iddiası yapılmamalı.
