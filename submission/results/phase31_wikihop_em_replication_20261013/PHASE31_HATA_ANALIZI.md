# Faz 31 WikiHop Hata Analizi

**Pilot:** 48 soru  
**Karar:** `PILOT_STOP_NO_VALIDATION`  
**Amaç:** Faz 28 selector politikasının WikiHop'ta neden EM kazanımı üretmediğini ayırmak.

## 1. Selector'ın fiilî etkisi

| Durum | Soru |
|---|---:|
| Verifier citation'ı geçerli, selector aktif | 29/48 |
| Citation'lar token bütçesine sığmadı, baseline'a dönüş | 11/48 |
| Verifier çekimser veya biçimsel olarak geçersiz, baseline'a dönüş | 8/48 |
| Kaynak cümle kümesi gerçekten değişti | 12/48 |
| Kaynak cümle kümesi değişmedi | 36/48 |

Dolayısıyla pilotun %75'inde iki kol okuyucuya aynı bağlamı verdi. Geçerli citation alınan 29 örneğin 17'sinde de citation tohumları baseline sıralamasını değiştirmedi.

### Geri dönüş nedenleri

- **11** örnekte 2–3 seçilmiş WikiHop destek pasajı aynı anda %25 token bütçesine sığmadı.
- **6** örnekte verifier çekimser kaldı.
- **1** örnekte JSON şeması geçersizdi.
- **1** örnekte tek citation döndü; minimum iki farklı kaynak koşulunu karşılamadı.

Bu, WikiHop desteklerini tek parça atom olarak kullanmanın yapısal bir uyumsuzluk yarattığını gösteriyor. Faz 28'deki cümle düzeyi seçim kuralı burada uzun destek pasajları yüzünden uygulanamıyor.

## 2. Cevap kanıtının bağlamda bulunması

Gold cevabın normalleştirilmiş metninin bağlamda geçip geçmediği tanısal olarak kontrol edildi. Bu, gold'un gerçekten çıkarılabilir olduğunu kanıtlamaz; yalnızca okuyucunun cevabı yüzeysel olarak görüp görmediğini gösterir.

| Gold yüzeyi | Kontrol | Semantic |
|---|---:|---:|
| İki bağlamda da yok | 19 | 19 |
| Yalnız semantic bağlamında var | 4 | — |
| Yalnız kontrol bağlamında var | — | 1 |
| İki bağlamda da var | 24 | 24 |

Semantic kolun 41 hatasının:

- **20'sinde** gold yüzeyi semantic bağlamında yoktu.
- **21'inde** gold yüzeyi bağlamda vardı, fakat okuyucu yine yanlış cevap verdi.

İlk 20 hatanın **19'u iki kolda da** meydana geliyor; selector'ın çözemediği aday/kanıt kapsaması sorunu bunlarda ortak. Yalnız bir örnekte selector gold yüzeyini kaldırdı (`WH_dev_2782`); kontrol de EM yapamadığı için bu kayıp eşleştirilmiş EM kaybına dönüşmedi.

## 3. Değişen 12 kaynak kümesinin sonucu

- 4 örnekte semantic kolu gold cevap yüzeyini ekledi.
- 1 örnekte gold yüzeyi semantic kolunda kayboldu.
- 7 örnekte yüzey kapsaması değişmedi.
- Bu 12 örneğin hiçbirinde semantic kolu kontrolü EM'de geçemedi.
- Tek eşleştirilmiş kayıp `WH_dev_1228` oldu.

Örnekler:

| Kimlik | Gold | Kontrol | Semantic | Teşhis |
|---|---|---|---|---|
| `WH_dev_1228` | Hong Kong | Hong Kong | Yuen Long District | Selector ara konumu öne çıkardı; tek EM kaybı |
| `WH_dev_2782` | Vitebsk region | Vitebsk | Belarus | Semantic gold yüzeyini kaldırdı; hedef yerine üst/yan varlık geldi |
| `WH_dev_4482` | Massachusetts | unknown | Boston | Gold bağlama eklendi ama okuyucu ara şehri seçti |
| `WH_dev_3416` | prostitution | unknown | unknown | Gold yüzeyi semantic bağlamına girdi, okuyucu kullanamadı |
| `WH_dev_3233` | family name | unknown | clan | İlişki türü yerine kaynakta geçen ara sınıf seçildi |
| `WH_dev_1040` | Al Mahrah governorate | Yemen | Yemen | Hedef yerine ülke/ara varlık seçildi |
| `WH_dev_18` | Andhra Pradesh | Srikakulam district | Srikakulam district | Hedef yerine daha alt coğrafi varlık seçildi |
| `WH_dev_1996` | Vorarlberg | Lustenau | Lustenau | Hedef yerine bağlı şehir seçildi |

## 4. Reader hatalarının yapısı

Semantic kolunda 7/48 doğru, 41/48 yanlış çıktı. Gold yüzeyi bağlamda bulunan 21 yanlışta şu tekrar eden davranışlar görüldü:

1. **Ara varlık seçimi:** Boston yerine Massachusetts, Yemen yerine Al Mahrah, Yuen Long District yerine Hong Kong, Srikakulam district yerine Andhra Pradesh.
2. **Daha kısa veya daha genel ad:** Islington yerine London Borough of Islington, Harmonix yerine Harmonix Music Systems, Malvern yerine Malvern Hills, Academy Award yerine Academy Award for Best Picture.
3. **Unknown'a gereksiz dönme:** Gold yüzeyi bağlamda bulunan `WH_dev_4932`, `WH_dev_1634`, `WH_dev_2202`, `WH_dev_2323`, `WH_dev_3416` ve `WH_dev_4166` örneklerinde okuyucu `unknown` verdi.
4. **İlişki türünü yanlış çözme:** `WH_dev_3233` için `family name` yerine `clan`, `WH_dev_4289` için `guitarist` yerine `bassist`, `WH_dev_2975` için `house` yerine `stage stand` üretildi.

WikiHop'un `query + candidates` yapısına rağmen mevcut reader istemi yalnızca soru ve bağlamı gönderiyor; aday cevap listesi gönderilmiyor. Bu, hem hedef ilişkiyi ara ilişkiden ayırmayı hem de Exact Match'in istediği ad granülerliğini zorlaştırıyor.

## 5. Kök nedenler

### A. Atom granülerliği ve token bütçesi uyuşmuyor

WikiHop'ta her destek belgesi tek atom kabul edildi. İki veya üç uzun belgeyi citation tohumu olarak birlikte bağlama koymak çoğu zaman %25 bütçeyi aşıyor. Bu yüzden 11/48 örnekte semantic selector hiç devreye girmedi.

### B. Selector etkisi seyrek ve çoğu kez no-op

Geçerli citation'ların önemli bölümü zaten MiniLM sıralamasındaki cümleleri seçti. Bu nedenle 48 sorunun yalnız 12'sinde okuyucu bağlamı değişti. Bu pilot, selector'ın WikiHop üzerinde yeterli müdahale gücüne sahip olduğunu göstermedi.

### C. Verifier'ın ara ilişkiyi doğru kurması reader hedefini garanti etmiyor

Verifier çıktıları çoğu kez mantıklı ara zincirler kurdu; ancak reader son hedef yerine zincirdeki ara varlığı verdi. `WH_dev_1228` bunun EM'e dönüşen açık örneği.

### D. Reader WikiHop görev biçimine göre kısıtlanmamış

Reader aday cevapları ve hedef ilişki yönünü görmediği için `unknown`, kısa ad veya ara varlık üretmeye yatkın. Bu, selector hatasından bağımsız ikinci bir darboğaz.

## 6. Sonraki deney için karar

96 soruluk ayrılmış WikiHop doğrulaması çalıştırılmayacak. Yeni deneme yapılacaksa iki değişiklik birlikte uygulanmamalı:

1. **Önce selector yapısal onarımı:** destek pasajlarını kaynak makalesi kimliğini koruyarak cümle düzeyinde atomlara bölmek; aynı %25 bütçe ve aynı reader ile yeni, ön kayıtlı pilot çalıştırmak. Bu, 11 token-cap fallback'inin gerçekten atom granülerliğinden kaynaklanıp kaynaklanmadığını izole eder.
2. **Ayrı reader ablation'ı:** yalnız sonraki bir pilotta WikiHop'un zaten sağladığı aday cevap listesini reader'a vermek ve hedef ilişkiyi açıkça istemek. Bu, bağlam mevcutken oluşan 21 reader hatasını selector etkisinden ayırır.

Bu iki değişiklikten biri pilot kapısını geçmeden diğeriyle birleştirilmemeli. Faz 28 HotpotQA doğrulaması geçerli kalıyor; Faz 31 sonucu yöntemin WikiHop'a doğrudan genellenemediğini gösteriyor.
