# TokenPack Faz 26 — HotpotQA özgün destek cümlesi yeterliliği

**12 Ekim 2026 | HotpotQA distractor PUBLIC validation | 96 yeni pilot | Nihai bilimsel karar: DIAGNOSTIC_COMPLETE_NO_METHOD_PROMOTION**

## 1. Bilimsel gerekçe ve hipotez

Faz 23–25 boyunca 2Wiki ve MuSiQue üzerinde daha kısa cevap/alternatif adlandırma sorununun bazı örnekleri açıkladığı ancak yeni veri ailesinde baskın hata mekanizması olarak güçlü şekilde tekrarlanmadığı gözlendi. Faz 26, yeni bir paketleme yöntemi denemek yerine **bilinen kaynak ve okuyucu yöntemiyle üç aşamanın nerede başarısız olduğunu** ölçer: (A) gerekli orijinal destek cümlesi kaynak adaylarında yok; (B) kaynak adaylarında var ama son bağlamda bütün destek yok; (C) bütün anotasyonlu destek cümleleri finalde var ama doğrudan Qwen cevabı EM yanlış. Birinci ve ikincil metrikler gold açılmadan kayıt altına alındı.

**Sabit yöntem:** Önceki Faz 8 `atomic-sentence-rank` özgün kaynak cümlesi seçicisi, önceden eğitilmiş sıfır-shot `cross-encoder/ms-marco-MiniLM-L6-v2`, aynı tam kaynak sentence renderer, gerçek `cl100k_base` tokenlarla **kaynağın %25 üst sınırı** ve önceki Faz 7 kısa doğrudan Qwen27B okuyucu isteminin Faz 13 tam yanıt Modal bağlantısı.

Bu faz **iki yöntemin üstünlük testi değil**, gold anotasyonlu kanıt erişimi ile son cevap arasındaki gözlemsel tanı çalışmasıdır. Faz 8'in 152 sorudaki performansını yeni 96 soruyla deneysel doğrudan karşılaştırmak veya iki farklı pilot yüzdesinden nedensel üstünlük türetmek hatalıdır.

## 2. Veri izolasyonu ve ön kayıt

- Pinned HotpotQA `distractor` validation, HF revizyon `1908d6afbbead072334abe2965f91bd2709910ab`, SHA256 `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`.
- Validation'daki **7.405** sorudan Faz 6'nın kullanılmış **512** benzersiz sorusu dışlandı. Faz7'nin 256 ve Faz8'in 152 sorusunun Faz6'nın 512'sinin altkümesi olduğu dosyalarla doğrulandı. Bunlar yeniden kullanılmadı.
- Geri kalandan yalnız `SHA256(seed|question_id)` ile deterministik **96 pilot + 192 ayrılmış validation** sorusu seçildi. 192'ye dokunulmadı.
- Önce bağlam oluştururken yalnız `id,question,context.title,context.sentences` sütunları fiziksel olarak projekte edilip okundu. `answer`, `supporting_facts`, `type` okuyucuya veya reranker'a verilmedi.
- Önce bağlamlar **96/96** ve okuyucu ham yanıtları **96/96** SHA256 ile ayrı ayrı mühürlendi; ancak sonra gold orijinal `(Wikipedia title,sentence-index)` anotasyonları yalnız yerel puanlama için açıldı.
- İlk yerel kontrol şeması HF metadatasında gold alan adları görünmesinden dolayı false alarm verdi; gerçek fiziksel seçilen alanlar `schema.remove_metadata()` üzerinden yeniden denetlendi. Başarılı model çağrıları öncesinde kod kontrolü değişti, fakat örneklem, parametre, yöntem, gold filtre veya sonuç kuralı değişmedi.
- Qwen'nin ilk **sağlık kontrolü TimeoutError** verdi, başarısız istek ve hash `HOTPOT_READER_HEALTH_INCIDENT1_AUDIT.json` dosyasında korundu; bir açık yeniden denemede başarıyla **96/96** cevap toplandı. Başarısız istek ücretlendirilmiş olabilir.

## 3. Bağlam üretimi, maliyet ve fiziksel üst sınır

- **3.895** özgün kaynak cümlesi yerel sıralayıcıda değerlendirildi.
- **828** özgün kaynak cümlesi finalde seçildi.
- Ortalama gerçek final **322,16 token**; ortalama %25 hard cap **327,47 token**; **0 hard cap ihlali**.
- Yeniden uygulanan kayıt indeksleri, Faz 8 orijinal `choose_atoms(...,bridge=False)` metniyle **96/96 byte-eşdeğer** çıktı verdi.
- **96 başarılı Qwen27B Modal reader inference** isteği, artı bir başarısız health-check denemesi. Süre üzerinden USD tahmini kesin fatura değildir. Yerel MiniLM'nin çağrıları Modal ücretli inference değildir.

## 4. Birincil 96 soruluk sonuçlar

| Ölçüm | Değer |
|---|---:|
| Exact Match doğru | **56/96 = %58,33** |
| Answer F1 ortalaması | **%68,26** |
| Gereken bütün gold destek cümleleri orijinal verilen kaynakta | **96/96 = %100** |
| Gereken bütün gold destek cümleleri finalde | **60/96 = %62,50** |
| Seçilen gold destek cümleleri | **178/222** |
| Soru-başına gold cümle recall ortalaması | **%81,34** |

96 soru, anotasyonlu destek ve EM'ye göre tam dört ayrık sınıfa ayrılır:

| Kaynak/okuyucu aşaması | Soru | EM doğru / yanlış |
|---|---:|---|
| A. Gold destek cümlesi kaynak adaylarında eksik | **0/96** | — |
| B. Gold destek cümleleri kaynakta, son bağlamda en az biri eksik | **36/96** | 11 doğru / **25 yanlış** |
| C. Gold destek cümleleri son bağlamda tam, okuyucu EM yanlış | **15/96** | **15 yanlış** |
| D. Gold destek cümleleri son bağlamda tam, okuyucu EM doğru | **45/96** | **45 doğru** |

Böylece **40 EM yanlışından 25'inde** anotasyonlu gerekli cümlelerin bir kısmı finalden eksik; **15'inde** bütün anotasyonlu cümleler mevcut olsa da okuyucu yanlış.

Koşullu okuyucu EM: gold destek-cümlesi tam grupta **45/60=%75**, gold destek-cümlesi eksik grupta **11/36=%30,56**. Bu gözlemsel, örnekler zorluk bakımından kontrol edilmediğinden **nedensel model geliştirme etkisi değildir**. Faz 26 tüm 96 soruda EM nominal Wilson %95 GA **[%48,34; %67,69]**; tam destek seçimi **[%52,51; %71,53]**.

Soru türü keşifsel: bridge **78** soru (EM doğru 45, bütün anotasyonlu destek finalde 45), comparison **18** soru (EM doğru 11, bütün destek finalde 15). Alt-gruplar küçük ve farklı zorluktadır.

## 5. Gold-aware, gerçek model olarak kullanılamayan kapasite oracle'ı

Gold yalnız reader tamamlandıktan sonra kullanılarak, orijinal destek `(title,sent_id)` cümlelerini **aynı Faz8 renderer** ile tek bir paket olarak dökmenin gerçek cl100k token maliyeti hesaplandı:

- **96/96 soruda** gereken bütün gold cümleler **%25 hard-cap içine teorik sığıyor.**
- Yalnız-gold paketinin ortalama maliyeti **100,39 token**.
- Aynı soruların ortalama toplam izinli bütçesi **327,47 token**.
- **36 soru**, gereken anotasyonlu cümleler bütçeye sığdığı hâlde sabit gold-blind sıralama/seçim politikası onları finalde birlikte koruyamadı.
- Bu 36 sorunun **25'i EM yanlış**, **11'i EM doğru**; alternatif veya açıklanmamış kaynak cümleleri ile okuyucu cevabı elde edilmiş olabilir.

Bu oracle **gold'u bilerek cümle seçtiği için deploy edilemez**, production yerine kullanılamaz ve gold'u prompta koyma izni vermez. Ancak eksikliğin temel sebebinin **kanonik anotasyonlu destek cümlelerinin token bütçesine matematiksel olarak sığmaması olmadığını** gösterir.

## 6. Güçlü sonuç ve bilimsel sınırlamalar

- **Sonraki birincil bilimsel müdahale için gerekçe:** gold-blind yeniden sıralamada, sorunun gerektirdiği **kanıt zincirini birlikte kapsayan cümlelerin**, ayrı ayrı yüksek relevanslı ama birbirinden bağımsız cümlelere tercih edilip edilemeyeceği. Önce ayrı yeni soru ID'leri, sabit maliyet ve karşılaştırmalı önkayıt gereklidir.
- Sadece daha fazla kaynak veya cümle seçmek değil, **tam ilişki zincirini doğru orijinal cümlelerden kurmak** gerekecek; adlandırma/kısaltma hata nedeni 2Wiki örneklerinde görülmüş ama MuSiQue'de genellenmiş değil.
- Gold `supporting_facts` anotasyonları semantik entailment'ın matematiksel tam tanımı değildir; farklı alternatif cevap kanıtları mevcut olabilir. **11/36** eksik-gold ama doğru cevap vakası bu farkı görünür kılar.
- C grubundaki 15 hatanın tümünü doğrudan modelin mantıksal çıkarım yeteneği eksik diye atfetmek de uygun değildir; kaynak metindeki çelişkiler, soru-varyant ve cevap biçimi hataları araştırılmalıdır.
- Bu aşamada **yeni yöntemin başarı artışı gösterilmedi**, 192 rezerv soru açılmadı, üretim kodu değiştirilmedi, QASPER Faz3D resmî mühürlü test açılmadı.

## 7. Dosya arşivi

`submission\results\phase26_hotpot_sentence_sufficiency_20261012`

Önemli değişmez kayıtlar:

- `PRE_GOLD_PROTOCOL.json` — örnekleme, sabit yöntem, metrikler.
- `CONTEXT_SELECTION_PRE_GOLD_FREEZE.json`, `gold_blind_contexts.jsonl`, `CONTEXTS_FROZEN.json`.
- `HOTPOT_READER_PRE_GOLD_FREEZE.json`, `hotpot_reader_responses.jsonl`, `HOTPOT_READER_RESPONSES_FROZEN.json`.
- `HOTPOT_READER_HEALTH_INCIDENT1_AUDIT.json` ve ilk başarısız sağlık kontrolü ham dosyaları.
- `POST_GOLD_SENTENCE_SUFFICIENCY.json` — özgün gold (başlık,cümle indeksi) ve EM/F1 96 vaka.
- `POST_GOLD_NONDEPLOYABLE_ORACLE_CAPACITY.json` — gold-aware yalnız teorik sığabilirlik.
- `PHASE26_TURKCE_NIHAI_RAPOR.md` — bu bilimsel rapor.

**Nihai karar: `DIAGNOSTIC_COMPLETE_NO_METHOD_PROMOTION`.** Sonraki fazda başarıdan önce hangi cümlelerin birlikte kaynak-kanıt zinciri oluşturduğunu gold-blind tanımlamak gerekecek.
