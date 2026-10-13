# TokenPack Faz 28 — Önceden ayrılmış 96 HotpotQA sorusunda doğrulanan kaynak-kanıt zinciri yöntemi

**13 Ekim 2026 | Sonuç: `VALIDATION_BOTH_PRIMARY_SIGNIFICANT` | Veriler: HotpotQA distractor PUBLIC validation**

## Yönetici özeti

Faz26 tanısı, HotpotQA'da bütün anotasyonlu gold özgün kanıt cümleleri %25 token sınırına sığmasına rağmen mevcut global cümle-sıralayıcının bunları her zaman birlikte seçemediğini bulmuştu. Faz27'de yalnız özgün kaynak başlığı zikrini teşvik eden seçici 48 sorunun sadece 4'ünü değiştirdi ve +5 tam kanıt kapsaması önkayıt eşiğini matematiksel olarak sağlayamayacağı için hiç okuyucu/gold açmadan durduruldu.

**Faz28** eski Faz8 sıfır-shot MiniLM cümle sıralamasını ve kaynak/25% token sınırını sabit tutup, tek yeni bileşen olarak **Qwen27B ile gold-blind, iki farklı Wikipedia kaynağından 2–3 özgün kanıt cümlesinin sorunun ara-varlık ve hedef ilişkisini birlikte desteklemesini isteyen kaynak-atif denetimini** ekledi. Qwen'in atıfları yine yalnız model yargısıdır, gerçek entailment ispatı değildir.

Önkayıtlı 48 soruluk pilotta tam anotasyonlu özgün cümle desteği 28/48→41/48 ve doğru Exact Match 25/48→32/48 arttı. Önceden kapatılmış/ayrılmış başka **96 yeni ID** üzerinde **değişmemiş yöntemle** test yapıldı ve iki birincil ölçüt daha sıkı Bonferroni düzeltilmiş eşiği de geçti.

### Esas 96 soruluk doğrulama sonucu

| Ölçüm | Kontrol: Faz8 atomic-sentence-rank | Yeni: semantic-joint-citations | Eşleştirilmiş değişim |
|---|---:|---:|---:|
| Exact Match doğru | **48/96 (%50,00)** | **61/96 (%63,54)** | **+13 doğru, +13,54 yüzde puan** |
| Answer F1 ortalaması | **%59,11** | **%74,09** | **+14,98 yüzde puan** |
| Tüm anotasyonlu gold özgün destek cümleleri seçilmiş | **47/96 (%48,96)** | **73/96 (%76,04)** | **+26 tam, +27,08 yüzde puan** |
| Ortalama anotasyonlu özgün destek cümlesi recall | **%77,65** | **%91,20** | **+13,54 yüzde puan** |
| Gerçek cl100k bağlam tokenı, ortalama | 325,32 | 325,72 | **+0,40 token** |
| %25 gerçek render cap ihlali | **0** | **0** | — |

İki **önkayıtlı ortak birincil** eşleştirilmiş son nokta:

- **EM**: yalnız yeni doğru **13**, yalnız kontrol doğru **0**, net **+13**, iki yönlü **exact McNemar p = 0,000244140625**. 10.000 eşleştirilmiş soru-bootstrap **nominal** %95 GA farkı **[+7,29; +20,83] yüzde puan**.
- **Bütün orijinal gold supporting_fact cümleleri seçilmiş**: yalnız yeni tam **26**, yalnız kontrol tam **0**, net **+26**, iki yönlü **exact McNemar p = 2,9802322387695312e-8**. Nominal %95 bootstrap fark GA **[+18,75; +36,46] yüzde puan**.
- **Doğrulama geçiş protokolü**: ikisi de pozitif VE ikisi de **Bonferroni α=0,025** altında exact p VE token cap ihlali 0. **Her üç koşul sağlandı.** Sonuç dosyasında `VALIDATION_BOTH_PRIMARY_SIGNIFICANT`.

Yan sonuç Answer F1 artışı +14,98 yüzde puan (nominal eşleştirilmiş bootstrap %95 fark aralığı +8,40 ila +22,31 yp). Bu F1 ölçümü ikincildir; önkayıtlı iki birincil ölçütün yerine kullanılmamıştır. 48 soruluk pilot ile 96 soruluk doğrulama, tek bir bağımsız p değeri oluşturmak için birleştirilmemiştir.

### Soru türleri, yalnız tanı

| Gold soru türü | n | Kontrol EM → Yeni EM | Kontrol tam destek → Yeni tam destek |
|---|---:|---:|---:|
| `bridge` | 79 | **37 → 49** | **32 → 57** |
| `comparison` | 17 | **11 → 12** | **15 → 16** |

Altgruplar küçük ve doğrulamanın bağımsız birincil hipotezi değildir. Tedavi, 96 sorunun **36'sında** özgün kaynak cümlesi setini değiştirdi. Kalan 60 soruda aynı cümlelerden oluşan istemlerin okuyucu cevabı byte-eşdeğer kaynak istemleri üzerinden **tekrar Qwen çağrılmadan** kullanıldı.

## Önkayıt, ID izolasyonu ve saf deney farkı

- Kaynak: HotpotQA `distractor` **kamuya açık validation** Parquet, hash `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`. **Gizli test değildir**.
- Faz28'in ilk önkaydında, HotpotQA Faz6/7/8'in 512 ve Faz26'nın 96 pilot+192 ayrılmış ve Faz27'nin 48 pilot+96 ayrılmış kimliği olmak üzere toplam **944 önceki kullanılan veya ayrılmış ID dışlanmıştır**.
- SHA256(`seed|id`) ile **48 Faz28 pilot + 96 önceden rezerv** sabitlenmiştir. Pilot +7 EM/+13 tam destek eşiğini geçtikten sonra doğrulama ID'leri **yeniden seçilmemiştir**.
- Doğrulama önkayıtı, herhangi bir yeni doğrulama modeli çalıştırılmadan önce `submission/results/phase28_hotpot_semantic_citation_validation_20261013/PRE_GOLD_PROTOCOL.json` dosyasında iki co-primary ve α=0,025 kuralını sabitlemiştir.
- Kontrol: mevcut Faz8 `atomic-sentence-rank`, yerel pretrained MS MARCO MiniLM kaynak-cümle sıralaması, değişmemiş (cl100k\_base\) renderer, mevcut **%25 tam kaynak token bütçesi**.
- Yeni kol: aynı MiniLM skorları ile **top-24 özgün cümle + her kaynak belgenin en yüksek skorlu bir özgün cümlesi** (toplam en fazla 36) arasından Qwen27B'nin **iki ayrı kaynakta bulunan 2–3 cümleyi**, sorunun ara-varlık/ilişki zincirini anlatabilen `citation_ids` biçiminde seçmesi. Kaynak/citation parser atıfların aynı orijinal havuzda olduğunu ve en az iki farklı kaynak içerdiğini doğrular. Atıflar %25 bütçeye sığarsa önce bu cümleler, ardından özgün global skor sırası ile greedy tamamlanır; çekimserlik/bozuk tek-kaynak çıktısı veya sığmayan tohumda **kontrolün orijinal cümleleri aynen korunur**.
- Her iki kolun **aynı soruda aynı kısa Qwen27B okuyucu promptu**, sıcaklık 0, aynı eski Faz13 tam 4.000 karakterlik röle ve çıktı bütçesi kullanılır. Gold `answer` veya `supporting_facts` hiç bir model istemine gönderilmemiştir.
- Orijinal kaynak cümlelerin tam korunması gösterildi; kaynak anotasyonu `(Wikipedia başlığı, orijinal cümle indeksi)` ile yalnız **model tahminleri ve final context freeze tamamlandıktan sonra** yerel olarak karşılaştırıldı.

## Uzak model maliyeti ve hata kontrolü

**Doğrulama aşaması**:
- **96 gerçek Qwen27B anlamsal kaynak-atıf çağrısı**; **89** vakada farklı kaynaklara ait citation seti önerildi ve **7** vakada geri dönüş uygulandı.
- İki kolda **192 okuyucu kaydı**, fakat yalnız **132 gerçek Qwen27B okuyucu çağrısı**; **60 adet byte-eşdeğer okuyucu isteminin** ham cevabı yeniden kullanıldı.
- Toplam **228 başarılı Qwen27B uzak çağrısı** doğrulama aşamasında. 48 soruluk pilotta ayrıca **48 verifier + 63 reader = 111 başarılı çağrı** vardı: araştırmanın iki aşamasında toplam **339 başarılı model çağrısı**. Gerçek Modal fatura dökümü incelenmedi; USD tahmini kesin fatura değildir.
- Üretim kullanımı koşulları farklıdır: Model seçimi için ekstra verifier çağrısı vardır; bu çalışma gecikme/hesaplama açısından temel okuyucuya ücretsiz bir iyileştirme iddia etmez.
- Aynı kontekstli 60 yanıtta Qwen tekrar çağrısı yerine önceden kaydedilmiş birebir istem hash'i kullanıldı; bu davranış hem kod hem tekrar-hash denetiminde doğrulandı.

## Bağımsız yerel tekrar denetimi

`VALIDATION_INTEGRITY_INDEPENDENT_AUDIT.json` dosyası, orijinal gold skor betiğinden bağımsız ikinci kontrolle şu sonuçları doğruladı:

1. Faz28 ilk önkayıt rezervindeki **tam 96 ID**, yeniden seçilmeden validation'da kullanılmıştır; 48 pilot ID ile çakışma yoktur.
2. Pilot seçim çekirdeğinin kaynak SHA-256 hash'i **aynıdır**.
3. **96/96 verifier istemi** ve **192/192 okuyucu istemi** soru + kaynak metninden yeniden üretilen istem SHA256'larıyla eşleşmiştir.
4. **60/60** yeniden kullanılan okuyucu cevabı, daha önceki aynı istem hash'ine ve aynı yanıta sahiptir.
5. Bağlamlar, her sorunun her kolunda %25 hard cap'i ihlal etmemiştir.
6. Eşleştirilmiş EM ayrışması **13 kazanç, 0 kayıp** ve tam destek ayrışması **26 kazanç, 0 kayıp**; iki exact McNemar p değeri ikinci kez aynı çıkmıştır.
7. `VALIDATION_POST_GOLD_RESULTS.json` skorları, `reader_responses.jsonl` ve `gold_blind_paired_contexts.jsonl` dosyalarının dondurulmuş SHA256 değerleriyle ilişkilidir.

Önce ilk skor betiğinin validation-only yapısal sayaç adı pilotunkinden farklı olduğu için bir `KeyError` oluştu. **Ölçüm kuralı değiştirilmeden** sadece sayaç anahtarı düzeltildi; ilk denemede sonuç dosyası oluşturulmadı. Tekrar çalışan değerlendirme önkayıtlı sonuçları kaydetti.

## Bilimsel yorum ve sınırlar

**Güçlü ve sınırları belirli sonuç:** Kamuya açık ama ilk Faz28 pilotunda hiç kullanılmamış, önceden dondurulmuş 96 doğrulama sorusunda kaynak-bağlı Qwen cümle seçimi, aynı kaynak ve token bütçesinde, eşleştirilmiş EM ve anotasyonlu destek-cümlesi kapsamasını **önkayıtlı anlamlılık kuralı altında birlikte iyileştirmiştir**.

Fakat:
- Bu **resmî mühürlü QASPER testi veya dışarıdan yöneticisi olan bağımsız gizli test değildir**. Eski fazlar HotpotQA validation ailesi üzerinde birçok yöntem denedi; çoklu araştırmacı tercihleri ve benchmark'a aşinalık etkisi daha geniş genellenebilirlik iddialarını sınırlar.
- Qwen seçtiği cümlelerin birlikte gerçekten mantıksal entailment oluşturduğunu ispatlamaz; `supporting_facts` anotasyonu örtüşmesi de tek başına “anlamsal kaynak doğruluğu” değil, orijinal `title+sent_id` kapsamasıdır.
- Bağımsız kör insan anlamsal hakemliği, çapraz-veri kümesi karşılaştırması, üretim gecikme/maliyet denetimi ve önceki mühürlü teste erişmeden gerçekten bağımsız değerlendirme **henüz yapılmamıştır**.
- Dolayısıyla sonuç **genellenebilir nihai ürün doğruluğunun garanti edilmesi değil**, güçlü bir yöntem kazanımının sabit rezerv soru setinde tekrarlanmış olmasıdır. Üretim kodu değiştirilmedi; resmî QASPER mühürlü test açılmadı.

## Sonraki bilimsel öncelik

1. Faz28 yöntemini **değiştirmeden** candidate shortlist, Qwen verifier yanıt biçimi, cümle seçimi ve okuyucu istemlerini sürüm kilidine almak.
2. Kullanılmamış **ayrı benchmark ailesinde** önceden kayıtlı ID'ler ve açık veri izolasyonu ile yeniden doğrulama yapmak; kullanıcı onayı veya uygun veriye erişim olmadan mühürlü QASPER resmî testi açmamak.
3. Kör anlamsal atıf hakemliği, hem inference maliyetini hem üretim gecikmesini içeren değerlendirme yapmak.
4. Yalnızca genellenebilirlik ve maliyet koşulları sağlandıktan sonra üretim koduna promote etmek.

**Nihai karar: `VALIDATION_BOTH_PRIMARY_SIGNIFICANT`. Pilot ve doğrulama arşivleri mühürlü, üretim kodu değişmedi.**

### Windows dosyaları

Pilot: `submission\results\phase28_hotpot_semantic_citation_chain_20261013`

Önkayıtlı doğrulama: `submission\results\phase28_hotpot_semantic_citation_validation_20261013`

Doğrulama dosyaları: `PRE_GOLD_PROTOCOL.json`, `CANDIDATES_FROZEN.json`, `VERIFIER_RESPONSES_FROZEN.json`, `CONTEXTS_FROZEN.json`, `READER_RESPONSES_FROZEN.json`, `VALIDATION_POST_GOLD_RESULTS.json`, `VALIDATION_INTEGRITY_INDEPENDENT_AUDIT.json`, `PHASE28_TURKCE_NIHAI_DOGRULAMA_RAPORU.md`.
