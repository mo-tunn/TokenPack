# TokenPack Faz 27 — Kaynak başlığıyla bağlanan kanıt çiftlerini önceliklendirme

**12 Ekim 2026 | HotpotQA distractor PUBLIC validation | 48 yeni pilot | Nihai karar: `STOP_PRE_GOLD_STRUCTURAL_FUTILITY`**

## Bilimsel neden

Faz 26'da 96 yeni HotpotQA sorusunun 36'sında bütün anotasyonlu özgün destek cümleleri %25 token bütçesine **sığdığı hâlde** global MiniLM cümle sıralaması bunların tamamını koruyamamıştı. Faz 27'de tek bir politika değişikliği tasarlandı: sorunun sağlanan Wikipedia kaynakları arasında, bir özgün cümle A'nın başka bir kaynağın başlığı B'yi açıkça anması durumunda **A cümlesi ile B kaynağının en yüksek skorlu cümlesini birlikte ilk seçmek**, kalan bütçeyi değişmemiş orijinal global cümle relevance sıralamasıyla doldurmak. Başlık bağlantısı yoksa kontrol çıktısı aynen korunur.

Bu mekanizma, gerçek iki-hop olgusal çıkarımın garantisi değil, yalnız **metinsel kaynak-başlığı bağlantısı**dır. Hiçbir gold cümle indeksi veya gold cevap, seçici/model girdisine verilmedi.

## Ön kayıt ve veri izolasyonu

- Kaynak: sabit HotpotQA `distractor` PUBLIC validation, revizyon `1908d6afbbead072334abe2965f91bd2709910ab`, SHA256 `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`.
- Toplam **7.405** validation sorusundan önceki Faz6/7/8'de kullanılmış 512 benzersiz ID ile Faz26'nın 96 pilot + **192 ayrılmış** ID'si, toplam **800**, çıkarıldı.
- Kalanlardan yalnız SHA256(`seed|id`) kullanılarak **48 yeni pilot + 96 yeni rezerv** kimliği önkayıtla sabitlendi.
- Kontrol: Faz8 `atomic-sentence-rank`, değiştirilmemiş pretrained MS MARCO MiniLM-L6-v2 skorları, tam özgün cümleler ve gerçek `cl100k_base` renderer.
- Tedavi: Faz27 önkayıtlı `phase27_joint_chain_core.py`: en yüksek skorlu **ilk 24** özgün cümleden biri, farklı bir kaynak başlığını noktalama/harf küçültme sonrası bütün sözcük ifadesi olarak taşıyorsa, bu cümle + ilgili kaynakta en yüksek MiniLM skorlu cümle çiftini, **iki orijinal cümlenin birlikte bütçeye sığması şartıyla** seed olarak seç. Birden fazla geçerli çiftte skor toplamına göre seç; kalanları kontrol sırasıyla greedily doldur. Eşleşme yoksa kontrol birebir. Aynı %25 sert token sınırı ve aynı kaynaklardan seçim.
- Ön kayıtlı geçiş: **Bütün gold destek cümlelerini koruyan soru sayısında en az +5 net** ve cevap doğru EM sorusunda **en az +3 net** kazanım, sıfır token cap ihlali. Ancak o zaman ayrılmış 96 kullanılabilecek.
- Sadece `id,question,context.title,context.sentences` fiziksel sütunları okundu; **cevap ve supporting_facts gold etiketleri hiç açılmadı**.

## Gerçekleşen gold-blind mekanizma testi

| Ölçüm | Sonuç |
|---|---:|
| Yeni 48 soru | 48 |
| Sıralanan orijinal cümle adayı | **1.965** |
| Metinsel başka başlığa bağlanan ve bütçeye sığan çift bulunan soru | **37/48** |
| **Gerçekten farklı özgün cümle seti seçilmiş soru** | **4/48** |
| Değişmeyen özgün cümle seti | **44/48** |
| Kontrol toplam seçilen cümle sayısı | 432 |
| Tedavi toplam seçilen cümle sayısı | 431 |
| Kontrol ortalama gerçek token | 324,94 |
| Tedavi ortalama gerçek token | 325,08 |
| Ortalama %25 token tavanı | 330,58 |
| Her iki kolda sert token sınırı ihlali | **0** |
| Kontrolün orijinal Faz8 renderer uyumsuzluğu | **0** |

**Matematiksel erken durdurma gerekçesi:**

Tam anotasyonlu gold destek cümlesi kapsaması, yalnız finalde seçilen özgün `(Wikipedia başlığı, cümle indeksi)` çiftlerine bağlıdır. **44 soruda** iki kolun seçtiği özgün cümle çiftleri **aynı**. Gold etiketlerinden bağımsız olarak bu 44 soruda iki kolun kapsama başarısı aynıdır. Farklılık bulunan **4 soruda**, tedavi kontrolü her soruda maksimum bir kez geçebilir. Dolayısıyla tüm 48 soru üzerinde olası net kapsama kazanımı için kesin **üst sınır +4 sorudur**. Ön kayıtlı birincil eşik ise **+5**.

**+4 (kanıtlanmış üst sınır) < +5 (ön kayıtlı eşik)** olduğundan, 48 sorunun tüm gold anotasyonları henüz hiç bilinmese bile pilot geçiş kapısının açılması matematiksel olarak mümkün değildir.

Bu nedenle **0 Qwen okuyucu çağrısı**, **0 reader EM puanlaması** ve **0 gold erişimi** ile `STOP_PRE_GOLD_STRUCTURAL_FUTILITY` kararı alındı. **96 ayrılmış validasyon sorusu kullanılmadı**.

Bu, tedavinin gold veya gerçek cevap EM'sinin daha kötü/iyi olduğunu göstermez. Yalnız bu mekanizmanın önkayıtlı destek kapsaması eşiğini sağlayamayacağını kesin olarak gösterir.

## Bilimsel yorum

Bir başka kaynağın başlığını cümlede görmek, kanıt zincirini güçlendirmek için yeterli değil: mevcut MiniLM global sıralayıcı zaten bu tip yüksek skorlu cümleleri sıkça seçmiş ve tedavi **48 örneğin sadece dördünü** değiştirmiştir. Faz26'daki `36/96 gold gerekli cümleler son bağlamdan eksik` bulgusu bu zayıf politika ile çözülmüş sayılmaz.

Sonraki deney, yalnız başlık eşleşmesine dayanmak yerine **açık soruya bağlı çok-hop yol/ilişki kanıtlarını** gerçek anlamda ayırt edebilen, gold-blind ve *önceden bağımsız yeni örneklemde kayıtlı* bir mekanizma olmalı. Ancak modelin yeni bir cümle seçmesi yetmez: hem **orijinal gold cümle kapsaması**, hem de **gerçek eşleştirilmiş okuyucu EM** artmalı. Aynı validation benchmark ailesinde çok sayıda küçük pilot denendiği için araştırmacı uyumu riski devam etmektedir.

## Veri ve altyapı bütünlüğü

- `PRE_GOLD_PROTOCOL.json`: SHA256 sabitlenmiş ID'ler, salt, yöntem ve iki geçiş eşiği.
- `CONTEXT_SELECTION_PRE_GOLD_FREEZE.json`, `gold_blind_paired_contexts.jsonl`, `CONTEXTS_FROZEN.json`: aynı 48 soruda iki gold-blind metin.
- `PRE_GOLD_FUTILITY_STOP.json`: cümle kümesi eşitliğinden çıkan **+4 kesin üst sınır** ve **+5 önkayıtlı eşik** kanıtı.
- `PHASE27_TURKCE_NIHAI_RAPOR.md`: bu nihai rapor.

**Faz 27'de hiç ücretli Qwen çağrısı yapılmadı.** Sadece yerel önceden eğitilmiş MiniLM cümle sıralayıcı çalıştırıldı. Önceki faz arşivleri, üretim TokenPack kodu, önceki ayrılmış veri ve resmî QASPER Faz3D mühürlü test değişmedi.

**Bilimsel karar: `STOP_PRE_GOLD_STRUCTURAL_FUTILITY`.** 96 ayrılmış soru kullanılmadan, cevap modeli çağrılmadan ve gold açılmadan durduruldu.
