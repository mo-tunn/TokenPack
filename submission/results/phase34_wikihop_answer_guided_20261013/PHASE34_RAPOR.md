# Faz 34 — WikiHop Aday Cevap Yönlendirmeli Selector Raporu

**Tarih:** 13 Ekim 2026  
**Benchmark:** QAngaroo WikiHop v1.1 resmi `dev` bölümü  
**Pilot:** 48 yeni kimlik; Faz 31–33 kimlikleri dışlandı

## Hipotez

Faz 33'te aday cevap listesini yalnız reader'a vermek EM seviyesini yükseltti, fakat selector farkı net +1'de kaldı. Faz 34'te aynı aday cevap listesi verifier'a da verildi. Hipotez, verifier'ın ara varlık yerine hedef adayı destekleyen cümleleri seçmesinin semantic selector kazanımını artıracağıydı.

## Sonuç

| Kol | EM | Ortalama context tokenı |
|---|---:|---:|
| `atomic-sentence-rank` | **22/48 (%45,83)** | 441,58 |
| `semantic-joint-citations` | **24/48 (%50,00)** | 441,42 |

Eşleştirilmiş sonuç:

- Semantic kazanım: **2**
- Semantic kaybı: **0**
- Net: **+2**
- İki yönlü exact McNemar p: **0,5**
- Ön kayıtlı pilot eşiği: **en az +3 net EM**
- Karar: **`PILOT_STOP_NO_VALIDATION`**

Semantic kolunun kazandığı iki örnek:

- `WH_dev_1833`: kontrol `unknown`, semantic `maggie renzi`
- `WH_dev_2419`: kontrol `film`, semantic `silent film`

## Selector davranışı

- Kaynak kümesi 4/48 soruda değişti.
- 36/48 soruda semantic verifier sonucu geçerli citation üretmedi veya seçim baseline'a döndü.
- Verifier parse durumları: 12 geçerli citation, 19 numeric citation biçim hatası, 16 çekimserlik, 1 JSON biçim hatası.
- Token-cap ihlali: **0**.

Numeric citation hataları, modelin `0, 19` gibi tamsayı kimlikler üretmesi; parser ise `S00`, `S19` biçimini gerektiriyor. Bu, pilot sonucunu geriye dönük değiştirmiyor: hata kaydı korunmuş ve başarısız çağrılar tekrar otomatik olarak düzeltilmemiştir.

## Deney bütünlüğü

- Verifier: 48 gerçek çağrı.
- Reader: 96 yanıt; 52 gerçek çağrı, 44 yeniden kullanım.
- Gold cevaplar reader freeze sonrasında açıldı.
- Gold cevapları uzak modele gönderilmedi.
- Ayrılmış 96 soruluk doğrulama seti kullanılmadı.
- Faz 31, 32 ve 33 arşivlerine dokunulmadı.

## Yorum

Aday cevabı verifier'a vermek semantic selector'ı net +2'ye taşıdı; bu önceki +0 ve +1 sonuçlarına göre yönsel bir iyileşme. Ancak pilot kapısı ve anlamlılık koşulu sağlanmadı. Etkinin büyük bölümü yalnız dört değişen kaynak kümesinden geliyor ve verifier biçim uyumsuzluğu nedeniyle müdahale seyrek kaldı.

Faz 34, WikiHop'ta doğrulama iddiasını desteklemiyor. Sonraki deneme yapılacaksa önce citation şema uyumsuzluğu gold'a bakmadan düzeltilmeli ve bu değişiklik ayrı bir ön kayıtlı pilotta sınanmalı. Mevcut pilotun tamamlanmış sonucu yeniden skorlanmamalı.
