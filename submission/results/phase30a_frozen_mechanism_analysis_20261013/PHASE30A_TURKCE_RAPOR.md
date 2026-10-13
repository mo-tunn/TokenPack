# Faz 30A — Donmuş kayıtların mekanizma ve heterojenlik analizi

**Tarih:** 13 Ekim 2026  
**Kapsam:** Faz 28 HotpotQA doğrulaması + Faz 29 MuSiQue pilotu  
**Uzak model çağrısı:** 0

## Amaç

Bu analiz, önceden tamamlanmış ve bağımsız denetimden geçmiş sonuç kayıtlarını yeniden kullanır. Yeni soru seçimi, yeni gold açılması veya yeni model çağrısı yapılmamıştır. Amaç, semantic-joint-citations yönteminin destek seçimi kazanımının hangi koşullarda cevap doğruluğuna dönüştüğünü ölçmektir.

Faz 28 doğrulaması önceden kayıtlı confirmatory sonuçtur. Faz 29 pilotu çapraz-benchmark yön göstergesidir. Birleştirilmiş sonuçlar yalnızca açıklayıcıdır ve yeni bir doğrulama p değeri olarak yorumlanmaz.

## Veri bütünlüğü

- HotpotQA kayıtları: 96 soru; bağımsız denetim `PASS_INDEPENDENT_RECOMPUTATION`.
- MuSiQue kayıtları: 48 soru; bağımsız denetim `PASS_INDEPENDENT_RECOMPUTATION`.
- HotpotQA sonuç SHA-256: `d9c25ce866e180ce8b0a2c77eddd3e0512bf1e77858609964d3cf2e3c4597d1e`.
- MuSiQue sonuç SHA-256: `8d4ce6b5f81512f93d966f31304464c09cd0ef9dabaa88534c47f026ee167a99`.
- Analiz yalnızca mevcut `per_case` kayıtlarını okur; `remote_calls=0`.

## Veri kümesi bazında özet

### Faz 28 — HotpotQA

- EM: 48/96 → 61/96; net 13, exact McNemar p = 0.000244140625.
- Tam destek kapsaması: 47/96 → 73/96; net 26, exact McNemar p = 2.98023223877e-08.
- Ortalama destek recall farkı: 13.54%; ortalama F1 farkı: 14.98%.
- Kaynak cümlesi kümesi 36/96 soruda değişti.

| Grup | n | EM kontrol → yeni | Tam destek kontrol → yeni | Δ destek recall | Δ F1 | Δ token |
|---|---:|---:|---:|---:|---:|---:|
| bridge | 79 | 37/79 → 49/79 | 32/79 → 57/79 | 16.03% | 16.94% | 0.48 |
| comparison | 17 | 11/17 → 12/17 | 15/17 → 16/17 | 1.96% | 5.88% | 0.00 |

### Faz 29 — MuSiQue

- EM: 15/48 → 17/48; net 2, exact McNemar p = 0.5.
- Tam destek paragrafı kapsamı: 21/48 → 26/48; net 5, exact McNemar p = 0.0625.
- Ortalama destek recall farkı: 5.73%; ortalama F1 farkı: 2.78%.
- Kaynak cümlesi kümesi 8/48 soruda değişti.

| Grup | n | EM kontrol → yeni | Tam destek kontrol → yeni | Δ destek recall | Δ F1 | Δ token |
|---|---:|---:|---:|---:|---:|---:|
| 2-hop | 18 | 7/18 → 9/18 | 12/18 → 14/18 | 5.56% | 11.11% | -0.17 |
| 3-hop | 18 | 4/18 → 4/18 | 7/18 → 7/18 | 5.56% | -3.70% | -0.22 |
| 4-hop | 12 | 4/12 → 4/12 | 2/12 → 5/12 | 6.25% | 0.00% | 0.00 |

## Mekanizma: destek kazanımı neden her zaman EM kazanımı olmadı?

| Veri kümesi | Destek recall artan | Bu artışta EM de artan | Destek artıp EM artmayan | EM artıp destek artmayan |
|---|---:|---:|---:|---:|
| HotpotQA | 30 | 12 | 18 | 1 |
| MuSiQue | 8 | 2 | 6 | 0 |

HotpotQA’da destek recall artışı olan vakaların %40.0’inde EM de arttı. MuSiQue’de bu oran %25.0 oldu. Bu, seçicinin kanıt kapsamını iyileştirmesinin cevap okuyucusu için yeterli veya doğru kullanılmış kanıt garantisi vermediğini gösteriyor.

Başka bir deyişle MuSiQue sonucu yöntem etkisinin yok olduğunu göstermiyor; destek seçimi ile cevap üretimi arasındaki aktarımın veri kümesine ve soru yapısına bağlı olduğunu gösteriyor.

## Heterojenlik bulguları

- HotpotQA’da değişen bağlamlar: 36 soru; değişmeyen bağlamlar: 60 soru.
- MuSiQue’de değişen bağlamlar: 8 soru; değişmeyen bağlamlar: 40 soru.
- Değişmeyen bağlamlarda delta sıfırdır; bu beklenen bir kontrol ve aynı istem yanıtlarının yeniden kullanımının doğrulanmasıdır.
- MuSiQue hop katmanları arasında etki büyüklüğü farklıdır; küçük alt gruplardaki p değerleri keşif amaçlıdır.

### HotpotQA: kaynak cümlesi kümesi değişimine göre

| Grup | n | EM kontrol → yeni | Tam destek kontrol → yeni | Δ destek recall | Δ F1 | Δ token |
|---|---:|---:|---:|---:|---:|---:|
| False | 60 | 37/60 → 37/60 | 45/60 → 45/60 | 0.00% | 0.00% | 0.00 |
| True | 36 | 11/36 → 24/36 | 2/36 → 28/36 | 36.11% | 39.95% | 1.06 |

### MuSiQue: kaynak cümlesi kümesi değişimine göre

| Grup | n | EM kontrol → yeni | Tam destek kontrol → yeni | Δ destek recall | Δ F1 | Δ token |
|---|---:|---:|---:|---:|---:|---:|
| False | 40 | 15/40 → 15/40 | 21/40 → 21/40 | 0.00% | 0.00% | 0.00 |
| True | 8 | 0/8 → 2/8 | 0/8 → 5/8 | 34.38% | 16.67% | -0.88 |

## Açıklayıcı havuz özeti

İki arşiv birlikte 144 eşleştirilmiş soru içerir. EM 63/144 → 78/144; tam destek 68/144 → 99/144. Ortalama destek recall farkı 10.94%, ortalama F1 farkı 10.91% ve ortalama token farkı 0.22.

Bu havuz özeti iki farklı benchmark ailesini tek bir confirmatory test gibi birleştirmez. Yalnızca yönün ve mekanizmanın genel görünümünü verir.

## Karar ve sonraki deney için sonuç

1. Semantic-joint-citations’ın destek seçimi üzerindeki etkisi iki arşivde de pozitiftir.
2. Cevap doğruluğu etkisi HotpotQA’da güçlü, MuSiQue’de küçük ve belirsizdir.
3. Bir sonraki deney selector’ı rastgele değiştirmemeli; önce okuyucunun iyileştirilmiş kanıtı kullanamama vakalarını ayrı bir ön kayıtlı ablation olarak test etmelidir.
4. Yeni bir benchmark doğrulaması yapılacaksa yöntem ve okuyucu sürümü önceden dondurulmalı, ayrı bir pilot ve doğrulama seti kullanılmalı, Faz 29’un ayrılmış 96 kimliği açılmamalıdır.

**Faz 30A sonucu:** Kanıt seçimi kazanımı tekrarlanıyor; EM kazanımı heterojendir. En güçlü sonraki aday, semantic selector sabitken cevap okuyucusunun kanıtı kullanma davranışını ölçen ayrı bir deneydir.
