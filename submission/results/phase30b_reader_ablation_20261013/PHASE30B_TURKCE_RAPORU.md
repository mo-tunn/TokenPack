# Faz 30B — Donmuş semantic bağlamlarda okuyucu ablation

**Tarih:** 13 Ekim 2026  
**Tür:** Keşifsel okuyucu tanı çalışması  
**Kayıt:** 44 değişmiş kaynak-cümlesi bağlamı  
**Yeni ücretli okuyucu çağrısı:** 44

## Tasarım

Semantic-joint-citations tarafından seçilmiş ve Faz 28/Faz 29’da kaynak cümlesi kümesi değişmiş bağlamlar aynen korunmuştur. Selector, soru kimlikleri, bağlamlar, model, sıcaklık, seed ve çıktı bütçesi değiştirilmemiştir. Tek değişken, okuyucu istemine kanıtları içsel olarak bağlamayı ve yalnız kısa cevap döndürmeyi söyleyen `reader-chain-v1` yönergesidir.

Çalışma aynı mevcut public-validation sorularını kullandığı için bağımsız doğrulama değildir. Gold cevaplar 44 yeni okuyucu kaydı tamamlanıp SHA-256 ile dondurulduktan sonra yerel olarak okunmuştur.

## Sonuç

- EM: semantic okuyucu 26/44 → reader-chain-v1 20/44; net -6, exact McNemar p = 0.03125.
- F1: ortalama 0.7040 → 0.5162; fark -0.1879, bootstrap %95 GA [-0.3064, -0.0844].
- Reader-chain-v1 boş cevapları: 0; `unknown` yanıtları: 13.
- Baseline’da doğru olup reader-chain-v1 tarafından `unknown` yapılan cevap: 6.

| Veri kümesi | n | Baseline EM | Reader-chain-v1 EM | Baseline F1 | Reader-chain-v1 F1 |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 36 | 24/36 | 19/36 | 0.8049 | 0.6031 |
| MuSiQue | 8 | 2/8 | 1/8 | 0.2500 | 0.1250 |

## Yorum

Bu sonuç yalnızca okuyucu isteminin, semantic selector’ın zaten değiştirdiği bağlamlarda cevap üretimini değiştirip değiştirmediğini ölçer. Pozitif bir sonuç okuyucu darboğazı için kanıt sağlar; negatif veya belirsiz bir sonuç selector’ın destek kazanımının cevap üretimine aktarılmadığını gösterir.

Bu varyantta EM kazancı görülmedi: altı eşleşmiş EM değişiminin tamamı kayıptı. Önceden doğru olan altı cevap yeni istemde `unknown` oldu. Dolayısıyla bu özel ‘kanıtları içsel olarak bağla’ talimatı okuyucuyu daha temkinli yaptı ve tanısal örneklemde performansı düşürdü. Bu bulgu başka okuyucu varyantlarını sınamaz.

Bu ablation’ın sonucu Faz 28’in confirmatory kararını değiştirmez ve yeni bir benchmark genelleme iddiası oluşturmaz.
