# Faz 30C — Okuyucu `unknown` koruma ablation

**Tür:** Keşifsel, yeni donmuş alt örneklem  
**Kapsam:** 36 HotpotQA + 8 MuSiQue; Faz 30B’de kullanılmamış bağlamlar  
**Yeni okuyucu çağrısı:** 44

## Tasarım

Faz 30B’de gözlenen `unknown` kayıplarını azaltmak için selector ve semantic-joint-citations bağlamları sabit tutuldu. Faz 28 ve Faz 29 kayıtlarında kaynak cümlesi kümesi değişmemiş sorular arasından kimlik başına SHA-256 ile önceden belirlenmiş 36 HotpotQA ve 8 MuSiQue seçildi. Yeni istem, desteklenen bir cevap varsa doğrudan vermeyi ve `unknown`ı yalnızca cevapsız durumlara ayırmayı söyler.

Bu çalışma aynı public-validation arşivlerini kullandığı için confirmatory veya bağımsız benchmark sonucu değildir. Gold cevaplar yanıtlar dondurulduktan sonra okundu.

## Sonuç

- EM: semantic okuyucu 25/44 → reader-guard-v1 24/44; net -1, exact McNemar p = 1.
- F1 farkı: 0.0136; bootstrap %95 GA [-0.0136, 0.0545].
- Yeni `unknown`: 1; daha önce doğru olup `unknown`a dönen: 0.

- HotpotQA: EM 23/36 → 22/36; F1 0.7204 → 0.7370.
- MuSiQue: EM 2/8 → 2/8; F1 0.4333 → 0.4333.

## Yorum

Bu sonuç yalnızca okuyucu isteminin abstention davranışını ölçer. Pozitif sonuç, Faz 30B’deki kaybın istem kaynaklı olduğunu düşündürür; sonuç nötr veya negatifse okuyucu değişikliğini bırakıp selector kazanımını raporlamak gerekir.

Sonuç Faz 28’in confirmatory kararını değiştirmez.
