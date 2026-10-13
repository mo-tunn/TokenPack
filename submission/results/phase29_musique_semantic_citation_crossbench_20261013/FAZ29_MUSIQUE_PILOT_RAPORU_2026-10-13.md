# TokenPack Faz 29 — MuSiQue Pilot Çapraz Kıyaslama Raporu

**Tarih:** 13 Ekim 2026  
**Karar:** STOP_NO_VALIDATION  
**Durum:** Pilot tamamlandı; bağımsız bütünlük denetimi geçti; ön kayıt gereği 96 soruluk doğrulama aşaması başlatılmadı.

## 1. Amaç ve kapsam

Faz 28’de kullanılan semantic-joint-citations bağlam seçicisi, daha önceki deney kimlikleri hariç tutularak MuSiQue’nin herkese açık validation bölümünde sınandı. Bu, HotpotQA’dan farklı bir benchmark ailesidir; ancak MuSiQue public validation daha önce Faz 9–11 ve Faz 25’te kullanıldığı için bağımsız, gizli bir test değildir. Önceki deneylerin 704 kimliği dışlandı; dışlamadan sonra 1.713 uygun kimlik vardı.

- Veri deposu: awinml/musique
- Sabit sürüm: 3ac762478df3609852f18dd33652a16820660a5e
- Satır sayısı: 2.417
- Veri SHA-256: f3b7c998962b0282479ddf58d2ea168f692adac391b88c03c87f66df3dcdd94a
- Önceden belirlenen tasarım: 48 pilot sorusu; ayrıca yalnızca iki pilot kapısı da geçerse kullanılacak 96 ayrılmış doğrulama kimliği.
- Kontrol kolu: Değiştirilmemiş atomic-sentence-rank.
- Deney kolu: Faz 28’de dondurulan semantic-joint-citations.
- Her iki kol: aynı Qwen okuyucu ve aynı yanıt istemi; bağlam token bütçesi aynı.

Altın alanlar (cevap, cevap takma adları, soru ayrıştırması ve supporting-paragraph etiketleri) okuyucu isteklerinin hiçbirine eklenmedi. Pilot cevapları tamamlanıp hash ile dondurulmadan etiketler açılmadı.

## 2. Önceden belirlenmiş yapısal kapı

Pilot bağlamlarında 48 sorunun 8’inde kaynak cümle kümesi değişti. Protokol, destek kapsamındaki gerekli net +5 artışın yapısal olarak mümkün sayılması için en az 5 değişmiş soru istiyordu. Bu eşik geçtiği için pilot okuyucu karşılaştırmasına izin verildi.

Semantic verifier’ın 48 isteğinde:
- 31 geçerli, birden fazla kaynağa dayanan citation kararı;
- 16 abstain;
- 1 tek kaynağa dayanan, zincir sayılmayan sonuç.

Buna göre bağlamların 8/48’inde kaynak cümlesi kümesi değişti; diğer sorularda kollar aynı bağlamı üretti.

## 3. Pilot sonuçları

| Ölçüt | Atomic-sentence-rank | Semantic-joint-citations | Eşleştirilmiş fark |
|---|---:|---:|---:|
| Canonical cevapta exact match (EM) | 15/48 (%31,25) | 17/48 (%35,42) | +2 soru; +4,17 yüzde puan |
| Tam destek-paragrafı kapsamı | 21/48 (%43,75) | 26/48 (%54,17) | +5 soru; +10,42 yüzde puan |
| Ortalama destek-paragrafı geri çağırımı | %76,04 | %81,77 | +5,73 yüzde puan |
| Ortalama cevap F1 | 0,4125 | 0,4403 | +0,0278 |
| Ortalama bağlam tokenı | 603,67 | 603,52 | −0,15 |

### Eşleştirilmiş belirsizlik ve testler

**EM**
- Deney kolu kazandı/kaybetti: 2/0; net +2.
- İki yönlü kesin McNemar p = 0,50.
- Eşleştirilmiş bootstrap %95 aralığı: 0,00 ila +10,42 yüzde puan.

**Tam destek-paragrafı kapsamı**
- Deney kolu kazandı/kaybetti: 5/0; net +5.
- İki yönlü kesin McNemar p = 0,0625.
- Eşleştirilmiş bootstrap %95 aralığı: +2,08 ila +18,75 yüzde puan.

**İkincil ölçütler**
- F1 farkı: +0,0278; bootstrap %95 aralığı −0,0278 ila +0,0972.
- Destek-paragrafı geri çağırım farkı: +0,0573; bootstrap %95 aralığı +0,0226 ila +0,0972.

Buradaki destek metriği, her anotasyonlu supporting paragrafın seçilen bağlamda en az bir özgün kaynak cümlesiyle temsil edilip edilmediğini ölçer. Bu, cümle düzeyinde entailment kanıtı değildir; tam paragraf kapsamı, seçilen cümlenin yanıtı anlamsal olarak gerektirdiğini tek başına göstermez. Ayrıca sonuçlar 48 soruluk pilot tahminleridir; gizli test değerlendirmesi olarak yorumlanmamalıdır.

## 4. Ön kayıt kapıları ve karar

Protokolün pilot kapıları şunlardı:
1. Net tam destek-paragrafı kapsamı artışı en az +5: **GEÇTİ** (+5).
2. Net EM doğru cevap artışı en az +3: **KALDI** (+2).
3. Render edilmiş bağlamlarda token bütçesi ihlali sıfır: **GEÇTİ** (0 ihlal).

Her iki birincil pilot kapısının da geçmesi gerekiyordu. EM kapısı başarısız olduğu için nihai karar **STOP_NO_VALIDATION**. Ayrılmış 96 kimliğe okuyucu çağrısı yapılmadı ve validation bölümü için puanlama yürütülmedi. Bu karar sonrası eşikleri değiştirmek ya da pilot üzerinde yöntemi ayarlayıp aynı pilotu yeniden sınamak ön kayıtla uyumlu olmazdı.

## 5. Bütçe ve okuyucu çıktılarının dondurulması

- Pilot sorusu: 48.
- Dondurulmuş okuyucu görevleri: 96 (her soru için iki kol).
- Gerçek ücretli okuyucu çağrısı: 56.
- Birebir aynı prompt hash’i nedeniyle yeniden kullanılan yanıt: 40.
- Boş okuyucu tahmini: 0.
- Bağlam token bütçesi ihlali: 0.
- Ortalama kap: 606,27 token; ortalama gerçek bağlam uzunluğu her iki kolda yaklaşık 604 token.
- Gerçek ücretli verifier çağrısı: 48.

Yanıtların dondurma SHA-256 değeri: 6c1cbc4f43026684efae4918a8106f5ce6b958e8dff10a4cc3ec496c5255e1c8  
Dondurulmuş bağlam SHA-256 değeri: b97fe149ed3633496815f779826f554805fbd9a114bdf3ed37223a60ce51ad39  
Puanlama sonuç dosyası SHA-256 değeri: 8d4ce6b5f81512f93d966f31304464c09cd0ef9dabaa88534c47f026ee167a99  
Verifier cevapları SHA-256 değeri: af373bf3c537f835eb431ed0fa4b6511171da5abd3a7da9a6bd9a260f344b3de

## 6. Bağımsız bütünlük denetimi

Bağımsız denetim durumu: **PASS_INDEPENDENT_RECOMPUTATION**.

Denetimde:
- Veri hash’i ve ön kayıtta listelenen tüm kod hash’leri doğrulandı.
- Önceden kaydedilmiş pilot kimlikleri birebir eşleşti.
- Candidate, verifier, context ve reader dosyalarının hash’leri doğrulandı.
- 48 verifier ve 96 reader prompt hash’i bağımsız olarak tekrar hesaplandı.
- 40 aynı-prompt yanıt yeniden kullanımı ve 56 gerçek reader çağrısı doğrulandı.
- EM ve tam destek-paragrafı kapsamı sonuçları ile kesin McNemar p değerleri yeniden hesaplanarak eşleşti.
- Maksimum token kap ihlali 0 olarak doğrulandı.
- Bağımsız hesaplama kararı tekrar STOP_NO_VALIDATION olarak verdi.

Denetim sonuç SHA-256’sı: 8d4ce6b5f81512f93d966f31304464c09cd0ef9dabaa88534c47f026ee167a99

## 7. Etik, veri ve genellenebilirlik notları

MuSiQue public validation; önceki deney kimlikleri dışlanmış olsa da araştırmacı uyarlaması riski taşır ve bağımsız gizli değerlendirme değildir. Bu yüzden bulgu yalnızca ön kayıtlı bu pilot için geçerlidir; doğrulama başarısı olarak sunulmamalıdır. Destek-paragrafı kapsamı, cümle düzeyinde doğruluk ya da nedensel kanıtla eşdeğer değildir.

QASPER resmi sealed test açılmadı. Üretim kodu değiştirilmedi. Ayrılmış 96 doğrulama kimliği dokunulmadan bırakıldı. Pilot başarısızlığına rağmen daha fazla model çağrısı veya sonradan eşik ayarlaması yapılmadı.

## 8. Dosyalar

Ana sonuç klasörü:
submission/results/phase29_musique_semantic_citation_crossbench_20261013/

Önemli dosyalar:
- PRE_GOLD_PROTOCOL.json
- pilot/PRE_GOLD_STRUCTURAL_DECISION.json
- pilot/CONTEXTS_FROZEN.json
- pilot/VERIFIER_RESPONSES_FROZEN.json
- pilot/READER_RESPONSES_FROZEN.json
- pilot/PILOT_POST_GOLD_RESULTS.json
- pilot/INTEGRITY_INDEPENDENT_AUDIT.json

**Sonuç:** Semantic-joint-citations bu 48 soruluk pilotta destek-paragrafı kapsamını artırdı ve EM’de küçük bir artış gösterdi; ancak önceden belirlenmiş EM geçiş eşiğini karşılamadı. Bu nedenle Faz 29 doğrulaması durduruldu; mevcut kanıt doğrulama başarısını desteklemiyor.