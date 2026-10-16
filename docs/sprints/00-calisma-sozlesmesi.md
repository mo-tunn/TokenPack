# Ortak çalışma sözleşmesi

Bu talimatlar altı sprintin tamamında geçerlidir. Hedef, TokenPack'teki somut kusurları düzeltmek ve araştırmayı güvenilir biçimde sürdürülebilir hale getirmektir. Genel öneri veya yalnız plan üretmek yeterli değildir.

## Başlangıç ve bağlam

1. Geçerli repo talimatlarını (`AGENTS.md`, mevcut `AGENT.md` ve alt dizin talimatları varsa) oku.
2. `git status --short`, mevcut HEAD ve ilgili dosyaları incele. Kullanıcıya ait değişikliklerin üzerine yazma; ilgisiz dirty dosyalar çalışmayı engellemez.
3. `docs/audit/2026-09-16-inceleme-ve-iyilestirme-plani.md`, bu sprintin promptu, `docs/sprints/STATUS.md` ve bağımlı sprintlerin sonuçlarını oku.
4. Audit bir tarihsel incelemedir: bulguyu güncel kodda kontrol et, çözülmüş işi yeniden uygulama. Dosya/isimler değişmişse mevcut karşılığını bul.
5. Gerekli önceki sprint gerçekten uygulanmamışsa bunu somut dosya/test kanıtıyla bildir; varmış gibi ilerleme. Küçük bağımlılık kusurlarını mevcut sprint içinde düzelt ve kaydet; tüm eksik bir sprinti sessizce üstlenme.
6. Kısa bir uygulama planı çıkar; ardından doğrudan uygula. Rutin tasarım tercihleri ve geri alınabilir repo düzenlemeleri için tekrar izin isteme.

## Çalışma biçimi

- Değişiklikleri görev blokları halinde uygula ve doğrula; yalnız analiz yapıp durma.
- Uzun görevde bulguları ve tamamlanan adımları diskte kaydet. Bağlam daralırsa aynı işi bu kayıttan sürdür; baştan başlamama.
- Repo'nun mevcut sade mimarisini kullan. Genel plugin sistemi, dağıtık servis veya büyük framework eklemek için bu görev yetki değildir.
- Yeni bağımlılık eklerken nedenini belirt; üçüncü taraf API kullanımını kurulu sürüm ve gerektiğinde resmi dokümantasyondan doğrula.
- Otomatik alt agent veya yeni Codex task açmak bu promptların gereği değildir. Çalışmayı mevcut task'ta tamamla.
- Git commit/push, PR, release, deploy ve paket yayını yapma. İncelenebilir çalışma ağacı diff'i bırak; mevcut kullanıcının commit düzenini varsayma.
- Tarihsel `submission/results/` ham JSONL/CSV çıktıları ve kullanıcı Word dosyaları korunacak. Yeni raporlar ayrı sürümlenmiş yollar kullanacak; eski ölçüm sessizce yeni yöntemle etiketlenmeyecek.
- Bulut/Modal/GPU işi, ücretli API çağrısı ve büyük model/dataset indirmesi başlatma. Yerelde mevcut model ve küçük fixture'larla çalış. Paket araçları için gerekirse repo içinde izole ortam kullan; global Python kurulumunu değiştirme.
- Platform veya model eksikliği yalnız ilgili entegrasyon doğrulamasını sınırlar; bağımsız kod, fixture, test ve dokümantasyon işlerini tamamlamaya devam et.

## Doğrulama kuralları

- Kritik davranış kusurları için önce kusuru gösteren küçük regresyon senaryosu oluştur; düzeltme sonrası geçtiğini göster.
- Kaynak kodun aynısını testte tekrar yazma. Dış davranış, sınır, oracle ve hata sözleşmesini doğrula.
- Mock testlerini gerçek backend/platform testi gibi sunma. Fake embedder ile pipeline sözleşmesi doğrulanabilir; model kalitesi doğrulanamaz.
- İlgili testleri çalıştır, sonra değişikliğe uygun genel suite kontrolü yap. Yeni gerekçe yoksa aynı testleri tekrar tekrar çalıştırma.
- Audit zamanındaki 120 test bir referanstır; güncel test sayısını ve sonucu raporla. Coverage tek başına kabul kriteri değildir.
- “Skip”, “hata”, “önceden var olan hata” ve “başarılı” durumlarını ayır. Uzak CI veya erişilmeyen platform için başarı uydurma.
- `git diff --check` ve eklediğin yeni dosyalar üzerinde biçim/bağlantı kontrolü yap. Untracked dosyaların normal `git diff` çıktısında görünmediğini dikkate al.

## Tamamlanma ve devir

İlgili sprint için `docs/sprints/results/sprint-XX.md` oluştur. En az şu başlıkları içersin:

1. **Durum:** tamamlandı / kısmen tamamlandı / engelli. Implementasyon durumu ile dış doğrulama durumunu ayrı yaz.
2. **Başlangıç:** HEAD, ilgili mevcut değişiklikler, doğrulanan başlangıç kusurları.
3. **Uygulananlar:** davranış değişikliği, dosyalar, önemli tasarım kararı ve uyumluluk etkisi.
4. **Kabul matrisi:** prompttaki her kabul kriteri → test/komut/kanıt → geçti/kaldı/çalıştırılmadı.
5. **Doğrulama:** tam komutlar, ortam, sonuç sayıları; mock/gerçek backend ayrımı.
6. **Açık işler:** neden, etki, gerekli sonraki eylem. İlgili kabul kapısı geçmeden “tamamlandı” yazma.
7. **Sonraki sprint devri:** yeni API/config/metadata şeması, değişen CLI seçenekleri, örnek komut ve riskler.

`STATUS.md` tablosunu bu kanıtlarla güncelle. Sprintin kod işi bitti ama gerçek model smoke'u çalışmadıysa bunu açık yaz; hangi sonraki işlerin ilerleyebileceğini belirt. Yalnız dış doğrulama beklediği için tüm bağımsız çalışmayı bırakma.

Son cevap kısa olsun: tamamlanan davranışlar, testler, açık sınırlar ve sonuç raporu yolu. Kullanıcı sıradaki promptu vermeden bir sonraki sprinte başlama.
