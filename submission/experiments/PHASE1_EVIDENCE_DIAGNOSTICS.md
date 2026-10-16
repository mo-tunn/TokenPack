# Faz 1: Evidence failure audit (2026-10-11)

Bu Faz 1, **TokenPack secicisini ve mevcut deney protokolunu degistirmez**.
Yalnizca tamamlanmis, yerel Qwen2.5-14B-Instruct cevaplarini okur.
Modal GPU veya harici inference API cagrisi **yapmaz**.

## Calistirma

TokenPack proje kokunde (Windows):

    C:\Python313\python.exe submission\experiments\evidence_failure_audit.py

Birim testleri:

    C:\Python313\python.exe -m unittest discover -s tests -p test_evidence_failure_audit.py -v

Yeni raporlarin klasoru:

    submission/results/phase1_evidence_diagnostics_20261011/

Uretilen raporlar:

- phase1_readout.md — ozet ve ayirt edici ornekler
- qasper_cases.csv ve qasper_summary.json — 50 soruluk eslesmis analiz
- qasper_manual_review.md — 5 kayip icin insan inceleme listesi
- qasper_ablation_cases.csv ve qasper_ablation_summary.json — ayni baglam, farkli istem ablasyonu
- longbench_cases.csv ve longbench_summary.json — 83 kisa soru
- longbench_64k_cases.csv ve longbench_64k_summary.json — 30 uzun baglamli soru

## Diagnostik tasarimi

QASPER:
- Ayni 50 vakada TokenPack-50 ve budget-top-k-50 sonuclarini case_id ile eslestirir.
- Secilen chunk ID kumesini, sirasini ve tam baglam esitligini karsilastirir.
- Gold evidence ifadelerinin yaklasik kelime-kapsamini ve bitisik besli
  kelime dizisi (5-gram) kapsamini olcer.
- TokenPack kayiplarini kanit eksigi, kanit korunmusken yanit eksigi,
  muhtemel siralama sorunu ve belirsiz gibi **aday inceleme**
  kategorilerine ayirir.
- Ayni baglam + ayni istem ile ayni baglam + farkli istem sonuclarini ayirir.
- Gold evidence yalnizca offline incelemede kullanilir.

LongBench:
- 83 kisa soru ve 30 orta/uzun baglamli soruyu ayri raporlar.
- Eslesmis soru kazanma/kaybetme/beraberlik sayilarini cikarir.
- Gold kanit araliklari olmadigi icin hata nedeni atamaz.

## Metodolojik sinirlar

- Kelime ve 5-gram ortusmesi semantik yeterlilik veya nedensellik kaniti degildir.
- QASPER hakemi cevap ureten ayni Qwen2.5-14B'dir; bagimsiz hakem degildir.
- Ayni baglamda farkli istemlerin sonucu degistirmesi, secici iyilesmesi degildir.
- Daha once gelistirmede kullanilan sorular ustunde kazanilan sonuclar
  yeni, ayrilmis bir hold-out kumesinde tekrar test edilmelidir.

## Faz 2'ye gecis icin

1. qasper_manual_review.md icindeki 5 hatayi kaynak ifadelerle elle dogrula.
2. Kanit eksigini yanit cikarma ve LLM-hakem kararsizligindan ayir.
3. Secilen chunklarin komsuluklarini, siralarini ve belge kimliklerini incele.
4. Yeni seciciyi eski seciciden bagimsiz, ayri bir deney kolunda gelistir.
5. Mevcut TokenPack-50/RAG-50 kayitlarini veya Qwen 27B pilot promptlarini
   yerinde degistirme.

Yeni kod: submission/experiments/evidence_failure_audit.py
Yeni testler: tests/test_evidence_failure_audit.py
Production modulleri ve Modal uygulamalari degistirilmedi.
