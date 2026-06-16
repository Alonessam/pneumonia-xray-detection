# Veri Seti Klasörü

Bu klasöre Kaggle'daki Chest X-Ray Images (Pneumonia) veri seti çıkarılmalıdır.

Beklenen yapı:

```text
data/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
```

Kaggle komutu:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data --unzip
```

Not: Kaggle'daki `val` klasörü çok küçüktür. Kod, `train` ve `val` klasörlerini eğitim havuzuna alıp bu havuzdan yeniden validation split üretir. Test klasörü final değerlendirme için ayrı tutulur.
