# DASS-DL Classification - 5 Dakikalık Başlangıç

## 🚀 Hızlı Başlat (Copy-Paste)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Example
```bash
python example_usage.py
```

**That's it!** ✅ 2-3 dakika sonra sonuçlar alacaksın.

---

## 📊 Output

Script tamamlandığında 3 dosya oluşacak:

1. **training_history.png** - Eğitim grafiği (Loss, Accuracy, Macro F1)
2. **confusion_matrices.png** - Sınıflandırma matrisleri
3. **dass_model_focal_loss.pt** - Eğitilmiş model

---

## 🎯 Kendi Verin ile Çalıştır

### Step 1: CSV'ni Hazırla

`your_data.csv` dosyası şu kolonları içermeli:
- `Q1A`, `Q2A`, ..., `Q42A` (DASS answers, 1-4 scale)
- `Q1E`, `Q2E`, ..., `Q42E` (response times)
- `Depression_Status_Normal`, `Depression_Status_Mild`, `Depression_Status_Moderate`, `Depression_Status_Severe`, `Depression_Status_Extremely Severe`
- `Anxiety_Status_*` (aynı format)
- `Stress_Status_*` (aynı format)
- Demographics (demographics columns)

### Step 2: example_usage.py Düzenle

```python
# Satır 50 civarında, bunu değiştir:

# df = create_synthetic_dass_data(n_samples=1000)  ← Bunu sil
df = pd.read_csv('your_data.csv')  ← Bunu ekle

# Sonra normal şekilde çalıştır:
# python example_usage.py
```

### Step 3: Çalıştır!
```bash
python example_usage.py
```

---

## 🔧 Extreme Imbalance İçin Tuning

Religion (378:1) veya Race (1197:1) gibi extreme imbalance'ı iyileştirmek için:

```python
# example_usage.py satır 20'de:
GAMMA = 2.0

# Bunu yap:
GAMMA = 3.5  ← Extreme imbalance için
```

Sonra yeniden çalıştır!

---

## 📈 Sonuçları Okuma

Script tamamlandığında şuna benzer bir output göreceksin:

```
EVALUATION RESULTS
======================================================================

DEPRESSION
──────────────────────────────────────────────────────────────────────────

Overall Metrics:
  Accuracy (NOT recommended):  0.6234
  Macro F1 (RECOMMENDED):      0.5987  ← BU'NA BAK!
  Weighted F1:                 0.6145

Per-Class Metrics:
Class                Precision       Recall           F1        Support
────────────────────────────────────────────────────────────────────────
Normal                   0.7234      0.8123      0.7647        2345
Mild                     0.6123      0.5678      0.5897        1234
Moderate                 0.4234      0.3456      0.3821         567
Severe                   0.2345      0.1234      0.1620         234  ← Gelişti!
Extremely Severe         0.1123      0.0567      0.0754          89  ← Şimdi yakalanıyor!
```

**Önemli:** Accuracy değil, **Macro F1**'e bak! (imbalanced data'da Accuracy yanıltıcı)

---

## ❓ Sorun Giderme

### Memory Error
```python
# example_usage.py satır 25'de:
BATCH_SIZE = 32

# Bunu yap:
BATCH_SIZE = 16  ← Küçültür
```

### Model Gelişmiyor
```python
# example_usage.py satır 22'de:
GAMMA = 2.0

# Bunu yap:
GAMMA = 3.5  ← Focal loss'u güçlendir
```

### Overfitting
```python
# dass_multilabel_classifier.py satır 150'de:
dropout_rates=[0.4, 0.3, 0.2]

# Bunu yap:
dropout_rates=[0.5, 0.4, 0.3]  ← Dropout artır
```

---

## 📚 Detaylı Bilgi İçin

- **README.md** - Full documentation
- **GETTING_STARTED.md** - Türkçe detaylı rehber (coming soon)
- **dass_multilabel_classifier.py** - Kod comments'leri oku

---

**That's it! Hızlı ve kolay 🚀**
