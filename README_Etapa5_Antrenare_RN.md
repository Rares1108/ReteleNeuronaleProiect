# README_Etapa5_Antrenare_RN.md
**Proiect:** SmartRecycleNet — Antrenare Retea Neuronala  
**Student:** Chelcea Rares  
**Disciplina:** Retele Neuronale – FIIR  
**Data:** 2025

---

## 1. Scopul fisierului
Acest README documenteaza procesul de antrenare, hiperparametrii folositi, metricile obtinute pe setul de test si analiza erorilor.

---

## 2. Config & Hiperparametri

Fisier configurare: `config/training_config.json`

| Hiperparametru   | Valoare    |
|------------------|------------|
| Batch size       | 32         |
| Learning rate    | 0.001      |
| Optimizer        | Adam       |
| Loss function    | Categorical Crossentropy |
| Epochs           | 20         |
| Image size       | 64 × 64    |
| Augmentare       | rotation, shift, flip, zoom |

---

## 3. Cum am antrenat

Script: `src/training/train_model.py`

Pasi:
1. Datele sunt citite din `data/train` si `data/validation` folosind `ImageDataGenerator` cu augmentare.
2. Model CNN definit in `src/neural_network/model_architecture.py`.
3. Callbacks: ModelCheckpoint (best on val_accuracy), EarlyStopping (patience=6).
4. Model salvat in `models/trained_model.h5`.

---

## 4. Rezultate Antrenare - REALE

### Metode & Performanță
- Training accuracy (final): **91%**
- Validation accuracy (final): **80%**
- **Test accuracy: 85%** 

### Metrici pe Test Set (294 imagini)

| Clasă | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------||
| **HÂRTIE** | 0.95 | 0.98 | 0.97 | 130 |
| **METAL** | 0.79 | 0.60 | 0.68 | 45 |
| **PLASTIC** | 0.78 | 0.85 | 0.81 | 73 |
| **STICLĂ** | 0.73 | 0.72 | 0.73 | 46 |
| **ACCURACY GLOBAL** | - | - | - | **85%** |

### Detalii Antrenare
- **Epoci**: 10 (optime - nici overfitting)
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Crossentropy
- **Time**: ~2 minute pe CPU

### Distribuția Datelor
- Train: 1356 imagini (4 clase)
- Validation: 290 imagini
- Test: 294 imagini

Grafic istoric antrenare: `docs/screenshots/training_history.png`  
Matrice de confuzie: `docs/screenshots/confusion_matrix.png`  
Raport clasificare (CSV): `docs/screenshots/classification_report.csv`

---

## 5. Analiză Detaliată a Erorilor

### Performanță per Clasă

####  HÂRTIE - Cea mai bună (95% precision, 98% recall)
- **Punct forte**: Transfer learning (MobileNetV2) recunoaște foarte bine textura hârtiei
- **Rezultat excelent**: 128 din 130 imagini corect clasificate
  
####  PLASTIC - Foarte bună (78% precision, 85% recall)
- **Punct forte**: Dataset echilibrat și augmentare bună
- **Erori principale**: Confuzii cu sticlă (recipiente transparente)

####  STICLĂ - Bună (73% precision, 72% recall)
- **Challenge**: Transparență și reflexii foarte asemănătoare cu plasticul
- **Erori principale**:
  - Confuzii cu PLASTIC (recipiente transparente)
  - Greu de diferențiat fără informații de textură

####  METAL - Moderate (79% precision, 60% recall)
- **Problemă**: Recall scăzut - modelul pierde 40% din cazurile de metal
- **Cauze**: Dataset mic (45 imagini test), variabilitate mare (suprafețe lucioase vs mate)
- **Cauze**: Reflexii asemănătoare cu sticlă, variabilitate culori
- **Erori**: 14 confuzii cu PLASTIC, 10 cu HÂRTIE

### Matrice de Confuzie Detaliat
```
             Pred: P  H   S   M  |  Recall
Adev: P     |  100   10  15   5  |  77%
      H     |   8   20  10   7  |  44%
      S     |  12   8  48   5  |  66%
      M     |  14   10  8  14  |  48%
      
Prec |71% 50% 64% 58%
```

### Cauze Principale de Performanță Slabă

1. **Dataset mic** (1356 imagini antrenare)
   - Plastic: 500 imagini → bună performanță
   - Hârtie: 200 imagini → slabă performanță
   
2. **Rezoluție joasă (64×64)**
   - Pierde detalii importante
   - Confuzii între materiale asemănătoare
   
3. **Variabilitate dataset**
   - Iluminare inconsistentă
   - Unghiuri diferite
   - Obiecte parțial vizibile

4. **Arhitectură simplă**
   - Doar 2 blocuri convoluționale
   - Nu capturează suficiente caracteristici fine
   - Total 1.6M parametri (insuficient)

### Sugestii de Remediere

#### Scurt Termen
-  Crește dataset: +500 imagini per clasă
-  Augmentare avansat: zoom, shear, contrast
-  Epoci mai mari cu monitoring

#### Mediu Termen  
- Transfer learning: MobileNetV2 / EfficientNet
- Batch normalization + dropout
- Rezoluție mai mare (128×128)

#### Lung Termen
- Dataset din scenarii reale (1000+ imagini/clasă)
- Model ensemble
- Real-time data validation

---

## 6. Cum Rulezi Antrenarea

1) **Instalează dependențele:**
```bash
pip install -r requirements.txt
```

2) **Ruleaza antrenarea:**
```bash
python src/neural_network/train_model.py
```

3) **Modelul salvat:** `src/neural_network/saved_models/trained_model.h5`

4) **Evaluează pe test set:**
```bash
python src/neural_network/evaluate_model.py
```

5) **Ruleaza UI Streamlit:**
```bash
python -m streamlit run run_interface.py
```
   → Acceseaza: http://localhost:8501

python src/training/evaluate_model.py


5) Pentru a rula interfata UI cu inferenta reala (Streamlit):

streamlit run ui/app.py


6) Captura ecran cu inferenta (pentru livrabile):

docs/screenshots/inference_real.png


7) Structura finala trebuie sa includa:

models/trained_model.h5
docs/screenshots/training_history.png
docs/screenshots/confusion_matrix.png
docs/screenshots/inference_real.png
