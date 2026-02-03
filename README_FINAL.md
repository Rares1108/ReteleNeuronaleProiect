#  RETELE NEURONALE - PROIECT COMPLET

## Student
**Chelcea Rares-Gabriel** | Grupa 634AB | FIIR, UPB

---

##  STATUS FINAL - ETAPA 5: ANTRENARE MODEL

###  Cerințe Completate

**1. Model Antrenat și Optimizat**
- **Accuracy:** 65% pe test set (294 imagini)
- **Clase:** Plastic, Hârtie, Sticlă, Metal (4 clase)
- **Salvat:** `models/trained_model.h5` (6.2 MB)
- **Arhitectură:** CNN simplu - 2 blocuri convoluționale, 1.6M parametri
- **Optim prin:** Testare multi-epoch (10 epochs = best, evită overfitting)

**2. UI Funcțional cu Inferență Reală**
- **Scriptul:** `run_interface.py` (Streamlit)
- **Funcționalități:** 
  - Upload imagini
  - Predicții în timp real (~0.5s per image)
  - Confidence scores pentru toate 4 clase
- **Acces:** http://localhost:8501

**3. Raport Detaliat Completat**
- **Fișier:** `README_Etapa5_Antrenare_RN.md`
- **Secțiuni:** Hiperparametri, Metrici, Matrice Confuzie, Analiză Erorilor

---

##  Metrici Model Final

| Clasă | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **PLASTIC** | 71% | 77% | 0.74 | 130 |
| **HÂRTIE** | 50% | 44% | 0.47 | 45 |
| **STICLĂ** | 64% | 66% | 0.65 | 73 |
| **METAL** | 58% | 48% | 0.52 | 46 |
| **ACCURACY GLOBAL** | — | — | — | **65%** |

### Analiza per Clasă
- **Best:** PLASTIC (77% recall) - model identifies plastic well
- **Worst:** HÂRTIE (44% recall) - often confused with PLASTIC (13% misclassification)
- **Balanced:** STICLĂ (66% recall), METAL (48% recall)

---

##  Hiperparametri Antrenare

| Parametru | Valoare |
|-----------|---------|
| **Epochs** | 10 |
| **Batch Size** | 32 |
| **Learning Rate** | 0.001 (Adam optimizer) |
| **Input Size** | 64×64 px RGB |
| **Data Augmentation** | Rotation ±15°, Shift ±10%, Horizontal Flip |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Cross-Entropy |
| **Validation Split** | 17.6% (290 imagini) |

---

##  Structură Finală Project

```
ReteleNeuronaleProiect-main/
│
├── models/
│   └── trained_model.h5                 MODEL ANTRENAT
│
├── src/neural_network/
│   ├── model_definition.py             - CNN Architecture
│   ├── train_model.py                  - Training Script
│   ├── evaluate_model.py               - Test Evaluation
│   └── saved_models/
│       └── trained_model.h5            - Backup Model
│
├── data/
│   ├── train/                          - 1356 imagini (4 clase)
│   ├── validation/                     - 290 imagini
│   └── test/                           - 294 imagini
│
├── docs/
│   ├── screenshots/
│   │   ├── training_history.png
│   │   ├── confusion_matrix.png
│   │   └── inference_demo.png
│   └── README.md
│
├── run_interface.py                     STREAMLIT UI
├── README_Etapa5_Antrenare_RN.md        RAPORT DETALIAT
├── README.md                           - Main Documentation
├── README_FINAL.md                     - This File
└── requirements.txt                    - Python Dependencies
```

---

##  Cum Rulezi Proiectul

### 1. Activare Mediu Virtual
```bash
cd ReteleNeuronaleProiect-main
.\venv\Scripts\Activate
```

### 2. Lansare UI cu Inferență Reală
```bash
streamlit run run_interface.py
```
 Acces: http://localhost:8501
- Upload o imagine de deșeu
- Modelul returnează predicția și confidence scores

### 3. (Optional) Reevaluare Model pe Test Set
```bash
python src/neural_network/evaluate_model.py
```
Afișează metrici detaliate, confusion matrix, și raport de clasificare.

### 4. (Optional) Reantrenare Model (necesită date)
```bash
python src/neural_network/train_model.py
```

---

##  Resurse Principale

| Resursă | Locație | Descriere |
|---------|---------|-----------|
| **Model Antrenat** | `models/trained_model.h5` | Format H5 Keras, 6.2 MB |
| **Raport Etapa 5** | `README_Etapa5_Antrenare_RN.md` | Detalii complete antrenare |
| **UI Aplicație** | `run_interface.py` | Streamlit inference interface |
| **Script Training** | `src/neural_network/train_model.py` | Antrenare model (10 epochs) |
| **Script Evaluare** | `src/neural_network/evaluate_model.py` | Evaluare pe test set |
| **Dataset** | `data/train/`, `data/validation/`, `data/test/` | 1940 imagini totale |

---

##  Analiza Erorilor Principale

### Top Confuzii (pe Test Set)
1. **HÂRTIE → PLASTIC** (13 imagini) - Ambele au culori ușoare
2. **METAL → STICLĂ** (6 imagini) - Reflectanță similară
3. **PLASTIC → HÂRTIE** (5 imagini) - Overlap de texturi

### Recomandări Remediere
- **Creștere dataset:** Mai multe imagini per clasă (min 500+)
- **Model complex:** VGG16 / ResNet50 cu transfer learning
- **Augmentation avansat:** Mixup, CutMix, augmentări mai aggressive
- **Preprocessing:** Detectare și focus pe obiect (masking)

---

##  Dependențe

```
TensorFlow==2.20.0
Keras==3.7.0
opencv-python-headless==4.10.1.26
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
Streamlit==1.52.2
```

Instale cu: `pip install -r requirements.txt`

---

##  Istoric Experimentare

| Experiment | Epochs | Accuracy | Notes |
|------------|--------|----------|-------|
| Complex CNN | 25 | 56% | Overfitting |
| Complex CNN | 30 | 59% | Worse validation |
| Simple CNN | **10** | **65%** |  OPTIMAL |

**Concluzie:** Model simplu cu 10 epochs > modele complexe (avoid overfitting pe dataset mic)

---

##  Checklist Final

- [x] Model antrenat și salvat (`models/trained_model.h5`)
- [x] Accuracy: 65% pe test set (191/294 correct)
- [x] 4 clase clasificate: Plastic, Hârtie, Sticlă, Metal
- [x] UI Streamlit funcțional cu inferență reală
- [x] Raport detaliat Etapa 5 completat
- [x] Metrici per clasă (Precision, Recall, F1)
- [x] Matrice confuzie și analiza erorilor
- [x] Recomandări pentru îmbunătățiri
- [x] Documentație completă

---

##  Notes

- **Training Time:** ~2 min per epoch (CPU)
- **Inference Time:** ~0.5s per image (CPU)
- **Model Size:** 6.2 MB (H5 format)
- **Data Total:** 1940 imagini (70% train, 15% val, 15% test)

---

**Status:**  **COMPLET ȘI FUNCȚIONAL**  
**Data Finalizare:** 3 Ianuarie 2026  
**Version:** 1.0  
**Student:** Chelcea Rares-Gabriel  
**Universitate:** POLITEHNICA București - FIIR  
**Disciplina:** Rețele Neuronale
