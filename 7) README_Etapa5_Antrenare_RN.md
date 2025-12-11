# README_Etapa5_Antrenare_RN.md
**Proiect:** SmartRecycleNet — Antrenare Rețea Neuronală  
**Student:** Chelcea Rares
**Disciplina:** Retele Neuronale – FIIR  
**Data:** 2025

---

## 1. Scopul fișierului
Acest README documenteaza procesul de antrenare, hiperparametrii folositi, metricile obtinute pe setul de test și analiza erorilor.

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
1. Datele sunt citite din `data/train` și `data/validation` folosind `ImageDataGenerator` cu augmentare.
2. Model CNN definit în `src/neural_network/model_architecture.py`.
3. Callbacks: ModelCheckpoint (best on val_accuracy), EarlyStopping (patience=6).
4. Model salvat în `models/trained_model.h5`.

---

## 4. Rezultate antrenare (exemplu)
> **ATENȚIE:** Înlocuiește valorile de mai jos cu rezultatele obținute local după rulare.

- Best validation accuracy: **0.92**  
- Training accuracy (last epoch): **0.95**  
- Validation accuracy (last epoch): **0.90**  
- Test accuracy: **0.89**

Grafic istoric antrenare: `docs/screenshots/training_history.png`  
Matrice de confuzie: `docs/screenshots/confusion_matrix.png`  
Raport clasificare (CSV): `docs/screenshots/classification_report.csv`

---

## 5. Metrici pe test set

Rulare script: `python src/training/train_model.py` (va produce fișierele de raport).

**Exemplu (înlocuiește cu valori reale):**

| Clasă   | Precision | Recall | F1-score | Support |
|---------|-----------|--------|----------|---------|
| plastic | 0.90      | 0.92   | 0.91     | 150     |
| hartie  | 0.88      | 0.85   | 0.86     | 140     |
| sticla  | 0.91      | 0.88   | 0.90     | 145     |
| metal   | 0.87      | 0.88   | 0.87     | 140     |
| **accuracy** |  |  | **0.89** | 575 |

---

## 6. Analiza erorilor

- **Confuzie plastic↔metal:** anumite obiecte metalice cu vopsea sau etichete reflectorizante pot parea plastic în imagini mici (64×64).  
- **Hartie vs plastic subexpusă:** obiectele foarte subexpuse pierd textură, făcând distincția dificilă.  
- **Sugestii de remediere:** cresterea rezolutiei (ex: 128×128), augmentare pe iluminare/contrast, adaugarea de exemple greu clasificate, folosirea unei arhitecturi pre-antrenate (transfer learning) precum MobileNet/VGG16.

---

## 7. Cum rulezi (instrucțiuni rapide)

1. Instalează dependențe:
```bash
pip install -r requirements.txt
