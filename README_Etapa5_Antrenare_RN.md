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

## 4. Rezultate antrenare (exemplu)
> **ATENTIE:** Inlocuieste valorile de mai jos cu rezultatele obtinute local dupa rulare.

- Best validation accuracy: **0.92**  
- Training accuracy (last epoch): **0.95**  
- Validation accuracy (last epoch): **0.90**  
- Test accuracy: **0.89**

Grafic istoric antrenare: `docs/screenshots/training_history.png`  
Matrice de confuzie: `docs/screenshots/confusion_matrix.png`  
Raport clasificare (CSV): `docs/screenshots/classification_report.csv`

---

## 5. Metrici pe test set

Rulare script: `python src/training/train_model.py` (va produce fisierele de raport).

**Exemplu (inlocuieste cu valori reale):**

| Clasa   | Precision | Recall | F1-score | Support |
|---------|-----------|--------|----------|---------|
| plastic | 0.90      | 0.92   | 0.91     | 150     |
| hartie  | 0.88      | 0.85   | 0.86     | 140     |
| sticla  | 0.91      | 0.88   | 0.90     | 145     |
| metal   | 0.87      | 0.88   | 0.87     | 140     |
| **accuracy** |  |  | **0.89** | 575 |

---

## 6. Analiza erorilor

- **Confuzie plastic↔metal:** anumite obiecte metalice cu vopsea sau etichete reflectorizante pot parea plastic in imagini mici (64×64).  
- **Hartie vs plastic subexpusa:** obiectele foarte subexpuse pierd textura, facand distinctia dificila.  
- **Sugestii de remediere:** cresterea rezolutiei (ex: 128×128), augmentare pe iluminare/contrast, adaugarea de exemple greu clasificate, folosirea unei arhitecturi pre-antrenate (transfer learning) precum MobileNet/VGG16.

---

## 7. Cum rulezi (instructiuni rapide)


1) Instaleaza dependentele:

pip install -r requirements.txt


2) Ruleaza antrenarea modelului:

python src/training/train_model.py


3) Dupa antrenare, modelul salvat va aparea in:

models/trained_model.h5


4) Pentru a testa modelul pe setul de test:

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
