## Disciplina
Rețele Neuronale – FIIR, Universitatea POLITEHNICA București

## Student
Chelcea Rares-Gabriel  
Grupa: 634AB  
An: 2025
Data: 27.11.2025

---

ReteleNeuronaleProiect/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)

---

## 2. Descrierea Setului de Date
   
### 2.1 Sursa datelor

Origine: dataset public – „Garbage Classification Dataset” (Kaggle)
Modul de achiziție: ☑ Fișier extern
Perioada / condițiile colectării: Dataset disponibil public; imaginile au fost colectate în condiții variate, cu iluminare și rezoluții diferite.

### 2.2 Caracteristicile dataset-ului

Număr total de observații: aprox. 2.500 imagini
Număr de caracteristici (features): 1 (imagine RGB)
Tipuri de date: ☑ Imagini
Format fișiere: ☑ PNG / ☑ JPG

| **Caracteristică** | **Tip**     | **Unitate** | **Descriere**                                                               | **Domeniu valori**         |
| ------------------ | ----------- | ----------- | --------------------------------------------------------------------------- | -------------------------- |
| imagine            | imagine RGB | px          | Reprezintă obiectul din categoria deșeului (plastic, hârtie, metal, sticlă) | 64×64 px după preprocesare |

---

## 3. Analiza Exploratorie a Datelor (EDA) – Sintetic
   
### 3.1 Statistici descriptive aplicate

* **Dimensiuni variabile ale imaginilor
* **Distribuții pe clase (plastic, hartie, sticla, metal)
* **Identificarea imaginilor neclare sau cu rezoluții foarte mici

### 3.2 Analiza calității datelor

* **Detectarea imaginilor cu lumină puternică sau umbre
* **Detectarea imaginilor rotite sau necentrate
* **Identificarea claselor suprapuse (ex: paper și cardboard → hartie)

### 3.3 Probleme identificate

* **Imaginile au rezoluții foarte variate
* **Diferențe mari de iluminare
* **Clase brute redundante (paper + cardboard)
* **Necesitatea standardizării formatului și dimensiunii

---

## 4. Preprocesarea Datelor
   
### 4.1 Curățarea datelor

* **Nu există duplicate evidente
* **Nu există valori lipsă, fiind un dataset de imagini
* **Eliminarea clasei inutile (trash – nefolosită și neîncărcată)
* **Unificarea claselor similare (paper + cardboard → hartie)

### 4.2 Transformarea caracteristicilor

* **Normalizare:** valori pixel aduse în intervalul 0–255
* **Conversie: BGR → RGB
* **Redimensionare: 64×64 px
* **Ajustarea dezechilibrului de clasă: implicit prin împărțire stratificată

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:
* 70–80% – train
* 10–15% – validation
* 10–15% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în data/processed/
* Seturi train/, validation/, test/ generate automat prin script
* Parametrii de preprocesare pot fi salvați opțional în config/

---

## 5. Fișiere Generate în Această Etapă

* 'data/raw/' – date brute
* 'data/processed/' – date curățate & transformate
* 'data/train/', 'data/validation/', 'data/test/' – seturi finale
* 'src/preprocessing/' – codul de preprocesare:
* preprocess_raw_to_processed.py
* split_processed_into_train_val_test.py
* data/README.md – descrierea dataset-ului
