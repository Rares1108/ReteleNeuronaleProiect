## Disciplina
Rețele Neuronale – FIIR, Universitatea POLITEHNICA București

## Student
Chelcea Rares-Gabriel  
Grupa: 634AB  
Data: 04.12.2025

---

<img width="568" height="435" alt="directory_RN" src="https://github.com/user-attachments/assets/e11af051-c065-4121-9f3e-8057a3c452a1" />



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

## 6. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă**                       | **Cum o rezolvă SIA-ul vostru**                                                 | **Modul software responsabil**                         |
| ----------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Clasificarea automată a deșeurilor pe categorii | Rețea neuronală clasifică imaginea în una dintre cele 4 clase cu >85% acuratețe | Neural Network Module + Preprocessing + UI             |
| Reducerea timpului de procesare                 | Procesarea și clasificarea se fac în < 1 secundă pentru o imagine               | Data Acquisition + Neural Network Module + Web Service |
| Standardizarea setului de date                  | Preprocesare uniformă (64×64 px, normalizare RGB) pentru toate imagini          | Preprocessing Module                                   |



## 7. Contribuția originală la setul de date

Total observații finale: 2,500 imagini (după preprocesare și filtrare)
Observații originale: 1,000 imagini generate/achiziționate manual (40%)

Tipul contribuției:
☑ Date generate prin simulare fizică
[ ] Date achiziționate cu senzori proprii
[ ] Etichetare/adnotare manuală
[ ] Date sintetice prin metode avansate

# Descriere detaliată:

Pentru a atinge cerința de minimum 40% date originale, am generat suplimentar un set de 1,000 de imagini sintetice folosind un simulator 3D care modelează deșeuri în diferite condiții de iluminare și poziții variate. Am variat parametrii de iluminare, unghiurile de vizualizare și fundalul pentru a obține un set divers și reprezentativ. Aceasta contribuție permite o mai bună generalizare a modelului și evită dependența exclusivă de datele publice.

Locația codului: src/data_acquisition/simulate_garbage_images.py
Locația datelor: data/generated/

Dovezi:

* Grafic comparativ: docs/generated_vs_real.png
* Setup experimental (screenshot simulare): docs/acquisition_setup.jpg
* Tabel statistici: docs/data_statistics.csv

## 8. Diagrama State Machine a Întregului Sistem

 <img width="522" height="781" alt="schema_buna drawio" src="https://github.com/user-attachments/assets/d08b71d9-af2e-4829-81ee-4ac8bf87bf16" />



Justificarea State Machine-ului ales

Am ales arhitectura de tip clasificare la senzor, deoarece proiectul nostru are ca scop clasificarea automată a imaginilor de deșeuri. Sistemul funcționează în mod continuu, ciclul fiind: capturarea imaginii → verificarea calității → preprocesare → inferență RN → afișare rezultat → așteptare pentru următoarea captură.

Stările principale sunt:

* IDLE: sistemul așteaptă trigger pentru achiziția de date

* ACQUIRE_DATA: capturarea imaginii de la cameră

* CHECK_QUALITY: verificarea calității imaginii (blur, iluminare)

* PREPROCESS: redimensionare și normalizare imagine

* INFERENCE: rularea modelului RN pe datele preprocesate

* DISPLAY: afișarea rezultatului în UI

* ERROR: gestionarea imaginilor invalide sau erori de captură

Tranzițiile critice permit recuperarea automată, de exemplu la erori imagine se încearcă recapturarea maxim 3 ori. Astfel, sistemul este robust și funcționează continuu în condiții reale de iluminare și poziționare.

## 9. Scheletul Complet al celor 3 Module Cerute la Curs
Modul	Locatie Python	Cerință minimă funcțională
1. Data Logging / Acquisition	src/data_acquisition/	Generează CSV cu minimum 100 samples fără erori, include 40% date originale
2. Neural Network Module	src/neural_network/	Model RN definit, compilat, salvat și încărcat (weights random)
3. Web Service / UI	src/app/	UI simplă care primește input și afișează output
10. Instrucțiuni Lansare

* Pentru generarea datelor:
  
python src/data_acquisition/simulate_garbage_images.py

* Pentru preprocesare:
  
python src/preprocessing/preprocess_raw_to_processed.py
python src/preprocessing/split_processed_into_train_val_test.py

* Pentru rularea UI (exemplu Streamlit):
  
streamlit run src/app/app.py

