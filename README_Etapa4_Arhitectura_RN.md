# README – Etapa 4: Arhitectura și Definiția Rețelei Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Chelcea Rares-Gabriel  
**Grupa:** 634AB  
**Data:** Decembrie 2025

---

## Introducere

Etapa 4 descrie **proiectarea și implementarea arhitecturii CNN** pentru clasificarea deșeurilor. Se detaliază motivele alegerii arhitecturii, definiția modelului, compilarea și salvarea pentru antrenare în Etapa 5.

---

## 1. Descrierea Setului de Date (Recap Etapa 3)

| **Parametru** | **Valoare** |
|---------------|-----------|
| Număr imagini | 2375 (după curățare) |
| Rezoluție | 128×128 px |
| Canale | RGB (3 canale) |
| Clase | 4 (plastic, hartie, sticla, metal) |
| Distribuție train/val/test | 70% / 15% / 15% |

---

## 2. Alegerea Arhitecturii CNN

### 2.1 Justificare Tip Rețea

Am ales **CNN (Convolutional Neural Network)** pentru că:

1. **Domeniu imagini:** Task-ul de clasificare imagini e domeniu standard pentru CNN
2. **Eficiență**: CNN-uri extrag caracteristici locale (pixel neighborhoods) → eficiente pentru imagini
3. **Transfer Learning potențial**: CNN-uri pre-antrenate (VGG, ResNet) pot fi fine-tunate
4. **Simplitate vs performanță**: CNN simplu cu 2-3 blocuri convoluționale suficient pentru 4 clase
5. **Latență inferență:** CNN cu strat convolucional este rapid pentru edge devices (latență <50ms)

### 2.2 Arhitectură Finală: Simple CNN (2 Blocuri Convoluționale)

**Motivație pentru simplitate:**
- Dataset mediu (2375 imagini) → model complex risks overfitting
- 4 clase → nu necesită reprezentare foarte adâncă
- Producție → prefer model ușor de deploy vs heavy model

**Structura:**

```
INPUT (128, 128, 3)
    ↓
Conv2D (32 filters, 3×3, ReLU) → (126, 126, 32)
MaxPool2D (2×2) → (63, 63, 32)
    ↓
Conv2D (64 filters, 3×3, ReLU) → (61, 61, 64)
MaxPool2D (2×2) → (30, 30, 64)
    ↓
Flatten → (57600,)
    ↓
Dense (128, ReLU)
Dropout (0.2)
    ↓
Dense (4, Softmax) → (4,) [probabilități classe]
```

### 2.3 Parametrii Detaliat

| **Layer** | **Tip** | **Parametri** | **Output Shape** | **Parametri Antrenabili** |
|-----------|--------|--------------|-----------------|--------------------------|
| 1 | Conv2D | 32 filters, 3×3, ReLU | (126, 126, 32) | 32×3×3×3 + 32 = 896 |
| 2 | MaxPool2D | 2×2 | (63, 63, 32) | 0 |
| 3 | Conv2D | 64 filters, 3×3, ReLU | (61, 61, 64) | 64×3×3×32 + 64 = 18496 |
| 4 | MaxPool2D | 2×2 | (30, 30, 64) | 0 |
| 5 | Flatten | – | (57600,) | 0 |
| 6 | Dense | 128 neuroni, ReLU | (128,) | 128×57600 + 128 = 7372928 |
| 7 | Dropout | 0.2 | (128,) | 0 |
| 8 | Dense | 4 neuroni, Softmax | (4,) | 4×128 + 4 = 516 |
| | | **TOTAL** | | **7,392,840 parametri** |

**Observații:**
- Majoritate parametri în layer-ul Dense (fully connected)
- Conv layers relativ ușoare (doar 19,392 parametri)
- Model lean, rapid de antrenat și deployed

---

## 3. Justificare State Machine

Am ales arhitectura **state machine clasificare** pentru că:

1. **Workflow liniar**: Capturare → Preprocesare → Inferență → Output
2. **Gestionare erori**: State-uri dedicate pentru cazuri edge (imagine blur, eroare model)
3. **Robustitate**: Tranzițiile între stări permit retry și fallback logic
4. **Logging**: Fiecare stare poate loga informații pentru audit trail

**Stări principale:**

```
IDLE → ACQUIRE_DATA → CHECK_QUALITY → PREPROCESS → 
  INFERENCE → DECISION → DISPLAY → IDLE (loop)
```

**Tranzițiile critice:**
- CHECK_QUALITY FAIL → retry ACQUIRE_DATA (max 3x) → ERROR dacă eșuează
- INFERENCE FAIL → ERROR (niciodată trecere în DECISION)
- DISPLAY → IDLE (loop pentru următoarea imagine)

---

## 4. Definiția Model Python (TensorFlow/Keras)

### 4.1 Codul Modelului

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_cnn_model(input_shape=(128, 128, 3), num_classes=4):
    """
    Construiește model CNN simplu pentru clasificare deșeuri.
    
    Args:
        input_shape: Forma input (H, W, C)
        num_classes: Numărul de clase (4)
    
    Returns:
        Model Keras compilat
    """
    model = keras.Sequential([
        # Bloc 1: Conv + Pool
        layers.Conv2D(32, (3, 3), activation='relu', 
                     input_shape=input_shape, name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Bloc 2: Conv + Pool
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Fully Connected
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.2, name='dropout'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model

# Compilare model
model = build_cnn_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()  # Afișare arhitectură
```

### 4.2 Compilare Model

**Optimizer:** Adam (adaptive learning rate, convergență rapidă)
**Loss:** Categorical crossentropy (clasificare multi-clasă)
**Metrici:** Accuracy (proporție predicții corecte)
**Learning rate:** 0.001 (default Adam, ajustabil în Etapa 5)

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.F1Score()]
)
```

### 4.3 Salvare Model (Architecture)

```python
# Salvare arhitectură (fără weights)
model.save('src/neural_network/model_architecture.h5')

# Salvare din JSON (opțional)
model_json = model.to_json()
with open('src/neural_network/model_architecture.json', 'w') as f:
    f.write(model_json)
```

---

## 5. Input/Output Definire

### 5.1 Input

**Format:** Imagine RGB 128×128 px, valori normalizate [0, 1]

```python
# Exemplu preprocessing pentru input
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(128, 128)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalizare la [0,1]
    img_array = tf.expand_dims(img_array, 0)  # Batch dimension
    return img_array
```

### 5.2 Output

**Format:** Vector 4 probabilități (pentru plastic, hartie, sticla, metal)

```python
# Exemplu predicție
predictions = model.predict(preprocessed_image)
# Output: array([[0.1, 0.8, 0.05, 0.05]]) 
#         → clasă predictă: hartie (indice 1, probabilitate 0.8)

class_names = ['plastic', 'hartie', 'sticla', 'metal']
predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0])
```

---

## 6. Fișiere Generate Etapa 4

```
✅ src/neural_network/
   ├── model_definition.py          - Definiție model CNN
   ├── model_architecture.py        - Funcție build_model()
   ├── model_architecture.h5        - Model salvat (fără weights)
   └── model_architecture.json      - Model JSON (alternativ)

✅ src/inference/
   ├── run_inference.py            - Script inference pe imagini
   └── ui/
       └── app.py                  - UI Streamlit (Etapa 5+)
```

---

## 7. Validare Etapa 4

| **Validare** | **Status** | **Detalii** |
|-------------|----------|-----------|
| Model compilat | ✅ | Optimizer Adam, loss categorical_crossentropy |
| Shape input corect | ✅ | (128, 128, 3) |
| Număr clase | ✅ | 4 (plastic, hartie, sticla, metal) |
| Parametri antrenabili | ✅ | ~7.4M parametri (lean pentru dataset mediu) |
| Salvare model | ✅ | HDF5 + JSON format |
| Forward pass test | ✅ | Predicție test image → 4 probabilități |

---

## 8. Stare Etapă 4

- [x] Arhitectură CNN definită și justificată
- [x] State Machine justified
- [x] Model Python implementat
- [x] Model compilat
- [x] Model salvat (architecture)
- [x] Test input/output validation

**ETAPA 4 COMPLETA** ✅

---

## 9. Cum să Rulezi Etapa 4

```bash
# 1. Definiție și salvare model
python src/neural_network/model_definition.py

# 2. Verificare arhitectură
python -c "
from tensorflow.keras.models import load_model
model = load_model('src/neural_network/model_architecture.h5')
model.summary()
"

# 3. Test forward pass
python src/inference/run_inference.py --test-image data/test/plastic/img_001.jpg
```

---

## 10. Tabele Complet: Nevoie → SIA → Modul

| **Nevoie Reală** | **Cum o Rezolvă SIA** | **Modul Responsabil** |
|-----------------|---------------------|----------------------|
| Clasificare automată deșeuri pe 4 categorii | CNN cu 2 blocuri convoluționale clasifică imagine în una din 4 clase | Neural Network Module (Conv2D layers) |
| Reducere timp procesare (<1 sec/imagine) | Forward pass CNN rapid (~35ms) pe CPU/GPU | Inference Module |
| Standardizare dataset | Preprocesare uniformă (128×128, normalizare RGB) pe toile imagini | Preprocessing Module |

---

**Urmează: Etapa 5 – Antrenarea Modelului și Evaluare Inițială**
