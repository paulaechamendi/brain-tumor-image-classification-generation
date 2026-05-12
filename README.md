# IMAGE Project: Brain Tumor Image Classification & Generation
# AUTHORS: Paula Echamendi & Carmen Miralles 

Proyecto de análisis completo de imágenes médicas de resonancia magnética (MRI) cerebral,
abarcando desde el análisis exploratorio hasta la generación sintética de imágenes mediante
redes generativas adversariales.

Autoras: Carmen Miralles y Paula Echamendi

---

## Estructura del proyecto

```
.
├── data/                          # Dataset Brain Tumor MRI (train / val / test)
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       └── ...
│
├── features/                      # Features HOG extraídas para ML clásico (NB2)
├── modelos/                       # Checkpoints de los modelos entrenados (NB3)
│   ├── cnn_scratch_best.pth
│   ├── resnet50_best.pth
│   └── efficientnetb0_best.pth
├── gan_models/                    # Generadores y discriminadores entrenados (NB5)
│   ├── generator_glioma.pth
│   ├── generator_meningioma.pth
│   ├── generator_notumor.pth
│   └── generator_pituitary.pth
│
├── 1. EDA tumores cerebrales.ipynb
├── 2. Feature extraction & ML.ipynb
├── 3. DL Clasificacion.ipynb
├── 4. Object Detection.ipynb
└── 5. Generative.ipynb
```

> **Las carpetas `data/`, `features/`, `modelos/` y `gan_models/` no están subidas al repositorio** porque el tamaño de los archivos supera el límite permitido por GitHub. Están disponibles en Google Drive.

---

## Notebooks

### 1. EDA — Análisis Exploratorio
Análisis completo del dataset antes de entrenar ningún modelo.

- Distribución de clases y balanceo del dataset
- Estadísticas de tamaño de imagen (alto, ancho, canales)
- Histogramas de color y distribución de píxeles por clase
- Medias y desviaciones estándar (μ, σ) por clase
- Imágenes representativas y casos extremos
- **Conclusión clave:** glioma (μ=32.5) y meningioma (μ=44.8) presentan distribuciones
  de píxel muy similares, anticipando que serán las clases más difíciles de separar.

---

### 2. Feature Extraction & ML Clásico
Baseline con machine learning tradicional usando features extraídas con técnicas clásicas y Deep Learning.

- Extracción de features HOG (*Histogram of Oriented Gradients*)
- Extracción de features con backbone ResNet50 pre-entrenado (sin fine-tuning)
- Clasificadores evaluados: SVM, Random Forest, KNN, Logistic Regression
- **Mejor resultado:** HOG + SVM → **90.5% accuracy** en test
- Este baseline sirve como referencia mínima que los modelos de Deep Learning deben superar.

---

### 3. DL Clasificación
Comparativa de tres arquitecturas de clasificación de imágenes con Deep Learning.

| Modelo | Parámetros | Estrategia | Test Accuracy |
|---|---|---|---|
| CNN from scratch | ~2M | Entrenamiento completo (25 epochs) | — |
| ResNet50 | ~25M | Fine-tuning en 2 fases (20 epochs) | — |
| EfficientNetB0 | ~5.3M | Fine-tuning en 2 fases (20 epochs) | — |

- Fine-tuning en dos fases: backbone congelado → descongelado con LR diferenciado (1e-3 fc / 1e-5 backbone)
- Data augmentation, Dropout(0.5), scheduler ReduceLROnPlateau
- Curvas de accuracy/loss y análisis de overfitting para cada modelo
- **Modelo seleccionado: ResNet50** — mayor test accuracy y F1 macro, utilizado en NB4 y NB5.

---

### 4. Object Detection — GradCAM
Localización de tumores sin bounding boxes mediante *Gradient-weighted Class Activation Mapping*.

**¿Por qué GradCAM?** El dataset no incluye anotaciones de posición (bounding boxes ni máscaras).
Etiquetar manualmente la localización del tumor en más de 7 000 imágenes requeriría personal
médico especializado. GradCAM reutiliza el modelo de clasificación entrenado (ResNet50) y
genera mapas de calor a partir de sus gradientes internos, sin necesitar ninguna anotación
adicional — detección débilmente supervisada (*weakly supervised*) a coste cero de etiquetado.

- Mapas de calor GradCAM superpuestos sobre las MRI, por clase
- Análisis de aciertos vs errores: qué regiones activa el modelo en cada caso
- **Resultado:** el modelo localiza correctamente la zona anatómica relevante
  (masa infiltrante en glioma, borde cortical en meningioma, región selar en pituitary)
  confirmando que ha aprendido features con valor diagnóstico real.

---

### 5. Generative — DCGAN
Generación sintética de imágenes MRI por clase mediante *Deep Convolutional GAN*.

- Una DCGAN entrenada por cada clase (glioma, meningioma, notumor, pituitary)
- Curvas Loss_G / Loss_D y seguimiento de D(real) / D(fake) durante el entrenamiento
- Evolución visual del generador epoch a epoch
- Comparativa real vs generado por clase
- **Indicador de éxito:** convergencia D(fake) → 0.5 (equilibrio de Nash)
- Las imágenes generadas reproducen la estructura global y distribución de brillo por clase
  (limitación de detalle fino por resolución 64×64 del DCGAN estándar).

---

## Dataset

**Brain Tumor MRI Dataset** — [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Clase | Descripción |
|---|---|
| `glioma` | Tumor maligno originado en las células gliales del sistema nervioso central |
| `meningioma` | Tumor generalmente benigno originado en las meninges (membranas que rodean el cerebro) |
| `pituitary` | Tumor en la glándula pituitaria (hipófisis), en la base del cráneo |
| `notumor` | MRI sin tumor aparente |

---

## Requisitos

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn scikit-learn
pip install opencv-python pillow tqdm
```

> Entrenado con PyTorch. Se recomienda GPU (Google Colab T4 o superior).
> Los checkpoints guardados en `modelos/` y `gan_models/` permiten cargar los modelos
> directamente sin necesidad de re-entrenar.

---

## Orden de ejecución

Los notebooks están diseñados para ejecutarse en orden secuencial:

```
NB1 (EDA) → NB2 (ML baseline) → NB3 (DL, genera modelos/) → NB4 (GradCAM) → NB5 (GAN, genera gan_models/)
```

NB4 y NB5 cargan directamente los checkpoints guardados por NB3, por lo que **NB3 debe
haberse ejecutado previamente** (o los archivos `.pth` deben estar presentes en `modelos/`).

---

## Autoras
Carmen Miralles & Paula Echamendi
Proyecto académico de NLP aplicado a datos no estructurados.
