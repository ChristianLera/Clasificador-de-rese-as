Aquí tienes un **README.md profesional y digno de GitHub**, actualizado con todas las funcionalidades del programa.

---

## 📄 `README.md` (copia y pega esto en tu repositorio)

```markdown
# 🎯 Amazon Review Analyzer

**Clasificador profesional de reseñas con Ensemble Learning | Procesamiento por lotes | Múltiples formatos**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![Tkinter](https://img.shields.io/badge/GUI-Tkinter-brightgreen.svg)](https://docs.python.org/3/library/tkinter.html)

---

## 📌 Tabla de Contenidos

- [¿Qué hace?](#qué-hace)
- [Características principales](#características-principales)
- [Capturas de pantalla](#capturas-de-pantalla)
- [Cómo funciona (Ensemble Learning)](#cómo-funciona)
- [Niveles de precisión](#niveles-de-precisión)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Cómo usar el programa](#cómo-usar-el-programa)
- [Gestión de caché](#gestión-de-caché)
- [Análisis de múltiples archivos](#análisis-de-múltiples-archivos)
- [Preguntas frecuentes](#preguntas-frecuentes)
- [Posibles errores y soluciones](#posibles-errores-y-soluciones)
- [Licencia](#licencia)

---

## 🎯 ¿Qué hace?

Clasifica reseñas de productos como **POSITIVAS** o **NEGATIVAS** utilizando Machine Learning avanzado.

| Entrada | Salida |
|---------|--------|
| "Este producto es excelente, lo recomiendo" | ✅ **POSITIVA** (95% confianza) |
| "Mala calidad, se rompió a los 2 días" | ❌ **NEGATIVA** (92% confianza) |

---

## ✨ Características principales

| Característica | Descripción |
|----------------|-------------|
| 🧠 **Ensemble Learning** | Combina 4 modelos para máxima precisión |
| 📊 **4 niveles de precisión** | Rápido (85-90%) → Ensemble (95-97%) |
| 🔄 **Actualización en cascada** | Entrenar un nivel actualiza los inferiores |
| ⚡ **Sin congelamiento** | Procesamiento en hilos, interfaz siempre responde |
| ⏹️ **Cancelación en tiempo real** | Detén entrenamiento/análisis cuando quieras |
| ⏱️ **Límite de tiempo configurable** | Establece tiempo máximo para operaciones |
| 📁 **Múltiples datasets** | Combina varios archivos CSV para entrenar |
| 📂 **Análisis por lotes** | Analiza múltiples archivos CSV a la vez |
| 📋 **Ventana de Log** | Todo lo que ocurre en tiempo real |
| 📄 **Múltiples formatos** | CSV, Excel, JSON, Parquet, TSV |
| 🌐 **Multilingüe** | Detecta español e inglés automáticamente |
| 💾 **Caché inteligente** | Acelera entrenamientos repetidos |
| 📤 **Exportación a CSV** | Guarda resultados para análisis externo |
| 🗑️ **Gestión de caché** | Visualiza y elimina archivos de caché |

---

## 📸 Capturas de pantalla

### Ventana principal
```
┌─────────────────────────────────────────────────────────────────┐
│  📊 Amazon Review Analyzer                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📁 1. GESTIONAR DATASETS                                        │
│     [📂 AÑADIR ARCHIVOS]  [🗑️ LIMPIAR LISTA]                    │
│     ├── dataset_electronica.csv (2.3 GB)                        │
│     ├── dataset_ropa.csv (1.1 GB)                               │
│     └── dataset_libros.csv (850 MB)                             │
│                                                                  │
│  🎯 2. ENTRENAR MODELO                                           │
│     ○ RÁPIDO  ○ PRECISO  ○ MÁXIMO  ○ ENSEMBLE                   │
│     [🎯 ENTRENAR MODELO]                                         │
│                                                                  │
│  📊 3. ANALIZAR RESEÑAS                                          │
│     [📝 Analizar Texto] [📁 Analizar Archivos CSV]              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Ventana de progreso con tiempos
```
┌─────────────────────────────────────────────────────────────────┐
│  🎯 ENTRENANDO MODELO ENSEMBLE                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ████████████░░░░░░░░░░░░░░  45%                                │
│                                                                  │
│  ⏱️ Tiempo transcurrido: 22 minutos                             │
│  ⏰ Tiempo restante: 27 minutos                                 │
│                                                                  │
│  📊 Procesando reseña 4,523,596/10,000,000                      │
│                                                                  │
│  [CANCELAR]                                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Cómo funciona (Ensemble Learning)

En lugar de usar un solo modelo, el programa combina **múltiples modelos** que "votan":

```
Reseña: "Excelente producto"
        │
    ┌───┼───┬───┐
    ▼   ▼   ▼   ▼
   LR   NB  RF  SVM    ← Cada modelo predice
    │   │   │   │
   Pos  Pos  Pos Neg
    │   │   │   │
    └───┴───┴───┘
        │
        ▼
    VOTACIÓN: 3 vs 1
        │
        ▼
    RESULTADO: POSITIVA
```

### Los 4 modelos

| Modelo | Nombre | Función |
|--------|--------|---------|
| **LR** | Logistic Regression | Clasificador lineal base |
| **NB** | Naive Bayes | Basado en probabilidades |
| **RF** | Random Forest | Múltiples árboles de decisión |
| **SVM** | Support Vector Machine | Encuentra bordes de separación |

---

## 📊 Niveles de precisión

| Nivel | Modelos | Precisión | Tiempo estimado | Cuándo usar |
|-------|---------|-----------|-----------------|-------------|
| **RÁPIDO** | LR + NB | 85-90% | 5-10 min | Pruebas rápidas |
| **PRECISO** | LR + NB + RF | 90-95% | 15-30 min | Uso diario |
| **MÁXIMO** | LR + NB + RF (trigramas) | 92-96% | 30-50 min | Alta precisión |
| **ENSEMBLE** | LR + NB + RF + SVM | 95-97% | 1-1.5 h | **Resultados profesionales** |

### Actualización en cascada

```
Entrenas ENSEMBLE → actualiza MÁXIMO, PRECISO, RÁPIDO
Entrenas MÁXIMO   → actualiza PRECISO, RÁPIDO
Entrenas PRECISO  → actualiza RÁPIDO
Entrenas RÁPIDO   → no actualiza a nadie
```

---

## 📁 Estructura del proyecto

```
amazon_review_classifier/
│
├── data/                      # Tus datasets
├── models/                    # Modelos guardados
│   ├── cache/                 # Caché de preprocesamiento
│   ├── modelo_rapido.pkl
│   ├── modelo_preciso.pkl
│   ├── modelo_maximo.pkl
│   └── modelo_ensemble.pkl
├── logs/                      # Logs del programa
├── src/                       # Código fuente
│   ├── __init__.py
│   ├── preprocess.py          # Limpieza de texto
│   ├── train.py               # Entrenamiento
│   ├── train_process.py       # Proceso independiente
│   ├── predict.py             # Predicciones
│   ├── models_config.py       # Configuración
│   ├── utils.py               # Utilidades
│   ├── dataset_manager.py     # Gestión de datasets
│   ├── cache_manager.py       # Gestión de caché
│   └── cache_preferences.py   # Preferencias de caché
├── requirements.txt
├── README.md
└── main.py                    # Punto de entrada
```

---

## 🚀 Instalación

### Requisitos previos
- Python 3.8 o superior
- pip (gestor de paquetes)

### Paso a paso

```bash
# 1. Clonar el repositorio
git clone https://github.com/tuusuario/amazon-review-analyzer.git
cd amazon-review-analyzer

# 2. Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar el programa
python main.py
```

### Dependencias

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8.0
joblib>=1.3.0
openpyxl>=3.1.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 📖 Cómo usar el programa

### 🔹 PASO 1: GESTIONAR DATASETS

- Haz clic en **"AÑADIR ARCHIVOS"**
- Selecciona **uno o varios** archivos CSV
- El programa los combinará automáticamente
- Puedes eliminar archivos individuales o limpiar toda la lista

### 🔹 PASO 2: ENTRENAR MODELO

- Selecciona el nivel de precisión (recomendado: **ENSEMBLE**)
- Configura límite de tiempo y límite de reseñas (opcional)
- Haz clic en **"ENTRENAR MODELO"**
- Puedes **cancelar** en cualquier momento

### 🔹 PASO 3: ANALIZAR RESEÑAS

**Opción A - Analizar texto:**
- Escribe reseñas manualmente (una por línea)
- Haz clic en **"ANALIZAR"**

**Opción B - Analizar múltiples archivos:**
- Ve a la pestaña **"Analizar Archivos CSV"**
- Haz clic en **"AÑADIR ARCHIVOS"**
- Selecciona varios archivos CSV
- Haz clic en **"ANALIZAR TODOS LOS ARCHIVOS"**

### 🔹 PASO 4: VER RESULTADOS

- **Resumen global** (total, positivas, negativas)
- **Resumen por archivo** (cada archivo con sus estadísticas)
- **Gráfico de barras** visual
- **Primeras reseñas** analizadas

### 🔹 PASO 5: EXPORTAR RESULTADOS

- Ve a la pestaña **"Resultados"**
- Haz clic en **"EXPORTAR RESULTADOS"**
- Se guardará un CSV con todas las clasificaciones

---

## 💾 Gestión de caché

La caché acelera entrenamientos repetidos guardando textos preprocesados y matrices TF-IDF.

### Acceder a la gestión de caché:

**Menú → Ver → Gestionar caché**

### La ventana muestra:
- Espacio total ocupado
- Número de archivos
- Lista de archivos con nombre, tamaño y fecha
- Opciones: eliminar archivo individual o toda la caché

### Nombres de archivo legibles:
```
preproc_dataset_electronica_20250425_Ensemble.pkl
vector_dataset_electronica_20250425_Ensemble_mf35000_ng13.pkl
```

---

## 📂 Análisis de múltiples archivos

### ¿Qué puedes hacer?

| Acción | Cómo |
|--------|------|
| Añadir archivos | Botón **"AÑADIR ARCHIVOS"** |
| Eliminar archivo | Selecciona y haz clic en **"Eliminar seleccionado"** |
| Limpiar lista | Botón **"LIMPIAR LISTA"** |
| Analizar todos | Botón **"ANALIZAR TODOS LOS ARCHIVOS"** |

### Resultados por archivo:

```
📁 RESUMEN POR ARCHIVO:

   📄 dataset_electronica.csv: 30,000 reseñas
      👍 Positivas: 13,328 (44.4%)
      👎 Negativas: 16,672 (55.6%)
      ⏱️ Tiempo: 2.5 minutos

   📄 dataset_ropa.csv: 3,500 reseñas
      👍 Positivas: 648 (18.5%)
      👎 Negativas: 2,852 (81.5%)
      ⏱️ Tiempo: 0.3 minutos
```

---

## 📂 Formatos soportados

| Formato | Extensión | Estado |
|---------|-----------|--------|
| CSV | .csv | ✅ Soportado |
| TSV | .tsv | ✅ Soportado |
| Excel | .xlsx, .xls | ✅ Soportado |
| JSON | .json | ✅ Soportado |
| Parquet | .parquet | ✅ Soportado |

---

## ❓ Preguntas frecuentes

### ¿Puedo usar reseñas en español?
**Sí.** El programa detecta automáticamente español e inglés.

### ¿Qué hago si el programa se "congela"?
**El programa NO se congela** porque usa hilos. Siempre puedes hacer clic en "CANCELAR".

### ¿Puedo guardar un modelo entrenado?
**Sí.** Los modelos se guardan automáticamente en la carpeta `models/`. También puedes exportarlos con nombre personalizado.

### ¿Cuántas reseñas puedo analizar?
**Miles sin problema.** El programa procesa por lotes. Puedes cancelar si es demasiado grande.

### ¿Qué diferencia hay entre los niveles?
**Más modelos = más precisión, pero más tiempo.** Para pruebas usa RÁPIDO. Para resultados profesionales usa **ENSEMBLE**.

### ¿Puedo analizar varios archivos a la vez?
**Sí.** Usa la pestaña "Analizar Archivos CSV" y añade múltiples archivos.

### ¿Dónde se guardan los resultados exportados?
En el archivo CSV que elijas al hacer clic en "EXPORTAR RESULTADOS".

---

## ⚠️ Posibles errores y soluciones

| Error | Solución |
|-------|----------|
| `No module named 'sklearn'` | `pip install scikit-learn` |
| `No module named 'nltk'` | `pip install nltk` |
| `No module named 'joblib'` | `pip install joblib` |
| Archivo CSV no se lee | Asegúrate de que tenga columna de texto y calificación |
| Modelo no entrena | Verifica que el dataset tenga reseñas válidas |
| El programa se cierra solo | Abre terminal y ejecuta `python main.py` para ver el error |
| Error de codificación en consola | Los emojis pueden no mostrarse en Windows. El programa sigue funcionando |

---

## 📄 Licencia

**MIT License** - Puedes usar, modificar y distribuir este programa libremente.

---

## 👨‍💻 Autor

**Christian Lera**

Proyecto profesional para portafolio. Clasificador de reseñas con Ensemble Learning.

---

## ⭐ ¿Te ha sido útil?

Si este proyecto te ha ayudado, considera darle una ⭐ en GitHub.

---

## 📞 Contacto

Para preguntas o sugerencias, abre un issue en el repositorio.

---
