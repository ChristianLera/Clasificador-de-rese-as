# 🎯 Amazon Review Analyzer

**Clasificador profesional de reseñas con Ensemble Learning | 95-97% de precisión**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative)](LICENSE)
[![Tkinter](https://img.shields.io/badge/GUI-Tkinter-brightgreen?style=for-the-badge&logo=python)](https://docs.python.org/3/library/tkinter.html)

---

## 📌 ¿Qué hace?

Analiza reseñas de productos y las clasifica como **POSITIVAS** o **NEGATIVAS** usando Machine Learning.

| Tu reseña | El programa dice |
|-----------|------------------|
| "Este producto es excelente, lo recomiendo" | 🟢 POSITIVA (95% confianza) |
| "Mala calidad, se rompió a los 2 días" | 🔴 NEGATIVA (92% confianza) |

---

## ✨ Características

- 🧠 **Ensemble Learning** - Combina 4 modelos que votan para decidir
- ⚡ **Sin congelamiento** - Usa hilos, la interfaz siempre responde
- 📁 **Múltiples datasets** - Combina varios CSV para entrenar
- 📂 **Análisis por lotes** - Analiza múltiples archivos a la vez
- 💾 **Caché inteligente** - Segundo entrenamiento mucho más rápido
- 🌐 **Multilingüe** - Detecta español e inglés automáticamente
- ⏱️ **Tiempos estimados** - Muestra tiempo restante en tiempo real
- 🗑️ **Gestión de caché** - Visualiza y elimina archivos de caché
- 📤 **Exportación a CSV** - Guarda resultados para analizar en Excel

---

## 🎯 Niveles de precisión

| Nivel | Modelos | Precisión | Tiempo |
|-------|---------|-----------|--------|
| ⚡ RAPIDO | LR + NB | 85-90% | 5-10 min |
| 🎯 PRECISO | LR + NB + RF | 90-95% | 15-30 min |
| 🏆 MAXIMO | LR + NB + RF (trigramas) | 92-96% | 30-50 min |
| 👑 ENSEMBLE | LR + NB + RF + SVM | 95-97% | 1-1.5 h |

> LR = Logistic Regression | NB = Naive Bayes | RF = Random Forest | SVM = Support Vector Machine

---

## 🔄 Actualización en cascada

Cuando entrenas un nivel, los inferiores se actualizan automaticamente:

- ENSEMBLE → actualiza MAXIMO, PRECISO y RAPIDO
- MAXIMO → actualiza PRECISO y RAPIDO
- PRECISO → actualiza RAPIDO
- RAPIDO → no actualiza a nadie

---

## 🚀 Instalacion

```bash
git clone https://github.com/TU_USUARIO/amazon-review-analyzer.git
cd amazon-review-analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## 📖 Como usar

| Paso | Que hacer |
|------|-----------|
| 1 | Boton "AÑADIR ARCHIVOS" → selecciona uno o varios CSV |
| 2 | Selecciona nivel (recomendado: ENSEMBLE) → "ENTRENAR MODELO" |
| 3 | Pestaña "Analizar Archivos CSV" → añade archivos → "ANALIZAR TODOS" |
| 4 | Pestaña "Resultados" → "EXPORTAR RESULTADOS" |

---

## 📂 Estructura

```
amazon_review_classifier/
├── data/                      # Tus datasets
├── models/                    # Modelos guardados
│   └── cache/                 # Cache de preprocesamiento
├── logs/                      # Logs del programa
├── src/                       # Codigo fuente
│   ├── preprocess.py          # Limpieza de texto
│   ├── train.py               # Entrenamiento
│   ├── predict.py             # Predicciones
│   ├── dataset_manager.py     # Gestion de datasets
│   └── cache_manager.py       # Gestion de cache
├── requirements.txt
├── README.md
└── main.py
```

---

## 💾 Gestion de cache

Menu → Ver → Gestionar cache

- Muestra espacio total y lista de archivos
- Permite eliminar archivos individuales o toda la cache
- Nombres legibles: preproc_dataset_fecha_nivel.pkl

---

## ❓ Preguntas frecuentes

**¿Puedo usar reseñas en español?**
Si, el programa detecta automaticamente español e ingles.

**¿Que hago si el programa se congela?**
No se congela. Siempre puedes hacer clic en "CANCELAR".

**¿Cuantas reseñas puedo analizar?**
Miles sin problema. Procesa por lotes y puedes cancelar.

**¿Que diferencia hay entre los niveles?**
Mas modelos = mas precision, pero mas tiempo.

**¿Puedo analizar varios archivos a la vez?**
Si, en la pestaña "Analizar Archivos CSV".

---

## ⚠️ Errores comunes

| Error | Solucion |
|-------|----------|
| No module named 'sklearn' | pip install scikit-learn |
| No module named 'nltk' | pip install nltk |
| CSV no se lee | Debe tener columna de texto y calificacion |

---

## 📄 Licencia

MIT License - Puedes usar, modificar y distribuir libremente.

---

## 👨‍💻 Autor

Christian Lera

Proyecto profesional para portafolio.

---

⭐ Si te ha sido util, dale una estrella en GitHub.
