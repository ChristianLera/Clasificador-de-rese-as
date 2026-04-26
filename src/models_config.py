"""
src/models_config.py
Configuración de los diferentes niveles de modelos
Jerarquía de actualización: Ensemble → Máximo → Preciso → Rápido
"""

# Configuración de cada nivel de precisión
MODELOS_NIVELES = {
    "Rápido": {
        "nombre_archivo": "modelo_rapido.pkl",
        "max_features": 5000,
        "num_resenas_default": 50000,
        "modelos_a_usar": ['lr', 'nb'],
        "ngram": (1, 2),
        "C": 1.5,
        "descripcion": "2 modelos (LR + NB) - 85-90% precisión",
        "tiempo_estimado": "5-10 minutos",
        "color": "#3498db"
    },
    "Preciso": {
        "nombre_archivo": "modelo_preciso.pkl",
        "max_features": 15000,
        "num_resenas_default": 200000,
        "modelos_a_usar": ['lr', 'nb', 'rf'],
        "ngram": (1, 2),
        "C": 2.5,
        "descripcion": "3 modelos (LR + NB + RF) - 90-95% precisión",
        "tiempo_estimado": "15-30 minutos",
        "color": "#2ecc71"
    },
    "Máximo": {
        "nombre_archivo": "modelo_maximo.pkl",
        "max_features": 25000,
        "num_resenas_default": 400000,
        "modelos_a_usar": ['lr', 'nb', 'rf'],
        "ngram": (1, 3),
        "C": 3.5,
        "descripcion": "3 modelos con trigramas - 92-96% precisión",
        "tiempo_estimado": "30-50 minutos",
        "color": "#e67e22"
    },
    "Ensemble": {
        "nombre_archivo": "modelo_ensemble.pkl",
        "max_features": 35000,
        "num_resenas_default": 500000,
        "modelos_a_usar": ['lr', 'nb', 'rf', 'svm'],
        "ngram": (1, 3),
        "C": 4.0,
        "descripcion": "4 modelos (LR + NB + RF + SVM) - 95-97% precisión",
        "tiempo_estimado": "1-1.5 horas",
        "color": "#9b59b6"
    }
}

# Jerarquía de actualización en cascada
JERARQUIA_ACTUALIZACION = {
    "Rápido": [],
    "Preciso": ["Rápido"],
    "Máximo": ["Preciso", "Rápido"],
    "Ensemble": ["Máximo", "Preciso", "Rápido"]
}

# Opciones de límite de tiempo
OPCIONES_TIMEOUT = [
    ("Sin límite", None),
    ("10 minutos", 600),
    ("30 minutos", 1800),
    ("1 hora", 3600),
    ("2 horas", 7200)
]

# Opciones disponibles en la interfaz
OPCIONES_PRECISION = [
    ("⚡ RÁPIDO", "Rápido"),
    ("🎯 PRECISO", "Preciso"),
    ("🏆 MÁXIMO", "Máximo"),
    ("👑 ENSEMBLE", "Ensemble")
]

NOMBRE_MODELOS = {
    "lr": "Logistic Regression",
    "nb": "Naive Bayes",
    "rf": "Random Forest",
    "svm": "SVM"
}

OPCIONES_LIMITE_RESENAS = [
    ("500,000 reseñas", 500000),
    ("1,000,000 reseñas", 1000000),
    ("2,000,000 reseñas", 2000000),
    ("5,000,000 reseñas", 5000000),
    ("10,000,000 reseñas", 10000000),
    ("Sin límite", None)
]

LIMITE_RESENAS_POR_DEFECTO = 1000000

# Texto de ayuda para el usuario
TEXTO_AYUDA = """
🎯 ¿QUÉ HACE ESTE PROGRAMA?

Clasifica reseñas de productos como POSITIVAS o NEGATIVAS utilizando 
Machine Learning con ENSEMBLE (combinación de múltiples modelos).
Soporta análisis de múltiples archivos y gestión de datasets combinados.

================================================================

📁 FORMATOS SOPORTADOS

✅ CSV (.csv) - Separado por comas
✅ TSV (.tsv) - Separado por tabulaciones  
✅ Excel (.xlsx, .xls)
✅ JSON (.json)
✅ Parquet (.parquet)

================================================================

🎯 NIVELES DE PRECISIÓN

┌─────────────┬─────────────────────────────┬─────────────┬─────────────────┐
│ NIVEL       │ MODELOS                     │ PRECISIÓN   │ TIEMPO ESTIMADO │
├─────────────┼─────────────────────────────┼─────────────┼─────────────────┤
│ RÁPIDO      │ LR + NB                     │ 85-90%      │ 5-10 min        │
│ PRECISO     │ LR + NB + RF                │ 90-95%      │ 15-30 min       │
│ MÁXIMO      │ LR + NB + RF (trigramas)    │ 92-96%      │ 30-50 min       │
│ ENSEMBLE    │ LR + NB + RF + SVM          │ 95-97%      │ 1-1.5 h         │
└─────────────┴─────────────────────────────┴─────────────┴─────────────────┘

LR = Logistic Regression
NB = Naive Bayes
RF = Random Forest
SVM = Support Vector Machine

================================================================

🔄 ACTUALIZACIÓN EN CASCADA

Cuando entrenas un modelo, los modelos inferiores se actualizan:
• ENSEMBLE → actualiza MÁXIMO, PRECISO y RÁPIDO
• MÁXIMO → actualiza PRECISO y RÁPIDO
• PRECISO → actualiza RÁPIDO
• RÁPIDO → no actualiza a nadie

================================================================

📁 GESTIÓN DE MÚLTIPLES DATASETS

Puedes cargar VARIOS archivos CSV a la vez:
• Botón "AÑADIR ARCHIVOS" → selecciona múltiples archivos
• Se combinan automáticamente en un solo dataset
• Elimina archivos de la lista si no los quieres

El entrenamiento usará TODOS los archivos combinados.

================================================================

📊 ANÁLISIS DE MÚLTIPLES ARCHIVOS

Puedes analizar VARIOS archivos CSV a la vez:
• Pestaña "Analizar Archivos CSV"
• Botón "AÑADIR ARCHIVOS" → selecciona múltiples archivos
• Botón "ANALIZAR TODOS LOS ARCHIVOS" → procesa todos

Resultados:
• Resumen global (total de todos los archivos)
• Resumen POR ARCHIVO (positivas, negativas, tiempo)
• Exportación a CSV con todos los resultados

================================================================

💾 GESTIÓN DE CACHÉ

La caché acelera entrenamientos repetidos:
• Menú "Ver" → "Gestionar caché"
• Muestra espacio ocupado y lista de archivos
• Permite eliminar archivos individuales o toda la caché

Los archivos de caché tienen nombres legibles:
preproc_nombre_dataset_fecha_nivel.pkl
vector_nombre_dataset_fecha_nivel_mf...pkl

================================================================

📋 CÓMO USAR EL PROGRAMA

PASO 1: GESTIONAR DATASETS
   - Haz clic en "AÑADIR ARCHIVOS" en la sección 1
   - Selecciona uno o varios archivos CSV
   - El programa los combinará automáticamente

PASO 2: ENTRENAR MODELO
   - Selecciona el nivel de precisión (Rápido/Preciso/Máximo/Ensemble)
   - Configura límite de tiempo y reseñas (opcional)
   - Haz clic en "ENTRENAR MODELO"

PASO 3: ANALIZAR RESEÑAS
   - Pestaña "Analizar Texto": escribe reseñas manualmente (una por línea)
   - Pestaña "Analizar Archivos CSV": añade archivos y analiza todos a la vez

PASO 4: VER RESULTADOS
   - Resultados globales (total, positivas, negativas)
   - Resumen por archivo (cada archivo con sus estadísticas)
   - Visualización con gráfico de barras

PASO 5: EXPORTAR RESULTADOS
   - Haz clic en "EXPORTAR RESULTADOS" en la pestaña de resultados
   - Se guarda un CSV con todas las clasificaciones

================================================================

⏹️ CANCELAR OPERACIONES

Durante entrenamiento o análisis:
   - Puedes hacer clic en "CANCELAR"
   - El programa terminará el lote actual y luego se detendrá
   - No se pierde el progreso ya procesado

================================================================

📊 INTERFAZ

• Barra de progreso: muestra el avance
• Ventana de Log: muestra detalles en tiempo real
• Tiempos estimados: antes y durante las operaciones
• Estado: indica qué está haciendo el programa

================================================================

🗑️ GESTIÓN DE CACHÉ

• Menú "Ver" → "Gestionar caché"
• Muestra espacio total ocupado
• Lista de archivos con tamaño y fecha
• Eliminar archivos individuales o toda la caché

================================================================

❓ PREGUNTAS FRECUENTES

¿Puedo usar reseñas en español?
   - Sí, el programa detecta automáticamente español e inglés

¿Qué hago si el programa se "congela"?
   - El programa NO se congela porque usa hilos
   - Siempre puedes cancelar la operación o hacer clic en otros botones

¿Puedo guardar un modelo entrenado?
   - Sí, los modelos se guardan automáticamente
   - También puedes exportarlos con nombre personalizado

¿Cuántas reseñas puedo analizar?
   - Miles sin problema. El programa procesa por lotes
   - Puedes cancelar si es demasiado grande

¿Puedo analizar varios archivos a la vez?
   - Sí, en la pestaña "Analizar Archivos CSV" puedes añadir múltiples archivos

¿Dónde se guardan los resultados exportados?
   - En el archivo CSV que elijas al hacer clic en "EXPORTAR RESULTADOS"

================================================================

💡 CONSEJOS

1. Para pruebas rápidas, usa el nivel RÁPIDO
2. Para resultados profesionales, usa ENSEMBLE
3. Si tienes muchas reseñas (>100,000), establece un límite de tiempo
4. Usa múltiples archivos CSV para organizar tus datos por categorías
5. La caché acelera mucho los entrenamientos repetidos
6. Puedes cancelar y reanudar entrenamientos más tarde
7. Revisa el LOG para ver el progreso detallado

================================================================

🔧 SOLUCIÓN DE PROBLEMAS

Error al cargar archivo:
   - Verifica que el archivo tenga columnas de texto y calificación
   - Asegúrate de que el formato sea CSV, Excel, JSON o Parquet

El programa se cierra al finalizar:
   - Espera 2 segundos, la ventana de progreso se cierra sola
   - El programa principal sigue abierto

Las predicciones son incorrectas:
   - Entrena con más datos (al menos 10,000 reseñas por clase)
   - Usa el nivel ENSEMBLE para máxima precisión
"""


def get_configuracion_nivel(nivel):
    """Obtiene la configuración de un nivel"""
    return MODELOS_NIVELES.get(nivel, MODELOS_NIVELES["Rápido"])


def get_niveles_a_actualizar(nivel_entrenado):
    """Obtiene qué niveles se deben actualizar al entrenar un nivel"""
    return JERARQUIA_ACTUALIZACION.get(nivel_entrenado, [])
