"""
src/utils.py
Utilidades generales: carga de datos, gestión de modelos, logging
"""

import os
import pandas as pd
import joblib
from datetime import datetime
from .models_config import MODELOS_NIVELES


def cargar_dataset(ruta_archivo, columna_texto=None, columna_rating=None, nrows=None):
    """
    Carga un dataset desde diferentes formatos
    """
    extension = os.path.splitext(ruta_archivo)[1].lower()
    
    if extension == '.csv':
        df = pd.read_csv(ruta_archivo, nrows=nrows)
    elif extension == '.tsv':
        df = pd.read_csv(ruta_archivo, sep='\t', nrows=nrows)
    elif extension in ['.xlsx', '.xls']:
        df = pd.read_excel(ruta_archivo, nrows=nrows)
    elif extension == '.json':
        df = pd.read_json(ruta_archivo)
        if nrows:
            df = df.head(nrows)
    elif extension == '.parquet':
        df = pd.read_parquet(ruta_archivo)
        if nrows:
            df = df.head(nrows)
    else:
        raise ValueError(f"Formato no soportado: {extension}")
    
    # Detectar columna de texto
    if columna_texto is None:
        for col in df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in ['review', 'text', 'resena', 'comentario', 'comment', 'body', 'reseña']):
                columna_texto = col
                break
        if columna_texto is None:
            columna_texto = df.columns[0]
    
    # Detectar columna de calificación
    if columna_rating is None:
        for col in df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in ['rating', 'overall', 'star', 'puntuacion', 'score', 'calificacion', 'estrellas']):
                columna_rating = col
                break
    
    # Crear DataFrame estandarizado
    df_resultado = pd.DataFrame()
    df_resultado['reviewText'] = df[columna_texto].astype(str)
    
    if columna_rating is not None:
        df_resultado['overall'] = pd.to_numeric(df[columna_rating], errors='coerce')
    else:
        df_resultado['overall'] = 5
    
    df_resultado = df_resultado.dropna(subset=['reviewText'])
    df_resultado = df_resultado[df_resultado['reviewText'].str.strip() != '']
    df_resultado = df_resultado[df_resultado['reviewText'] != 'nan']
    
    return df_resultado


def guardar_modelo(nivel, datos, carpeta_modelos):
    """Guarda un modelo en disco"""
    config = MODELOS_NIVELES.get(nivel)
    if config:
        archivo = os.path.join(carpeta_modelos, config['nombre_archivo'])
        joblib.dump(datos, archivo)
        return archivo
    return None


def cargar_modelo(nivel, carpeta_modelos):
    """Carga un modelo desde disco"""
    config = MODELOS_NIVELES.get(nivel)
    if config:
        archivo = os.path.join(carpeta_modelos, config['nombre_archivo'])
        if os.path.exists(archivo):
            return joblib.load(archivo)
    return None


def listar_modelos_guardados(carpeta_modelos):
    """Lista los modelos guardados en la carpeta"""
    modelos_info = {}
    
    for nivel, config in MODELOS_NIVELES.items():
        archivo = os.path.join(carpeta_modelos, config['nombre_archivo'])
        existe = os.path.exists(archivo)
        
        info = {
            'existe': existe,
            'archivo': config['nombre_archivo'],
            'descripcion': config['descripcion']
        }
        
        if existe:
            try:
                data = joblib.load(archivo)
                info['accuracy'] = data.get('accuracy', 0)
                info['modelos'] = data.get('modelos_usados', [])
                info['fecha'] = os.path.getmtime(archivo)
            except:
                info['accuracy'] = 0
                info['modelos'] = []
        
        modelos_info[nivel] = info
    
    return modelos_info


def obtener_modelo_por_defecto(carpeta_modelos):
    """Obtiene el modelo por defecto (Ensemble si existe, sino el mejor disponible)"""
    jerarquia = ['Ensemble', 'Máximo', 'Preciso', 'Rápido']
    
    for nivel in jerarquia:
        config = MODELOS_NIVELES.get(nivel)
        if config:
            archivo = os.path.join(carpeta_modelos, config['nombre_archivo'])
            if os.path.exists(archivo):
                return nivel
    
    return None


class Logger:
    """Clase para manejar logging en tiempo real"""
    
    def __init__(self, archivo_log=None):
        self.archivo_log = archivo_log
        if archivo_log:
            os.makedirs(os.path.dirname(archivo_log), exist_ok=True)
    
    def log(self, mensaje, callback=None):
        """Registra un mensaje con timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        mensaje_completo = f"[{timestamp}] {mensaje}"
        
        if callback:
            callback(mensaje_completo)
        
        if self.archivo_log:
            with open(self.archivo_log, 'a', encoding='utf-8') as f:
                f.write(mensaje_completo + '\n')
        
        return mensaje_completo
