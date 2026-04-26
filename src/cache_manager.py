"""
src/cache_manager.py
Gestión de caché para datos preprocesados y vectorizados
CON NOMBRES LEGIBLES PARA IDENTIFICAR CADA DATASET
"""

import os
import joblib
import hashlib
import json
from datetime import datetime
import re


class CacheManager:
    """Maneja el almacenamiento en caché de datos preprocesados"""
    
    def __init__(self, carpeta_cache):
        self.carpeta_cache = carpeta_cache
        os.makedirs(carpeta_cache, exist_ok=True)
    
    def _get_nombre_dataset(self, df):
        """Intenta obtener un nombre legible para el dataset"""
        # Intentar obtener nombre del archivo original (si está disponible)
        if hasattr(df, 'nombre_archivo') and df.nombre_archivo:
            nombre = os.path.basename(df.nombre_archivo)
            nombre = os.path.splitext(nombre)[0]
            nombre = re.sub(r'[^\w\-_\.]', '_', nombre)
            print(f"📛 CACHE: usando nombre_archivo = {nombre}")
            return nombre[:50]
        
        # Si no, usar un hash legible
        primeras_resenas = df['reviewText'].head(5).tolist()
        texto_muestra = ' '.join(primeras_resenas)[:50]
        hash_corto = hashlib.md5(texto_muestra.encode()).hexdigest()[:8]
        print(f"⚠️ CACHE: usando hash porque no hay nombre_archivo")
        return f"dataset_{hash_corto}"
    
    def _get_hash_dataset(self, df, limite_resenas, nivel):
        """Genera un hash para identificar el dataset (usado internamente)"""
        num_filas = len(df)
        primeras_resenas = df['reviewText'].head(100).tolist()
        contenido_hash = hashlib.md5(str(primeras_resenas).encode()).hexdigest()[:10]
        return f"{num_filas}_{contenido_hash}_{limite_resenas}_{nivel}"
    
    def _get_nombre_archivo_cache(self, df, limite_resenas, nivel, tipo, extra=""):
        """Genera un nombre de archivo LEGIBLE para la caché"""
        nombre_dataset = self._get_nombre_dataset(df)
        fecha = datetime.now().strftime("%Y%m%d")
        limite_str = f"_{limite_resenas}" if limite_resenas else ""
        
        if tipo == "preproc":
            return f"preproc_{nombre_dataset}_{fecha}_{nivel}{limite_str}.pkl"
        elif tipo == "vector":
            return f"vector_{nombre_dataset}_{fecha}_{nivel}{limite_str}_{extra}.pkl"
        return f"cache_{nombre_dataset}_{fecha}_{nivel}{limite_str}.pkl"
    
    def existe_preprocesado(self, df, limite_resenas, nivel):
        """Verifica si existe el caché de preprocesamiento"""
        nombre_base = self._get_nombre_dataset(df)
        
        for archivo in os.listdir(self.carpeta_cache):
            if archivo.startswith(f"preproc_{nombre_base}_") and archivo.endswith(".pkl"):
                return True
        
        hash_id = self._get_hash_dataset(df, limite_resenas, nivel)
        archivo_hash = os.path.join(self.carpeta_cache, f"preproc_{hash_id}.pkl")
        return os.path.exists(archivo_hash)
    
    def guardar_preprocesado(self, df, limite_resenas, nivel, textos_procesados, etiquetas):
        """Guarda los datos preprocesados en caché con nombre legible"""
        nombre_archivo = self._get_nombre_archivo_cache(df, limite_resenas, nivel, "preproc")
        archivo_cache = os.path.join(self.carpeta_cache, nombre_archivo)
        
        datos = {
            'textos_procesados': textos_procesados,
            'etiquetas': etiquetas,
            'fecha': datetime.now().isoformat(),
            'nivel': nivel,
            'num_filas': len(df),
            'limite_resenas': limite_resenas,
            'nombre_dataset': self._get_nombre_dataset(df)
        }
        joblib.dump(datos, archivo_cache, compress=3)
        return archivo_cache
    
    def guardar_preprocesado(self, df, limite_resenas, nivel, textos_procesados, etiquetas, nombre_dataset_override=None):
        """Guarda los datos preprocesados en caché con nombre legible"""
        if nombre_dataset_override:
            nombre_dataset = nombre_dataset_override
        elif hasattr(df, 'nombre_archivo') and df.nombre_archivo:
            nombre_dataset = self._get_nombre_dataset(df)
        else:
            nombre_dataset = "dataset_desconocido"
        
        fecha = datetime.now().strftime("%Y%m%d")
        limite_str = f"_{limite_resenas}" if limite_resenas else ""
        nombre_archivo = f"preproc_{nombre_dataset}_{fecha}_{nivel}{limite_str}.pkl"
        archivo_cache = os.path.join(self.carpeta_cache, nombre_archivo)
        
        datos = {
            'textos_procesados': textos_procesados,
            'etiquetas': etiquetas,
            'fecha': datetime.now().isoformat(),
            'nivel': nivel,
            'num_filas': len(df),
            'limite_resenas': limite_resenas,
            'nombre_dataset': nombre_dataset
        }
        joblib.dump(datos, archivo_cache, compress=3)
        return archivo_cache
    
    def existe_vectorizado(self, df, limite_resenas, nivel, max_features, ngram):
        """Verifica si existe el caché de vectorización"""
        nombre_dataset = self._get_nombre_dataset(df)
        
        for archivo in os.listdir(self.carpeta_cache):
            if archivo.startswith(f"vector_{nombre_dataset}_") and archivo.endswith(".pkl"):
                return True
        
        hash_id = self._get_hash_dataset(df, limite_resenas, nivel)
        hash_extra = hashlib.md5(f"{max_features}_{ngram}".encode()).hexdigest()[:8]
        archivo_hash = os.path.join(self.carpeta_cache, f"vector_{hash_id}_{hash_extra}.pkl")
        return os.path.exists(archivo_hash)
    
    def guardar_vectorizado(self, df, limite_resenas, nivel, max_features, ngram, X, y, vectorizer, nombre_dataset_override=None):
        """Guarda los datos vectorizados en caché con nombre legible"""
        from scipy.sparse import save_npz
        
        if nombre_dataset_override:
            nombre_dataset = nombre_dataset_override
        elif hasattr(df, 'nombre_archivo') and df.nombre_archivo:
            nombre_dataset = self._get_nombre_dataset(df)
        else:
            nombre_dataset = "dataset_desconocido"
        
        fecha = datetime.now().strftime("%Y%m%d")
        limite_str = f"_{limite_resenas}" if limite_resenas else ""
        extra = f"mf{max_features}_ng{ngram[0]}{ngram[1]}" if ngram else "std"
        nombre_archivo = f"vector_{nombre_dataset}_{fecha}_{nivel}{limite_str}_{extra}.pkl"
        archivo_cache = os.path.join(self.carpeta_cache, nombre_archivo)
        
        archivo_sparse = archivo_cache.replace('.pkl', '_X.npz')
        save_npz(archivo_sparse, X)
        
        datos = {
            'y': y,
            'vectorizer': vectorizer,
            'max_features': max_features,
            'ngram': ngram,
            'archivo_sparse': archivo_sparse,
            'fecha': datetime.now().isoformat(),
            'nivel': nivel,
            'num_filas': len(df),
            'nombre_dataset': nombre_dataset
        }
        joblib.dump(datos, archivo_cache, compress=3)
        return archivo_cache
    
    def cargar_vectorizado(self, df, limite_resenas, nivel, max_features, ngram):
        """Carga los datos vectorizados desde caché"""
        from scipy.sparse import load_npz
        
        nombre_dataset = self._get_nombre_dataset(df)
        
        for archivo in os.listdir(self.carpeta_cache):
            if archivo.startswith(f"vector_{nombre_dataset}_") and archivo.endswith(".pkl"):
                archivo_cache = os.path.join(self.carpeta_cache, archivo)
                if os.path.exists(archivo_cache):
                    datos = joblib.load(archivo_cache)
                    X = load_npz(datos['archivo_sparse'])
                    return {
                        'X': X,
                        'y': datos['y'],
                        'vectorizer': datos['vectorizer']
                    }
        
        hash_id = self._get_hash_dataset(df, limite_resenas, nivel)
        hash_extra = hashlib.md5(f"{max_features}_{ngram}".encode()).hexdigest()[:8]
        archivo_hash = os.path.join(self.carpeta_cache, f"vector_{hash_id}_{hash_extra}.pkl")
        if os.path.exists(archivo_hash):
            datos = joblib.load(archivo_hash)
            X = load_npz(datos['archivo_sparse'])
            return {
                'X': X,
                'y': datos['y'],
                'vectorizer': datos['vectorizer']
            }
        return None
    
    def obtener_info_cache(self):
        """Obtiene información de todos los archivos de caché"""
        info = []
        for archivo in os.listdir(self.carpeta_cache):
            ruta = os.path.join(self.carpeta_cache, archivo)
            if os.path.isfile(ruta):
                tamaño = os.path.getsize(ruta)
                fecha_mod = os.path.getmtime(ruta)
                info.append({
                    'nombre': archivo,
                    'ruta': ruta,
                    'tamaño': tamaño,
                    'fecha': fecha_mod,
                    'tipo': 'preproc' if 'preproc' in archivo else 'vector' if 'vector' in archivo else 'otro'
                })
        return sorted(info, key=lambda x: x['fecha'], reverse=True)
