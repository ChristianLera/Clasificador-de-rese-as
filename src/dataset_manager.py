"""
src/dataset_manager.py
Gestión de múltiples datasets para entrenamiento y análisis
"""

import os
import pandas as pd
import time
from datetime import datetime


class DatasetManager:
    """Maneja la carga y combinación de múltiples datasets"""
    
    def __init__(self):
        self.datasets = []  # Lista de rutas de archivos
        self.df_combinado = None
        self.info_datasets = []
    
    def añadir_dataset(self, ruta_archivo):
        """Añade un dataset a la lista"""
        if ruta_archivo not in self.datasets:
            self.datasets.append(ruta_archivo)
            self._actualizar_info()
            return True
        return False
    
    def eliminar_dataset(self, indice):
        """Elimina un dataset de la lista"""
        if 0 <= indice < len(self.datasets):
            self.datasets.pop(indice)
            self._actualizar_info()
            return True
        return False
    
    def limpiar_todos(self):
        """Limpia todos los datasets"""
        self.datasets = []
        self.df_combinado = None
        self.info_datasets = []
    
    def _actualizar_info(self):
        """Actualiza la información de los datasets"""
        self.info_datasets = []
        total_filas = 0
        total_tamaño = 0
        
        for ruta in self.datasets:
            try:
                nombre = os.path.basename(ruta)
                tamaño = os.path.getsize(ruta) / (1024 * 1024)  # MB
                
                # Contar filas rápidamente (solo primeras líneas para estimar)
                with open(ruta, 'r', encoding='utf-8') as f:
                    primeras_lineas = f.read(1024 * 1024)  # 1MB
                    estimado = primeras_lineas.count('\n')
                
                self.info_datasets.append({
                    'ruta': ruta,
                    'nombre': nombre,
                    'tamaño_mb': tamaño,
                    'filas_estimadas': estimado * 10  # Estimación burda
                })
                total_tamaño += tamaño
                total_filas += estimado * 10
            except:
                pass
        
        self.total_filas = total_filas
        self.total_tamaño_mb = total_tamaño
    
    def combinar_datasets(self, log_callback=None):
        """Combina todos los datasets en un solo DataFrame"""
        if not self.datasets:
            return None
        
        if log_callback:
            log_callback(f"🔄 Combinando {len(self.datasets)} datasets...")
        
        dfs = []
        for i, ruta in enumerate(self.datasets):
            if log_callback:
                log_callback(f"   Cargando {i+1}/{len(self.datasets)}: {os.path.basename(ruta)}")
            
            try:
                df = pd.read_csv(ruta)
                dfs.append(df)
            except Exception as e:
                if log_callback:
                    log_callback(f"   ⚠️ Error en {ruta}: {e}")
        
        if dfs:
            self.df_combinado = pd.concat(dfs, ignore_index=True)
            self.df_combinado = self.df_combinado.drop_duplicates(subset=['reviewText'], keep='first')
            
            if log_callback:
                log_callback(f"✅ Combinación completada: {len(self.df_combinado):,} reseñas únicas")
            
            return self.df_combinado
        
        return None
    
    def guardar_combinado(self, ruta_destino):
        """Guarda el dataset combinado en un archivo"""
        if self.df_combinado is not None:
            self.df_combinado.to_csv(ruta_destino, index=False)
            return True
        return False
    
    def obtener_resumen(self):
        """Obtiene un resumen de los datasets cargados"""
        if not self.datasets:
            return "⚪ No hay datasets cargados"
        
        resumen = f"📁 {len(self.datasets)} archivos cargados\n"
        for info in self.info_datasets:
            resumen += f"   📄 {info['nombre']} ({info['tamaño_mb']:.1f} MB, ~{info['filas_estimadas']:,} reseñas)\n"
        
        if self.total_tamaño_mb > 1024:
            tamaño_str = f"{self.total_tamaño_mb / 1024:.2f} GB"
        else:
            tamaño_str = f"{self.total_tamaño_mb:.1f} MB"
        
        resumen += f"\n📊 Total: {tamaño_str}, ~{self.total_filas:,} reseñas"
        return resumen
    
    def calcular_tiempo_entrenamiento(self, nivel, limite_resenas=None):
        """
        Calcula el tiempo estimado de entrenamiento
        Basado en el nivel y número de reseñas
        """
        if not self.datasets:
            return None
        
        num_resenas = self.total_filas
        if limite_resenas and limite_resenas < num_resenas:
            num_resenas = limite_resenas
        
        # Tiempo base por nivel (en segundos por 1M reseñas)
        tiempos_base = {
            "Rápido": 300,    # 5 minutos por 1M reseñas
            "Preciso": 900,   # 15 minutos por 1M reseñas
            "Máximo": 1800,   # 30 minutos por 1M reseñas
            "Ensemble": 3600  # 60 minutos por 1M reseñas
        }
        
        tiempo_base = tiempos_base.get(nivel, 1800)
        tiempo_segundos = (num_resenas / 1000000) * tiempo_base
        
        return {
            'segundos': tiempo_segundos,
            'minutos': tiempo_segundos / 60,
            'horas': tiempo_segundos / 3600,
            'resenas': num_resenas,
            'nivel': nivel
        }
    
    def calcular_tiempo_analisis(self, num_resenas):
        """
        Calcula el tiempo estimado de análisis
        Aproximadamente 1000 reseñas por segundo
        """
        if not num_resenas:
            return None
        
        # Velocidad estimada: 1000 reseñas/segundo
        tiempo_segundos = num_resenas / 1000
        
        return {
            'segundos': tiempo_segundos,
            'minutos': tiempo_segundos / 60,
            'horas': tiempo_segundos / 3600,
            'resenas': num_resenas
        }
    
    def formatear_tiempo(self, segundos):
        """Formatea segundos en texto legible"""
        if segundos < 60:
            return f"{segundos:.0f} segundos"
        elif segundos < 3600:
            minutos = segundos / 60
            return f"{minutos:.0f} minutos"
        else:
            horas = segundos / 3600
            minutos = (segundos % 3600) / 60
            return f"{horas:.0f} horas {minutos:.0f} minutos"
