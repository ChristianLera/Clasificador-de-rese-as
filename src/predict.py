"""
src/predict.py
Predicción con modelos guardados y soporte de cancelación
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from .preprocess import Preprocesador
from .models_config import MODELOS_NIVELES
import warnings
warnings.filterwarnings('ignore')


class Predictor:
    """Clase para hacer predicciones con modelos guardados"""
    
    def __init__(self, carpeta_modelos):
        self.carpeta_modelos = carpeta_modelos
        self.preprocesador = Preprocesador()
        self.modelo_actual = None
        self.nivel_actual = None
        self.vectorizer_actual = None
        self.accuracy_actual = 0
        self.cancelar = False
        self.timeout_limite = None
        self.tiempo_inicio = None
    
    def cancelar_analisis(self):
        """Solicita cancelar el análisis actual"""
        self.cancelar = True
    
    def reset_cancelar(self):
        """Resetea la bandera de cancelación"""
        self.cancelar = False
    
    def set_timeout(self, segundos):
        """Establece un límite de tiempo para el análisis"""
        self.timeout_limite = segundos
        self.tiempo_inicio = time.time()
    
    def tiempo_restante(self):
        """Devuelve el tiempo restante en segundos o None si no hay límite"""
        if self.timeout_limite is None or self.tiempo_inicio is None:
            return None
        transcurrido = time.time() - self.tiempo_inicio
        restante = self.timeout_limite - transcurrido
        return max(0, restante)
    
    def tiempo_excedido(self):
        """Verifica si se ha excedido el tiempo límite"""
        if self.timeout_limite is None:
            return False
        return (time.time() - self.tiempo_inicio) > self.timeout_limite
    
    def cargar_modelo(self, nivel):
        """Carga un modelo específico"""
        config = MODELOS_NIVELES.get(nivel)
        if not config:
            return False
        
        archivo = os.path.join(self.carpeta_modelos, config['nombre_archivo'])
        
        if not os.path.exists(archivo):
            return False
        
        try:
            data = joblib.load(archivo)
            self.modelo_actual = data['models']
            self.vectorizer_actual = data['vectorizer']
            self.nivel_actual = nivel
            self.accuracy_actual = data.get('accuracy', 0)
            return True
        except Exception as e:
            print(f"Error cargando modelo {nivel}: {e}")
            return False
    
    def predecir(self, texto):
        """Predice el sentimiento de una reseña"""
        if self.modelo_actual is None:
            return None
        
        texto_procesado, idioma = self.preprocesador.preprocesar(texto)
        
        if not texto_procesado:
            return {
                'sentimiento': 'Neutral',
                'confianza': 0,
                'prob_positiva': 0.5,
                'prob_negativa': 0.5,
                'idioma': 'Español' if idioma == 'es' else 'Inglés'
            }
        
        vector = self.vectorizer_actual.transform([texto_procesado])
        
        votos = []
        probabilidades = []
        
        for nombre, modelo in self.modelo_actual.items():
            try:
                if hasattr(modelo, 'predict_proba'):
                    proba = modelo.predict_proba(vector)[0][1]
                    probabilidades.append(proba)
                    votos.append(1 if proba >= 0.5 else 0)
                else:
                    pred = modelo.predict(vector)[0]
                    votos.append(pred)
            except:
                continue
        
        if not votos:
            return None
        
        prediccion_final = 1 if sum(votos) > len(votos) / 2 else 0
        
        if probabilidades:
            confianza = sum(probabilidades) / len(probabilidades) * 100
        else:
            confianza = 50.0
        
        return {
            'sentimiento': 'Positiva' if prediccion_final == 1 else 'Negativa',
            'confianza': confianza,
            'prob_positiva': confianza / 100,
            'prob_negativa': 1 - (confianza / 100),
            'idioma': 'Español' if idioma == 'es' else 'Inglés',
            'votos': sum(votos),
            'total_modelos': len(votos),
            'texto_procesado': texto_procesado
        }
    
    def predecir_lote(self, textos, callback_progreso=None, log_callback=None):
        """
        Predice múltiples reseñas con soporte de cancelación
        """
        self.reset_cancelar()
        resultados = []
        total = len(textos)
        
        if log_callback:
            log_callback(f"📊 Iniciando análisis de {total:,} reseñas")
            if self.timeout_limite:
                log_callback(f"⏱️ Límite de tiempo: {self.timeout_limite//60} minutos")
        
        for i, texto in enumerate(textos):
            if self.cancelar:
                if log_callback:
                    log_callback("⏹️ Análisis cancelado por el usuario")
                break
            
            if self.tiempo_excedido():
                if log_callback:
                    log_callback(f"⏰ Tiempo límite alcanzado ({self.timeout_limite//60} minutos)")
                break
            
            if texto and texto.strip():
                r = self.predecir(texto)
                if r:
                    r['texto_original'] = texto
                    resultados.append(r)
            
            if callback_progreso and (i + 1) % 10 == 0:
                porcentaje = ((i + 1) / total) * 100
                callback_progreso(porcentaje, f"Analizando {i+1}/{total}")
            
            if log_callback and (i + 1) % 100 == 0:
                log_callback(f"📝 Procesadas {i+1}/{total} reseñas")
        
        positivas = sum(1 for r in resultados if r['sentimiento'] == 'Positiva')
        
        if log_callback:
            log_callback(f"✅ Análisis completado: {len(resultados)} reseñas")
            log_callback(f"📊 Resultados: {positivas} positivas, {len(resultados)-positivas} negativas")
        
        return {
            'total': len(resultados),
            'positivas': positivas,
            'negativas': len(resultados) - positivas,
            'porcentaje_positivas': positivas / len(resultados) * 100 if resultados else 0,
            'confianza_promedio': sum(r['confianza'] for r in resultados) / len(resultados) if resultados else 0,
            'tiempo_cancelado': self.cancelar,
            'tiempo_excedido': self.tiempo_excedido(),
            'resultados': resultados
        }
