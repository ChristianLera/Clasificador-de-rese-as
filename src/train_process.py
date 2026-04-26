"""
src/train_process.py
Proceso independiente para entrenamiento (no bloquea Tkinter)
CON SOPORTE DE CACHÉ - VERSIÓN CORREGIDA
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from multiprocessing import Process, Queue
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Añadir src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import Preprocesador
from models_config import get_configuracion_nivel, get_niveles_a_actualizar, NOMBRE_MODELOS, MODELOS_NIVELES
from cache_preferences import CachePreferences
from cache_manager import CacheManager   # <--- USAR LA CLASE EXTERNA


class ModeloEntrenadorProceso:
    """Versión del entrenador para usar en proceso separado"""
    
    def __init__(self, carpeta_modelos, queue):
        self.carpeta_modelos = carpeta_modelos
        self.queue = queue
        self.nombre_archivo = None
        self.cancelar = False
        self.timeout_limite = None
        self.tiempo_inicio = None
        os.makedirs(carpeta_modelos, exist_ok=True)
    
    def enviar_progreso(self, porcentaje, mensaje):
        """Envía progreso a la interfaz"""
        self.queue.put(('progreso', porcentaje, mensaje))
    
    def enviar_log(self, mensaje):
        """Envía log a la interfaz"""
        self.queue.put(('log', mensaje))
    
    def set_timeout(self, segundos):
        self.timeout_limite = segundos
        self.tiempo_inicio = time.time()
    
    def _get_modelos_disponibles(self):
        return {
            'lr': LogisticRegression(max_iter=1000, C=1.5, class_weight='balanced', random_state=42),
            'nb': MultinomialNB(alpha=1.0),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'svm': LinearSVC(max_iter=2000, C=1.0, dual=True, class_weight='balanced', random_state=42)
        }
    
    def _entrenar_modelos(self, X_train, y_train, modelos_a_usar):
        todos_modelos = self._get_modelos_disponibles()
        modelos_entrenados = {}
        
        for nombre in modelos_a_usar:
            if nombre in todos_modelos:
                self.enviar_log(f"🔄 Entrenando {NOMBRE_MODELOS.get(nombre, nombre)}...")
                modelo = todos_modelos[nombre]
                modelo.fit(X_train, y_train)
                modelos_entrenados[nombre] = modelo
                self.enviar_log(f"✅ {NOMBRE_MODELOS.get(nombre, nombre)} listo")
        
        return modelos_entrenados
    
    def _prediccion_ensemble(self, modelos, X):
        predicciones = []
        probabilidades = []
        
        for nombre, modelo in modelos.items():
            try:
                if hasattr(modelo, 'predict_proba'):
                    proba = modelo.predict_proba(X)[:, 1]
                    probabilidades.append(proba)
                    predicciones.append((proba >= 0.5).astype(int))
                else:
                    pred = modelo.predict(X)
                    predicciones.append(pred)
            except:
                continue
        
        if not predicciones:
            return np.zeros(len(X)), np.zeros(len(X))
        
        predicciones_stack = np.array(predicciones)
        prediccion_final = (np.sum(predicciones_stack, axis=0) >= len(predicciones) / 2).astype(int)
        
        if probabilidades:
            confianza = np.mean(probabilidades, axis=0) * 100
        else:
            confianza = np.full(len(X), 50.0)
        
        return prediccion_final, confianza
    
    def entrenar(self, nivel, df, modo_actualizacion='mezclar', limite_resenas=None):
        """Entrena el modelo con soporte de caché"""
        
        config = get_configuracion_nivel(nivel)
        preprocesador = Preprocesador()
        
        self.enviar_log(f"🚀 Iniciando entrenamiento del modelo {nivel}")
        self.enviar_log(f"📊 Configuración: {config['descripcion']}")
        
        # ========================================
        # ASIGNAR NOMBRE DEL ARCHIVO AL DATAFRAME
        # ========================================
        if self.nombre_archivo:
            df.nombre_archivo = self.nombre_archivo
            self.enviar_log(f"📛 DEBUG: nombre_archivo asignado = {self.nombre_archivo}")
        else:
            self.enviar_log(f"⚠️ DEBUG: nombre_archivo es None en el entrenador")
        
        # ========================================
        # MEZCLAR CON DATOS ANTERIORES
        # ========================================
        
        archivo_datos_anteriores = os.path.join(self.carpeta_modelos, "datos_compartidos.csv")
        
        if modo_actualizacion == 'mezclar' and os.path.exists(archivo_datos_anteriores):
            self.enviar_log(f"📂 Cargando datos anteriores...")
            try:
                df_anterior = pd.read_csv(archivo_datos_anteriores)
                self.enviar_log(f"📊 Datos anteriores: {len(df_anterior):,} reseñas")
                df_combinado = pd.concat([df_anterior, df], ignore_index=True)
                df_combinado = df_combinado.drop_duplicates(subset=['reviewText'], keep='first')
                self.enviar_log(f"✅ Total después de mezclar: {len(df_combinado):,} reseñas")
                df = df_combinado
            except Exception as e:
                self.enviar_log(f"⚠️ Error al mezclar: {e}")
        
        # Guardar datos para futuras mezclas
        df.to_csv(archivo_datos_anteriores, index=False)
        self.enviar_log(f"📊 Usando {len(df):,} reseñas para entrenamiento")
        
        # Aplicar límite
        if limite_resenas and len(df) > limite_resenas:
            self.enviar_log(f"📊 Limitando a {limite_resenas:,} reseñas")
            df = df.sample(n=limite_resenas, random_state=42)
            self.enviar_log(f"📊 Usando {len(df):,} reseñas después del límite")
        
        # ========================================
        # PREPARAR TEXTOS Y ETIQUETAS
        # ========================================
        
        # Obtener textos no nulos
        textos = df['reviewText'].dropna().tolist()
        total_textos = len(textos)
        
        if total_textos == 0:
            self.enviar_log(f"❌ No hay reseñas válidas para entrenar")
            return 0
        
        # Preparar etiquetas correlacionadas con los textos
        # Usamos el mismo índice para asegurar correspondencia
        indices_validos = df['reviewText'].notna()
        df_temp = df[indices_validos].copy()
        
        df_temp['overall'] = pd.to_numeric(df_temp['overall'], errors='coerce')
        df_temp = df_temp.dropna(subset=['overall'])
        df_temp = df_temp[df_temp['overall'] != 3]
        
        # Asegurar que tenemos las mismas filas
        etiquetas_completas = (df_temp['overall'] >= 4).astype(int).values
        
        # Verificar que coincidan longitudes
        if len(textos) != len(etiquetas_completas):
            self.enviar_log(f"⚠️ Discrepancia inicial: textos={len(textos)}, etiquetas={len(etiquetas_completas)}")
            min_len = min(len(textos), len(etiquetas_completas))
            textos = textos[:min_len]
            etiquetas_completas = etiquetas_completas[:min_len]
            self.enviar_log(f"📊 Ajustado a: textos={len(textos)}, etiquetas={len(etiquetas_completas)}")
        
        # ========================================
        # CACHÉ DE PREPROCESAMIENTO
        # ========================================
        
        carpeta_cache = os.path.join(self.carpeta_modelos, "cache")
        cache = CacheManager(carpeta_cache)
        
        usar_cache = modo_actualizacion == 'mezclar'
        
        textos_procesados = None
        y = None
        
        self.enviar_log(f"🔍 Verificando caché para {len(textos):,} reseñas...")
        
        if usar_cache and cache.existe_preprocesado(df, limite_resenas, nivel):
            self.enviar_log(f"⚡ Usando caché de preprocesamiento...")
            datos_cache = cache.cargar_preprocesado(df, limite_resenas, nivel)
            if datos_cache and 'textos_procesados' in datos_cache and 'etiquetas' in datos_cache:
                textos_procesados = datos_cache['textos_procesados']
                y = datos_cache['etiquetas']
                
                # Verificar consistencia
                if len(textos_procesados) != len(y):
                    self.enviar_log(f"⚠️ Error en caché: textos={len(textos_procesados)}, etiquetas={len(y)}")
                    textos_procesados = None
                    y = None
                else:
                    self.enviar_log(f"✅ Caché cargado: {len(textos_procesados):,} reseñas preprocesadas")
        
        if textos_procesados is None:
            # ========================================
            # PREPROCESAMIENTO NORMAL
            # ========================================
            
            self.enviar_log(f"📝 Preprocesando {total_textos:,} reseñas...")
            
            textos_procesados = []
            for i, texto in enumerate(textos):
                if i % 10000 == 0:
                    self.enviar_progreso((i / total_textos) * 30, f"Preprocesando {i}/{total_textos}")
                
                if texto and isinstance(texto, str):
                    procesado, _ = preprocesador.preprocesar(texto)
                    textos_procesados.append(procesado if procesado else "")
                else:
                    textos_procesados.append("")
            
            # Asignar etiquetas
            y = etiquetas_completas[:len(textos_procesados)]
            
            # Guardar en caché
            if usar_cache:
                self.enviar_log(f"💾 Guardando preprocesado en caché...")
                nombre_override = self.nombre_archivo.replace('.csv', '') if self.nombre_archivo else None
                cache.guardar_preprocesado(df, limite_resenas, nivel, textos_procesados, y, nombre_override)
        
        # ========================================
        # CACHÉ DE VECTORIZACIÓN
        # ========================================
        
        X = None
        vectorizer = None
        
        if usar_cache and textos_procesados and len(textos_procesados) > 0:
            if cache.existe_vectorizado(df, limite_resenas, nivel, config['max_features'], config['ngram']):
                self.enviar_log(f"⚡ Usando caché de vectorización...")
                datos_cache = cache.cargar_vectorizado(df, limite_resenas, nivel, config['max_features'], config['ngram'])
                if datos_cache:
                    X = datos_cache['X']
                    y = datos_cache['y']
                    vectorizer = datos_cache['vectorizer']
                    self.enviar_log(f"✅ Caché de vectorización cargado")
        
        if X is None and textos_procesados and len(textos_procesados) > 0:
            # ========================================
            # VECTORIZACIÓN NORMAL
            # ========================================
            
            self.enviar_log(f"📊 Vectorizando textos...")
            self.enviar_progreso(35, "Vectorizando...")
            
            vectorizer = TfidfVectorizer(
                max_features=config['max_features'],
                ngram_range=config['ngram'],
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
            X = vectorizer.fit_transform(textos_procesados)
            
            self.enviar_log(f"✅ Vectorización completada: {X.shape[0]} documentos, {X.shape[1]} características")
            
            # Guardar en caché
            if usar_cache:
                self.enviar_log(f"💾 Guardando vectorización en caché...")
                nombre_override = self.nombre_archivo.replace('.csv', '') if self.nombre_archivo else None
                cache.guardar_vectorizado(df, limite_resenas, nivel, config['max_features'], config['ngram'], X, y, vectorizer, nombre_override)
        
        # Verificar que tenemos datos para entrenar
        if X is None or y is None or X.shape[0] == 0:
            self.enviar_log(f"❌ No hay datos para entrenar")
            return 0
        
        # ========================================
        # DIVIDIR DATOS
        # ========================================
        
        self.enviar_progreso(40, "Dividiendo datos...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.enviar_log(f"🎯 Datos listos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
        self.enviar_log(f"🧠 Entrenando modelos ensemble...")
        
        self.enviar_progreso(45, "Entrenando modelos...")
        
        # ========================================
        # ENTRENAR MODELOS
        # ========================================
        
        modelos_entrenados = self._entrenar_modelos(X_train, y_train, config['modelos_a_usar'])
        
        self.enviar_progreso(90, "Evaluando modelo...")
        
        # Evaluar
        y_pred, _ = self._prediccion_ensemble(modelos_entrenados, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.enviar_log(f"📈 Precisión del modelo {nivel}: {accuracy:.1%}")
        
        # ========================================
        # GUARDAR MODELO PRINCIPAL
        # ========================================
        
        archivo_principal = os.path.join(self.carpeta_modelos, config['nombre_archivo'])
        datos_principal = {
            'models': modelos_entrenados,
            'vectorizer': vectorizer,
            'accuracy': accuracy,
            'nivel': nivel,
            'modelos_usados': config['modelos_a_usar'],
            'max_features': config['max_features'],
            'ngram': config['ngram']
        }
        joblib.dump(datos_principal, archivo_principal)
        self.enviar_log(f"💾 Modelo {nivel} guardado: {archivo_principal}")
        
        # ========================================
        # ACTUALIZACIÓN EN CASCADA
        # ========================================
        
        self.enviar_log(f"🔄 Actualizando modelos en cascada...")
        niveles_a_actualizar = get_niveles_a_actualizar(nivel)
        
        for nivel_inferior in niveles_a_actualizar:
            self.enviar_log(f"   → Actualizando {nivel_inferior}...")
            config_inf = get_configuracion_nivel(nivel_inferior)
            modelos_filtrados = {m: modelos_entrenados[m] for m in config_inf['modelos_a_usar'] if m in modelos_entrenados}
            
            if modelos_filtrados:
                archivo_inf = os.path.join(self.carpeta_modelos, config_inf['nombre_archivo'])
                joblib.dump({
                    'models': modelos_filtrados,
                    'vectorizer': vectorizer,
                    'accuracy': accuracy,
                    'nivel': nivel_inferior,
                    'modelos_usados': config_inf['modelos_a_usar'],
                    'max_features': config_inf['max_features'],
                    'ngram': config_inf['ngram']
                }, archivo_inf)
                self.enviar_log(f"      ✅ {nivel_inferior} actualizado")
        
        self.enviar_progreso(100, f"✅ Completado: {accuracy:.1%}")
        self.enviar_log(f"🎉 Entrenamiento completado. Precisión: {accuracy:.1%}")
        
        return accuracy


def entrenar_proceso(queue, carpeta_modelos, nivel, archivo_datos, modo_actualizacion, limite_resenas, timeout_minutos, nombre_archivo):
    """
    Función que se ejecuta en un proceso separado
    """
    try:
        # Cargar datos
        df = pd.read_csv(archivo_datos)
        
        # Crear entrenador
        entrenador = ModeloEntrenadorProceso(carpeta_modelos, queue)
        entrenador.nombre_archivo = nombre_archivo
        
        if timeout_minutos:
            entrenador.set_timeout(timeout_minutos * 60)
        
        # Entrenar
        accuracy = entrenador.entrenar(nivel, df, modo_actualizacion, limite_resenas)
        
        # Enviar resultado final
        queue.put(('completado', accuracy))
        
    except Exception as e:
        import traceback
        queue.put(('error', f"{str(e)}\n{traceback.format_exc()}"))
