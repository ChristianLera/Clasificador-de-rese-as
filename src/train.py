"""
src/train.py
Entrenamiento de modelos con Ensemble Learning y actualización en cascada
Soporte para cancelación, límite de tiempo, y MEZCLA DE DATOS
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from .models_config import (
    get_configuracion_nivel, 
    get_niveles_a_actualizar,
    NOMBRE_MODELOS,
    MODELOS_NIVELES
)


class ModeloEntrenador:
    """Clase para entrenar y actualizar modelos con soporte de cancelación y mezcla de datos"""
    
    def __init__(self, carpeta_modelos):
        self.carpeta_modelos = carpeta_modelos
        os.makedirs(carpeta_modelos, exist_ok=True)
        self.cancelar = False
        self.timeout_limite = None
        self.tiempo_inicio = None
    
    def cancelar_entrenamiento(self):
        """Solicita cancelar el entrenamiento actual"""
        self.cancelar = True
    
    def reset_cancelar(self):
        """Resetea la bandera de cancelación"""
        self.cancelar = False
    
    def set_timeout(self, segundos):
        """Establece un límite de tiempo para el entrenamiento"""
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
    
    def _get_modelos_disponibles(self):
        """Diccionario de modelos disponibles para ensemble"""
        return {
            'lr': LogisticRegression(max_iter=1000, C=1.5, class_weight='balanced', random_state=42),
            'nb': MultinomialNB(alpha=1.0),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'svm': LinearSVC(max_iter=2000, C=1.0, dual=True, class_weight='balanced', random_state=42)
        }
    
    def _entrenar_modelos(self, X_train, y_train, modelos_a_usar, log_callback=None):
        """Entrena los modelos especificados con soporte de cancelación"""
        todos_modelos = self._get_modelos_disponibles()
        modelos_entrenados = {}
        
        for nombre in modelos_a_usar:
            if self.cancelar:
                if log_callback:
                    log_callback("⏹️ Cancelación solicitada durante entrenamiento...")
                break
            
            if nombre in todos_modelos:
                if log_callback:
                    log_callback(f"🔄 Entrenando {NOMBRE_MODELOS.get(nombre, nombre)}...")
                
                modelo = todos_modelos[nombre]
                modelo.fit(X_train, y_train)
                modelos_entrenados[nombre] = modelo
                
                if log_callback:
                    log_callback(f"✅ {NOMBRE_MODELOS.get(nombre, nombre)} listo")
        
        return modelos_entrenados
    
    def _prediccion_ensemble(self, modelos, X):
        """Realiza predicción combinando múltiples modelos"""
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
    
    def _actualizar_desde_superior(self, nivel_superior, nivel_inferior, modelos_superior, vectorizer, accuracy, log_callback=None):
        """
        Actualiza o CREA un modelo inferior desde uno superior (cascada)
        Si el modelo inferior no existe, lo CREA
        Si ya existe, lo ACTUALIZA
        """
        config_inferior = get_configuracion_nivel(nivel_inferior)
        
        # Filtrar solo los modelos que corresponden al nivel inferior
        modelos_filtrados = {}
        for nombre_modelo in config_inferior['modelos_a_usar']:
            if nombre_modelo in modelos_superior:
                modelos_filtrados[nombre_modelo] = modelos_superior[nombre_modelo]
        
        if not modelos_filtrados:
            if log_callback:
                log_callback(f"      ⚠️ No se pudo crear/actualizar {nivel_inferior}: faltan modelos")
            return False
        
        archivo_inferior = os.path.join(self.carpeta_modelos, config_inferior['nombre_archivo'])
        
        # Verificar si ya existe
        existe = os.path.exists(archivo_inferior)
        
        # Crear datos del modelo inferior
        datos_inferior = {
            'models': modelos_filtrados,
            'vectorizer': vectorizer,
            'accuracy': accuracy,
            'nivel': nivel_inferior,
            'modelos_usados': config_inferior['modelos_a_usar'],
            'max_features': config_inferior['max_features'],
            'ngram': config_inferior['ngram']
        }
        
        # Guardar (crea o sobreescribe)
        joblib.dump(datos_inferior, archivo_inferior)
        
        if log_callback:
            if existe:
                log_callback(f"      ✅ {nivel_inferior} ACTUALIZADO: {list(modelos_filtrados.keys())}")
            else:
                log_callback(f"      ✅ {nivel_inferior} CREADO: {list(modelos_filtrados.keys())}")
        
        return True
    
    def entrenar(self, nivel, df, callback_progreso=None, log_callback=None, modo_actualizacion='mezclar', limite_resenas=None):
        """
        Entrena un modelo con posibilidad de MEZCLAR con datos anteriores
        AHORA con: procesamiento por lotes, caché, paralelización
        """
        from .preprocess import Preprocesador
        from .cache_manager import CacheManager
        from multiprocessing import Pool
        import functools
        import tempfile
        
        self.reset_cancelar()
        config = get_configuracion_nivel(nivel)
        preprocesador = Preprocesador()
        
        # Crear carpeta de caché
        carpeta_cache = os.path.join(self.carpeta_modelos, "cache")
        cache = CacheManager(carpeta_cache)
        
        if log_callback:
            log_callback(f"🚀 Iniciando entrenamiento del modelo {nivel}")
            log_callback(f"📊 Configuración: {config['descripcion']}")
            if self.timeout_limite:
                log_callback(f"⏱️ Límite de tiempo: {self.timeout_limite//60} minutos")
        
        # ========================================
        # MEZCLAR CON DATOS ANTERIORES SI EXISTE y modo_actualizacion = 'mezclar'
        # ========================================
        
        archivo_datos_anteriores = os.path.join(self.carpeta_modelos, f"datos_compartidos.csv")
        df_anterior = None
        
        if modo_actualizacion == 'mezclar' and os.path.exists(archivo_datos_anteriores):
            if log_callback:
                log_callback(f"📂 Detectado dataset anterior. Cargando para mezclar...")
            
            try:
                df_anterior = pd.read_csv(archivo_datos_anteriores)
                if log_callback:
                    log_callback(f"📊 Datos anteriores: {len(df_anterior):,} reseñas")
            except Exception as e:
                if log_callback:
                    log_callback(f"⚠️ No se pudo cargar dataset anterior: {e}")
        
        # ========================================
        # COMBINAR DATOS ANTIGUOS + NUEVOS
        # ========================================
        
        if df_anterior is not None and modo_actualizacion == 'mezclar':
            if log_callback:
                log_callback(f"🔄 Mezclando {len(df_anterior):,} reseñas antiguas + {len(df):,} nuevas")
            
            # Combinar dataframes
            df_combinado = pd.concat([df_anterior, df], ignore_index=True)
            
            # Eliminar duplicados por texto
            df_combinado = df_combinado.drop_duplicates(subset=['reviewText'], keep='first')
            
            if log_callback:
                log_callback(f"✅ Total después de mezclar: {len(df_combinado):,} reseñas únicas")
            
            df = df_combinado
        
        # Guardar los datos usados para futuras mezclas
        df.to_csv(archivo_datos_anteriores, index=False)
        
        if log_callback:
            log_callback(f"📊 Usando {len(df):,} reseñas para entrenamiento")
        
        # ========================================
        # APLICAR LÍMITE DE RESEÑAS
        # ========================================
        
        if limite_resenas and len(df) > limite_resenas:
            if log_callback:
                log_callback(f"📊 Limitando a {limite_resenas:,} reseñas (de {len(df):,})")
            df = df.sample(n=limite_resenas, random_state=42)
            if log_callback:
                log_callback(f"📊 Usando {len(df):,} reseñas después del límite")
        
        # ========================================
        # CACHÉ DE PREPROCESAMIENTO
        # ========================================
        
        usar_cache = modo_actualizacion == 'mezclar'
        
        textos_procesados = None
        y = None
        
        if usar_cache and cache.existe_preprocesado(df, limite_resenas, nivel):
            if log_callback:
                log_callback(f"⚡ Usando caché de preprocesamiento...")
            
            datos_cache = cache.cargar_preprocesado(df, limite_resenas, nivel)
            textos_procesados = datos_cache['textos_procesados']
            y = datos_cache['etiquetas']
            
            if log_callback:
                log_callback(f"✅ Caché cargado: {len(textos_procesados):,} reseñas preprocesadas")
        
        if textos_procesados is None:
            # ========================================
            # PREPROCESAMIENTO NORMAL
            # ========================================
            
            textos = df['reviewText'].dropna().tolist()
            total_textos = len(textos)
            
            # Preparar etiquetas
            df_temp = df.copy()
            df_temp['overall'] = pd.to_numeric(df_temp['overall'], errors='coerce')
            df_temp = df_temp.dropna()
            df_temp = df_temp[df_temp['overall'] != 3]
            etiquetas_temp = (df_temp['overall'] >= 4).astype(int).values
            
            if log_callback:
                log_callback(f"📝 Preprocesando {total_textos:,} reseñas...")
            
            textos_procesados = []
            lote_size = 5000
            
            for i in range(0, total_textos, lote_size):
                if self.cancelar:
                    if log_callback:
                        log_callback("⏹️ Cancelación solicitada durante preprocesamiento")
                    return 0, None
                
                if self.tiempo_excedido():
                    if log_callback:
                        log_callback(f"⏰ Tiempo límite alcanzado")
                    return 0, None
                
                lote = textos[i:i+lote_size]
                for texto in lote:
                    procesado, _ = preprocesador.preprocesar(texto)
                    textos_procesados.append(procesado)
                
                if callback_progreso:
                    porcentaje = ((i + lote_size) / total_textos) * 30
                    callback_progreso(porcentaje, f"Preprocesando {i+len(lote)}/{total_textos}")
            
            y = etiquetas_temp[:len(textos_procesados)]
            
            # Guardar en caché
            if usar_cache:
                if log_callback:
                    log_callback(f"💾 Guardando preprocesado en caché...")
                cache.guardar_preprocesado(df, limite_resenas, nivel, textos_procesados, y)
        
        # ========================================
        # CACHÉ DE VECTORIZACIÓN
        # ========================================
        
        X = None
        vectorizer = None
        
        if usar_cache and cache.existe_vectorizado(df, limite_resenas, nivel, config['max_features'], config['ngram']):
            if log_callback:
                log_callback(f"⚡ Usando caché de vectorización...")
            
            datos_cache = cache.cargar_vectorizado(df, limite_resenas, nivel, config['max_features'], config['ngram'])
            X = datos_cache['X']
            y = datos_cache['y']
            vectorizer = datos_cache['vectorizer']
            
            if log_callback:
                log_callback(f"✅ Caché de vectorización cargado")
        
        if X is None:
            # ========================================
            # VECTORIZACIÓN NORMAL
            # ========================================
            
            if log_callback:
                log_callback(f"📊 Vectorizando textos...")
            
            vectorizer = TfidfVectorizer(
                max_features=config['max_features'],
                ngram_range=config['ngram'],
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
            
            if callback_progreso:
                callback_progreso(35, "Vectorizando...")
            
            X = vectorizer.fit_transform(textos_procesados)
            
            if log_callback:
                log_callback(f"✅ Vectorización completada: {X.shape[0]} documentos, {X.shape[1]} características")
            
            # Guardar en caché
            if usar_cache:
                if log_callback:
                    log_callback(f"💾 Guardando vectorización en caché...")
                cache.guardar_vectorizado(df, limite_resenas, nivel, config['max_features'], config['ngram'], X, y, vectorizer)        

        # ========================================
        # DIVIDIR DATOS
        # ========================================
        
        if callback_progreso:
            callback_progreso(40, "Dividiendo datos...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if log_callback:
            log_callback(f"🎯 Datos listos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
            log_callback(f"🧠 Entrenando modelos ensemble...")
        
        if callback_progreso:
            callback_progreso(45, "Entrenando modelos...")
        
        # ========================================
        # ENTRENAR MODELOS
        # ========================================
        
        modelos_entrenados = self._entrenar_modelos(X_train, y_train, config['modelos_a_usar'], log_callback)
        
        if self.cancelar:
            if log_callback:
                log_callback("⏹️ Entrenamiento cancelado")
            return 0, None
        
        if callback_progreso:
            callback_progreso(90, "Evaluando modelo...")
        
        # Evaluar
        y_pred, confianza = self._prediccion_ensemble(modelos_entrenados, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if log_callback:
            log_callback(f"📈 Precisión del modelo {nivel}: {accuracy:.1%}")
        
        # Guardar el modelo principal
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
        
        if log_callback:
            log_callback(f"💾 Modelo {nivel} guardado: {archivo_principal}")
        
        # ========================================
        # ACTUALIZACIÓN EN CASCADA (CREAR O ACTUALIZAR)
        # ========================================
        
        if log_callback:
            log_callback(f"🔄 Actualizando/Creando modelos en cascada...")
        
        # Obtener qué niveles se deben actualizar/crear
        niveles_a_actualizar = get_niveles_a_actualizar(nivel)
        
        for nivel_inferior in niveles_a_actualizar:
            if log_callback:
                log_callback(f"   → Procesando {nivel_inferior}...")
            
            self._actualizar_desde_superior(
                nivel_superior=nivel,
                nivel_inferior=nivel_inferior,
                modelos_superior=modelos_entrenados,
                vectorizer=vectorizer,
                accuracy=accuracy,
                log_callback=log_callback
            )
        
        if callback_progreso:
            callback_progreso(100, f"✅ Modelo {nivel} completado ({accuracy:.1%})")
        
        if log_callback:
            log_callback(f"🎉 Entrenamiento completado. Precisión: {accuracy:.1%}")
        
        # Limpiar archivo temporal
        try:
            os.unlink(temp_dataset)
        except:
            pass
        
        return accuracy, modelos_entrenados
    
    def guardar_modelo(self, nivel, modelos, vectorizer, accuracy, config, log_callback=None):
        """Guarda un modelo en disco"""
        archivo = os.path.join(self.carpeta_modelos, config['nombre_archivo'])
        
        datos_modelo = {
            'models': modelos,
            'vectorizer': vectorizer,
            'accuracy': accuracy,
            'nivel': nivel,
            'modelos_usados': config['modelos_a_usar'],
            'max_features': config['max_features'],
            'ngram': config['ngram']
        }
        
        joblib.dump(datos_modelo, archivo)
        
        if log_callback:
            log_callback(f"💾 Modelo guardado: {archivo}")
    
    def cargar_modelo(self, nivel):
        """Carga un modelo guardado"""
        config = get_configuracion_nivel(nivel)
        archivo = os.path.join(self.carpeta_modelos, config['nombre_archivo'])
        
        if os.path.exists(archivo):
            try:
                return joblib.load(archivo)
            except:
                return None
        return None
    
    def listar_modelos(self):
        """Lista los modelos guardados"""
        modelos_info = {}
        for nivel, config in MODELOS_NIVELES.items():
            archivo = os.path.join(self.carpeta_modelos, config['nombre_archivo'])
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
                except:
                    info['accuracy'] = 0
                    info['modelos'] = []
            
            modelos_info[nivel] = info
        
        return modelos_info
