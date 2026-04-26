
"""
src/preprocess.py
Preprocesamiento de texto: limpieza, stopwords, detección de idioma
"""

import re
import nltk
import numpy as np

# Descargar recursos NLTK (solo primera vez)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)


class DetectorIdioma:
    """Detecta si un texto está en español o inglés"""
    
    def __init__(self):
        # Palabras características del español
        self.palabras_es = set([
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'pero', 'porque',
            'muy', 'bueno', 'malo', 'producto', 'calidad', 'precio', 'recomiendo',
            'excelente', 'terrible', 'buena', 'mala', 'gustó', 'encantó', 'horrible',
            'pésimo', 'decepcionante', 'maravilloso', 'funciona', 'comprar', 'envío'
        ])
        
        # Palabras características del inglés
        self.palabras_en = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'because', 'very', 'good', 'bad',
            'product', 'quality', 'price', 'recommend', 'excellent', 'terrible',
            'great', 'awesome', 'amazing', 'disappointed', 'horrible', 'wonderful',
            'works', 'buy', 'shipping', 'love', 'hate', 'best', 'worst'
        ])
    
    def detectar(self, texto):
        """Detecta el idioma del texto"""
        if not isinstance(texto, str) or not texto.strip():
            return 'en'
        
        texto_limpio = texto.lower()
        palabras = set(re.findall(r'\b\w+\b', texto_limpio))
        
        coincidencias_es = len(palabras & self.palabras_es)
        coincidencias_en = len(palabras & self.palabras_en)
        
        if coincidencias_es == 0 and coincidencias_en == 0:
            return 'en'
        
        return 'es' if coincidencias_es > coincidencias_en else 'en'


class Preprocesador:
    """Preprocesa textos para entrenamiento y predicción"""
    
    def __init__(self):
        self.detector = DetectorIdioma()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        
        try:
            self.stopwords_es = set(nltk.corpus.stopwords.words('spanish'))
        except:
            self.stopwords_es = set()
        
        self.stopwords_en = set(nltk.corpus.stopwords.words('english'))
    
    def limpiar(self, texto):
        """Limpia el texto: minúsculas, elimina puntuación, números, espacios extra"""
        if not isinstance(texto, str):
            return ""
        
        texto = texto.lower()
        texto = re.sub(r'http\S+|www\S+|https\S+', '', texto)
        texto = re.sub(r'@\w+|#\w+', '', texto)
        texto = re.sub(r'\d+', '', texto)
        texto = re.sub(r'[^\w\s]', '', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        return texto
    
    def eliminar_stopwords(self, texto, idioma):
        """Elimina stopwords según el idioma"""
        palabras = texto.split()
        
        if idioma == 'es':
            stopwords = self.stopwords_es
        else:
            stopwords = self.stopwords_en
        
        palabras_filtradas = [
            p for p in palabras 
            if p not in stopwords and len(p) > 2
        ]
        
        return ' '.join(palabras_filtradas)
    
    def lematizar(self, texto):
        """Aplica lematización (solo para inglés)"""
        palabras = texto.split()
        palabras_lematizadas = [self.lemmatizer.lemmatize(p) for p in palabras]
        return ' '.join(palabras_lematizadas)
    
    def preprocesar(self, texto):
        """Pipeline completo de preprocesamiento"""
        if not texto or not isinstance(texto, str):
            return "", 'en'
        
        idioma = self.detector.detectar(texto)
        texto = self.limpiar(texto)
        
        if not texto:
            return "", idioma
        
        texto = self.eliminar_stopwords(texto, idioma)
        
        if idioma == 'en' and texto:
            texto = self.lematizar(texto)
        
        return texto, idioma
