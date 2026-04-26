"""
src/cache_preferences.py
Guarda las preferencias del usuario sobre el uso de caché
"""

import os
import json


class CachePreferences:
    def __init__(self, carpeta_modelos):
        self.archivo_prefs = os.path.join(carpeta_modelos, "cache_preferences.json")
        self.preferences = self._cargar()
    
    def _cargar(self):
        """Carga las preferencias desde el archivo"""
        if os.path.exists(self.archivo_prefs):
            try:
                with open(self.archivo_prefs, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _guardar(self):
        """Guarda las preferencias en el archivo"""
        try:
            with open(self.archivo_prefs, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, indent=2)
        except:
            pass
    
    def get_preferencia(self, dataset_hash):
        """Obtiene la preferencia para un dataset específico"""
        return self.preferences.get(dataset_hash, None)
    
    def set_preferencia(self, dataset_hash, valor):
        """Guarda la preferencia para un dataset específico"""
        self.preferences[dataset_hash] = valor
        self._guardar()
    
    def eliminar_preferencia(self, dataset_hash):
        """Elimina la preferencia para un dataset"""
        if dataset_hash in self.preferences:
            del self.preferences[dataset_hash]
            self._guardar()
