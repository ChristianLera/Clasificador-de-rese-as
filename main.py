"""
main.py
Punto de entrada principal del programa
VERSIÓN CORREGIDA - Carga en hilos, interfaz fluida
"""

import os
import sys
import warnings

# SILENCIAR TODOS LOS WARNINGS (INCLUIDOS LOS DE JOBLIB)
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Redirigir stderr para capturar warnings de joblib
import contextlib
sys.stderr = open(os.devnull, 'w')

import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import joblib
from multiprocessing import Process, Queue
from src.train_process import entrenar_proceso

# Añadir src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import Preprocesador
from src.train import ModeloEntrenador
from src.predict import Predictor
from src.models_config import OPCIONES_PRECISION, MODELOS_NIVELES, OPCIONES_TIMEOUT, TEXTO_AYUDA, OPCIONES_LIMITE_RESENAS
from src.utils import cargar_dataset, listar_modelos_guardados, obtener_modelo_por_defecto
from src.dataset_manager import DatasetManager

import traceback
import sys

def excepthook(exc_type, exc_value, exc_traceback):
    """Captura cualquier excepción no manejada y la muestra en un messagebox"""
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print("ERROR CAPTURADO:", error_msg)
    messagebox.showerror("Error inesperado", f"Error:\n{exc_value}\n\nRevisa la consola para más detalles.")
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = excepthook

# Colores - Tema claro profesional
COLORES = {
    'bg': '#f5f5f5',
    'fg': '#333333',
    'positive': '#27ae60',
    'negative': '#e74c3c',
    'accent': '#2980b9',
    'border': '#dddddd',
    'secondary': '#666666',
    'help': '#3498db',
    'warning': '#f39c12',
    'log_bg': '#ffffff'
}

# Rutas
CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATASETS = os.path.join(CARPETA_ACTUAL, "data")
CARPETA_MODELOS = os.path.join(CARPETA_ACTUAL, "models")
CARPETA_LOGS = os.path.join(CARPETA_ACTUAL, "logs")

# Crear carpetas
os.makedirs(CARPETA_DATASETS, exist_ok=True)
os.makedirs(CARPETA_MODELOS, exist_ok=True)
os.makedirs(CARPETA_LOGS, exist_ok=True)

ARCHIVO_DATASET = os.path.join(CARPETA_DATASETS, "dataset_actual.csv")


class ToolTip:
    def __init__(self, widget, texto):
        self.widget = widget
        self.texto = texto
        self.tooltip = None
        widget.bind('<Enter>', self.mostrar)
        widget.bind('<Leave>', self.ocultar)
    
    def mostrar(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip, text=self.texto, justify='left',
                        background='#ffffe0', foreground='#333333', relief='solid', 
                        borderwidth=1, font=('Segoe UI', 9), wraplength=300)
        label.pack()
    
    def ocultar(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class VentanaLog:
    def __init__(self, parent):
        self.parent = parent
        self.ventana = None
        self.text_log = None
        self.abierta = False
        self.crear_ventana()
    
    def crear_ventana(self):
        if self.ventana is not None and self.ventana.winfo_exists():
            self.ventana.lift()
            return
        
        self.ventana = tk.Toplevel(self.parent)
        self.ventana.title("Log del Programa")
        self.ventana.geometry("600x400")
        self.ventana.configure(bg=COLORES['bg'])
        self.abierta = True
        
        self.ventana.protocol("WM_DELETE_WINDOW", self.cerrar)
        
        self.text_log = scrolledtext.ScrolledText(self.ventana, font=('Consolas', 9),
                                                    bg=COLORES['log_bg'], fg=COLORES['fg'])
        self.text_log.pack(fill='both', expand=True, padx=5, pady=5)
        
        btn_frame = tk.Frame(self.ventana, bg=COLORES['bg'])
        btn_frame.pack(fill='x', pady=5)
        
        tk.Button(btn_frame, text="Limpiar", command=self.limpiar,
                 bg=COLORES['accent'], fg='white', padx=10).pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="Cerrar", command=self.cerrar,
                 bg=COLORES['border'], fg=COLORES['fg'], padx=10).pack(side='right', padx=5)
    
    def agregar(self, mensaje):
        if not self.abierta:
            return
        try:
            if self.ventana is None or not self.ventana.winfo_exists():
                self.abierta = False
                return
            if self.text_log is None:
                return
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.text_log.insert(tk.END, f"[{timestamp}] {mensaje}\n")
            self.text_log.see(tk.END)
            self.ventana.update()
        except (tk.TclError, RuntimeError, AttributeError):
            self.abierta = False
    
    def limpiar(self):
        if not self.abierta:
            return
        try:
            if self.text_log and self.ventana and self.ventana.winfo_exists():
                self.text_log.delete('1.0', tk.END)
        except (tk.TclError, RuntimeError):
            self.abierta = False
    
    def cerrar(self):
        self.abierta = False
        if self.ventana:
            try:
                self.ventana.destroy()
            except:
                pass
            self.ventana = None
            self.text_log = None


class VentanaProgreso:
    def __init__(self, parent, titulo, on_cancelar=None):
        self.parent = parent
        self.on_cancelar = on_cancelar
        self.cancelado = False
        
        self.ventana = tk.Toplevel(parent)
        self.ventana.title(titulo)
        self.ventana.geometry("550x400")
        self.ventana.configure(bg=COLORES['bg'])
        self.ventana.transient(parent)
        
        self.label_titulo = tk.Label(self.ventana, text=titulo, font=('Segoe UI', 12, 'bold'),
                                      bg=COLORES['bg'], fg=COLORES['accent'])
        self.label_titulo.pack(pady=10)
        
        self.progress = ttk.Progressbar(self.ventana, mode='determinate', length=450)
        self.progress.pack(pady=10)
        
        self.label_estado = tk.Label(self.ventana, text="Iniciando...", bg=COLORES['bg'], fg=COLORES['fg'])
        self.label_estado.pack(pady=5)
        
        self.label_tiempo = tk.Label(self.ventana, text="", bg=COLORES['bg'], fg=COLORES['help'], font=('Segoe UI', 9))
        self.label_tiempo.pack(pady=5)
        
        self.text_log = scrolledtext.ScrolledText(self.ventana, height=8, font=('Consolas', 8),
                                                    bg=COLORES['log_bg'], fg=COLORES['fg'])
        self.text_log.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.btn_cancelar = tk.Button(self.ventana, text="Cancelar", command=self._cancelar,
                                       bg=COLORES['negative'], fg='white', padx=15, pady=5)
        self.btn_cancelar.pack(pady=10)
    
    def _cancelar(self):
        self.cancelado = True
        self.btn_cancelar.config(state='disabled', text="Cancelando...")
        if self.on_cancelar:
            self.on_cancelar()
        self.agregar_log("Cancelación solicitada... Terminando lote actual")
    
    def actualizar_progreso(self, valor, texto=""):
        self.progress['value'] = valor
        if texto:
            self.label_estado.config(text=texto)
        self.ventana.update()
    
    def agregar_log(self, mensaje):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.text_log.insert(tk.END, f"[{timestamp}] {mensaje}\n")
        self.text_log.see(tk.END)
        self.ventana.update()
    
    def cerrar(self):
        """Cierra la ventana de forma segura (NO el programa principal)"""
        try:
            if hasattr(self, 'ventana') and self.ventana and self.ventana.winfo_exists():
                self.ventana.destroy()
        except:
            pass
        self.ventana = None  


class VentanaAyuda:
    def __init__(self, parent):
        self.ventana = tk.Toplevel(parent)
        self.ventana.title("Ayuda")
        self.ventana.geometry("700x550")
        self.ventana.configure(bg=COLORES['bg'])
        
        texto = scrolledtext.ScrolledText(self.ventana, font=('Segoe UI', 10), wrap=tk.WORD,
                                           bg=COLORES['log_bg'], fg=COLORES['fg'])
        texto.pack(fill='both', expand=True, padx=10, pady=10)
        texto.insert('1.0', TEXTO_AYUDA)
        texto.config(state='disabled')
        
        tk.Button(self.ventana, text="Cerrar", command=self.ventana.destroy,
                 bg=COLORES['accent'], fg='white', padx=20, pady=5).pack(pady=10)


class VentanaSelectorModelo:
    def __init__(self, parent, modelos_info, callback):
        self.callback = callback
        self.ventana = tk.Toplevel(parent)
        self.ventana.title("Seleccionar Modelo")
        self.ventana.geometry("500x400")
        self.ventana.configure(bg=COLORES['bg'])
        self.ventana.transient(parent)
        self.ventana.grab_set()
        
        tk.Label(self.ventana, text="Seleccionar Modelo", font=('Segoe UI', 14, 'bold'),
                bg=COLORES['bg'], fg=COLORES['accent']).pack(pady=10)
        
        frame = tk.Frame(self.ventana, bg=COLORES['bg'])
        frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.var = tk.StringVar(value="Ensemble")
        
        orden = ['Ensemble', 'Máximo', 'Preciso', 'Rápido']
        for nivel in orden:
            info = modelos_info.get(nivel, {})
            if info.get('existe', False):
                acc = info.get('accuracy', 0)
                desc = info.get('descripcion', '')
                texto = f"{nivel} - {desc}"
                if acc > 0:
                    texto = f"{nivel} - {desc} (Precisión: {acc:.1%})"
                rb = tk.Radiobutton(frame, text=texto, variable=self.var, value=nivel,
                                    bg=COLORES['bg'], anchor='w', justify='left')
                rb.pack(fill='x', pady=3)
        
        tk.Radiobutton(frame, text="Entrenar modelo nuevo", variable=self.var, value="nuevo",
                      bg=COLORES['bg'], fg=COLORES['warning']).pack(fill='x', pady=10)
        
        btn_frame = tk.Frame(self.ventana, bg=COLORES['bg'])
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="Cargar", command=self.confirmar,
                 bg=COLORES['accent'], fg='white', padx=20, pady=5).pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="Cancelar", command=self.cancelar,
                 bg=COLORES['border'], fg=COLORES['fg'], padx=15, pady=5).pack(side='left', padx=5)
    
    def confirmar(self):
        seleccion = self.var.get()
        self.ventana.destroy()
        self.callback(seleccion)
    
    def cancelar(self):
        self.ventana.destroy()
        self.callback(None)


class VentanaOpciones:
    def __init__(self, parent, callback, tipo="entrenamiento"):
        self.callback = callback
        self.ventana = tk.Toplevel(parent)
        self.ventana.title("Configurar Operación")
        self.ventana.geometry("400x600")
        self.ventana.configure(bg=COLORES['bg'])
        self.ventana.transient(parent)
        self.ventana.grab_set()
        
        tk.Label(self.ventana, text="Configurar Operación", font=('Segoe UI', 12, 'bold'),
                bg=COLORES['bg'], fg=COLORES['accent']).pack(pady=10)
        
        frame = tk.Frame(self.ventana, bg=COLORES['bg'])
        frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        if tipo == "entrenamiento":
            tk.Label(frame, text="Modo de entrenamiento:", bg=COLORES['bg'], font=('Segoe UI', 10, 'bold')).pack(anchor='w')
            self.modo_var = tk.StringVar(value="mezclar")
            tk.Radiobutton(frame, text="Mezclar con datos antiguos (más preciso)", variable=self.modo_var, value="mezclar", bg=COLORES['bg']).pack(anchor='w')
            tk.Radiobutton(frame, text="Añadir sin reentrenar todo (más rápido)", variable=self.modo_var, value="anadir", bg=COLORES['bg']).pack(anchor='w')
            tk.Frame(frame, height=10, bg=COLORES['bg']).pack()
        
        tk.Label(frame, text="Límite de tiempo:", bg=COLORES['bg'], font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        self.timeout_var = tk.StringVar(value="Sin límite")
        for texto, _ in OPCIONES_TIMEOUT:
            tk.Radiobutton(frame, text=texto, variable=self.timeout_var, value=texto, bg=COLORES['bg']).pack(anchor='w')
        
        if tipo == "entrenamiento":
            tk.Frame(frame, height=10, bg=COLORES['bg']).pack()
            tk.Label(frame, text="Límite de reseñas a usar:", bg=COLORES['bg'], font=('Segoe UI', 10, 'bold')).pack(anchor='w')
            self.limite_var = tk.StringVar(value="1,000,000 reseñas")
            for texto, valor in OPCIONES_LIMITE_RESENAS:
                tk.Radiobutton(frame, text=texto, variable=self.limite_var, value=texto, bg=COLORES['bg']).pack(anchor='w')
        
        btn_frame = tk.Frame(self.ventana, bg=COLORES['bg'])
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="Iniciar", command=self.confirmar,
                 bg='#27ae60', fg='white', padx=20, pady=5).pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="Cancelar", command=self.cancelar,
                 bg=COLORES['border'], fg=COLORES['fg'], padx=15, pady=5).pack(side='left', padx=5)
    
    def confirmar(self):
        timeout_texto = self.timeout_var.get()
        segundos = None
        for texto, valor in OPCIONES_TIMEOUT:
            if texto == timeout_texto:
                segundos = valor
                break
        
        modo = self.modo_var.get() if hasattr(self, 'modo_var') else None
        limite_resenas = None
        if hasattr(self, 'limite_var'):
            for texto, valor in OPCIONES_LIMITE_RESENAS:
                if texto == self.limite_var.get():
                    limite_resenas = valor
                    break
        
        self.ventana.destroy()
        self.callback(segundos, modo, limite_resenas)
    
    def cancelar(self):
        self.ventana.destroy()
        self.callback(None, None, None)


class Aplicacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Amazon Review Analyzer")
        # Obtener el tamaño de la pantalla
        ancho_pantalla = self.root.winfo_screenwidth()
        alto_pantalla = self.root.winfo_screenheight()
        
        # Usar el 90% de la pantalla (márgenes del 5% en cada lado)
        ancho_ventana = int(ancho_pantalla * 0.9)
        alto_ventana = int(alto_pantalla * 0.9)
        
        # Centrar la ventana
        x = (ancho_pantalla - ancho_ventana) // 2
        y = (alto_pantalla - alto_ventana) // 2
        
        self.root.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")
        self.root.configure(bg=COLORES['bg'])
        
        self.entrenador = ModeloEntrenador(CARPETA_MODELOS)
        self.predictor = Predictor(CARPETA_MODELOS)
        self.ventana_log = None
        self.proceso_activo = False
        self.hilo_actual = None
        self.ventana_progreso = None
        self.proceso_entrenamiento = None
        self.queue = None
        self.dataset_manager = DatasetManager()
        self.archivos_analisis = []
        self.csv_var = tk.StringVar()
        self.ruta_var = tk.StringVar()
        self.resultados_actuales = None
        
        self.setup_ui()
        self.root.after(100, self.seleccionar_modelo_inicio)
    
    def agregar_log_si_ventana(self, mensaje):
        if self.ventana_log:
            self.ventana_log.agregar(mensaje)
    
    def setup_ui(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        menu_modelo = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Modelo", menu=menu_modelo)
        menu_modelo.add_command(label="Cambiar modelo", command=self.cambiar_modelo)
        menu_modelo.add_command(label="Exportar modelo", command=self.exportar_modelo)
        menu_modelo.add_separator()
        menu_modelo.add_command(label="Info del modelo", command=self.mostrar_info)
        
        menu_ver = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ver", menu=menu_ver)
        menu_ver.add_command(label="Abrir log", command=self.abrir_log)
        menu_ver.add_separator()
        menu_ver.add_command(label="🗑️ Gestionar caché", command=self.gestionar_cache)
        
        menu_ayuda = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ayuda", menu=menu_ayuda)
        menu_ayuda.add_command(label="Guía de uso", command=self.mostrar_ayuda)
        menu_ayuda.add_command(label="Acerca de", command=self.mostrar_acerca)
        
        main_frame = tk.Frame(self.root, bg=COLORES['bg'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        titulo = tk.Label(main_frame, text="Amazon Review Analyzer", 
                          font=('Segoe UI', 20, 'bold'), bg=COLORES['bg'], fg=COLORES['accent'])
        titulo.pack(pady=5)
        
        self.estado_label = tk.Label(main_frame, text="", bg=COLORES['bg'], fg=COLORES['secondary'])
        self.estado_label.pack(pady=5)
        self.actualizar_estado()
        
        # SECCIÓN 1: GESTIONAR DATASETS
        frame1 = tk.LabelFrame(main_frame, text="📁 1. GESTIONAR DATASETS", font=('Segoe UI', 11, 'bold'),
                                bg=COLORES['bg'], fg=COLORES['accent'], padx=10, pady=10)
        frame1.pack(fill='x', pady=5)
        
        btn_frame1 = tk.Frame(frame1, bg=COLORES['bg'])
        btn_frame1.pack(fill='x', pady=5)
        
        self.btn_añadir_dataset = tk.Button(btn_frame1, text="📂 AÑADIR ARCHIVOS", command=self.añadir_datasets,
                                             bg=COLORES['accent'], fg='white', padx=15, pady=5)
        self.btn_añadir_dataset.pack(side='left', padx=5)
        
        self.btn_limpiar_datasets = tk.Button(btn_frame1, text="🗑️ LIMPIAR LISTA", command=self.limpiar_datasets,
                                               bg=COLORES['negative'], fg='white', padx=15, pady=5)
        self.btn_limpiar_datasets.pack(side='left', padx=5)
        
        self.lista_datasets_frame = tk.Frame(frame1, bg=COLORES['bg'])
        self.lista_datasets_frame.pack(fill='both', expand=True, pady=5)
        
        self.lista_datasets = tk.Listbox(self.lista_datasets_frame, height=4, font=('Segoe UI', 9),
                                          bg='#fafafa', fg=COLORES['fg'], selectmode=tk.SINGLE)
        self.lista_datasets.pack(side='left', fill='both', expand=True)
        
        scroll_datasets = tk.Scrollbar(self.lista_datasets_frame, orient='vertical', command=self.lista_datasets.yview)
        scroll_datasets.pack(side='right', fill='y')
        self.lista_datasets.config(yscrollcommand=scroll_datasets.set)
        
        btn_eliminar_sel = tk.Button(frame1, text="Eliminar seleccionado", command=self.eliminar_dataset_seleccionado,
                                      bg=COLORES['border'], fg=COLORES['fg'], padx=10, pady=2)
        btn_eliminar_sel.pack(anchor='w', pady=2)
        
        self.resumen_datasets_label = tk.Label(frame1, text="⚪ No hay datasets cargados",
                                                bg=COLORES['bg'], fg=COLORES['secondary'], font=('Segoe UI', 9))
        self.resumen_datasets_label.pack(anchor='w', pady=5)
        
        self.tiempo_entrenamiento_label = tk.Label(frame1, text="", bg=COLORES['bg'], fg=COLORES['help'], font=('Segoe UI', 9))
        self.tiempo_entrenamiento_label.pack(anchor='w', pady=2)
        
        self.dataset_label = tk.Label(frame1, text="", bg=COLORES['bg'], fg=COLORES['warning'])
        self.dataset_label.pack(anchor='w', pady=5)
        
        # SECCIÓN 2: ENTRENAR MODELO
        frame2 = tk.LabelFrame(main_frame, text="🎯 2. ENTRENAR MODELO", font=('Segoe UI', 11, 'bold'),
                                bg=COLORES['bg'], fg=COLORES['accent'], padx=10, pady=10)
        frame2.pack(fill='x', pady=5)
        
        opciones_frame = tk.Frame(frame2, bg=COLORES['bg'])
        opciones_frame.pack()
        
        self.nivel_var = tk.StringVar(value="Preciso")
        for texto, valor in OPCIONES_PRECISION:
            color = {'Ensemble': '#9b59b6', 'Máximo': '#e67e22', 'Preciso': '#2ecc71', 'Rápido': '#3498db'}.get(valor, COLORES['accent'])
            rb = tk.Radiobutton(opciones_frame, text=texto, variable=self.nivel_var, value=valor,
                                bg=COLORES['bg'], fg=color, font=('Segoe UI', 9, 'bold'), selectcolor=COLORES['bg'])
            rb.pack(side='left', padx=10)
        
        self.btn_entrenar = tk.Button(frame2, text="🎯 ENTRENAR MODELO", command=self.iniciar_entrenamiento,
                                       bg='#27ae60', fg='white', font=('Segoe UI', 11, 'bold'), padx=20, pady=5)
        self.btn_entrenar.pack(pady=10)
        
        # SECCIÓN 3: ANALIZAR RESEÑAS
        frame3 = tk.LabelFrame(main_frame, text="📊 3. ANALIZAR RESEÑAS", font=('Segoe UI', 11, 'bold'),
                                bg=COLORES['bg'], fg=COLORES['accent'], padx=10, pady=10)
        frame3.pack(fill='both', expand=True, pady=5)
        
        notebook = ttk.Notebook(frame3)
        notebook.pack(fill='both', expand=True)
        
        # Pestaña texto
        tab_texto = tk.Frame(notebook, bg=COLORES['bg'])
        notebook.add(tab_texto, text="📝 Analizar Texto")
        
        texto_main_frame = tk.Frame(tab_texto, bg=COLORES['bg'])
        texto_main_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        tk.Label(texto_main_frame, text="✏️ Ingresa tus reseñas (una por línea):",
                font=('Segoe UI', 10, 'bold'), bg=COLORES['bg'], fg=COLORES['fg']).pack(anchor='w', pady=(0, 5))
        
        self.texto_resenas = scrolledtext.ScrolledText(texto_main_frame, height=12, font=('Segoe UI', 10),
                                                        bg='#fafafa', relief='solid', borderwidth=1)
        self.texto_resenas.pack(fill='both', expand=True, pady=5)
        
        btn_texto_frame = tk.Frame(texto_main_frame, bg=COLORES['bg'])
        btn_texto_frame.pack(pady=10)
        
        self.btn_analizar_texto = tk.Button(btn_texto_frame, text="🔍 ANALIZAR", command=self.analizar_texto,
                                             bg=COLORES['accent'], fg='white', font=('Segoe UI', 11, 'bold'),
                                             padx=30, pady=8, cursor='hand2')
        self.btn_analizar_texto.pack()
        
        # Pestaña archivos
        tab_archivos = tk.Frame(notebook, bg=COLORES['bg'])
        notebook.add(tab_archivos, text="📁 Analizar Archivos CSV")
        
        main_archivo_frame = tk.Frame(tab_archivos, bg=COLORES['bg'])
        main_archivo_frame.pack(fill='both', expand=True, padx=20, pady=15)
        
        tk.Label(main_archivo_frame, text="📂 Selecciona múltiples archivos CSV para analizar",
                font=('Segoe UI', 12, 'bold'), bg=COLORES['bg'], fg=COLORES['accent']).pack(anchor='w', pady=(0, 15))
        
        btn_archivo_frame = tk.Frame(main_archivo_frame, bg=COLORES['bg'])
        btn_archivo_frame.pack(fill='x', pady=5)
        
        self.btn_añadir_analisis = tk.Button(btn_archivo_frame, text="📂 AÑADIR ARCHIVOS", command=self.añadir_archivos_analisis,
                                              bg=COLORES['accent'], fg='white', padx=15, pady=5)
        self.btn_añadir_analisis.pack(side='left', padx=5)
        
        self.btn_limpiar_analisis = tk.Button(btn_archivo_frame, text="🗑️ LIMPIAR LISTA", command=self.limpiar_archivos_analisis,
                                               bg=COLORES['negative'], fg='white', padx=15, pady=5)
        self.btn_limpiar_analisis.pack(side='left', padx=5)
        
        self.lista_analisis_frame = tk.Frame(main_archivo_frame, bg=COLORES['bg'])
        self.lista_analisis_frame.pack(fill='both', expand=True, pady=5)
        
        self.lista_analisis = tk.Listbox(self.lista_analisis_frame, height=6, font=('Segoe UI', 9),
                                          bg='#fafafa', fg=COLORES['fg'], selectmode=tk.SINGLE)
        self.lista_analisis.pack(side='left', fill='both', expand=True)
        
        scroll_analisis = tk.Scrollbar(self.lista_analisis_frame, orient='vertical', command=self.lista_analisis.yview)
        scroll_analisis.pack(side='right', fill='y')
        self.lista_analisis.config(yscrollcommand=scroll_analisis.set)
        
        btn_eliminar_analisis = tk.Button(main_archivo_frame, text="Eliminar seleccionado", command=self.eliminar_archivo_analisis,
                                           bg=COLORES['border'], fg=COLORES['fg'], padx=10, pady=2)
        btn_eliminar_analisis.pack(anchor='w', pady=2)
        
        self.resumen_analisis_label = tk.Label(main_archivo_frame, text="⚪ No hay archivos para analizar",
                                                bg=COLORES['bg'], fg=COLORES['secondary'], font=('Segoe UI', 9))
        self.resumen_analisis_label.pack(anchor='w', pady=5)
        
        self.tiempo_analisis_label = tk.Label(main_archivo_frame, text="",
                                               bg=COLORES['bg'], fg=COLORES['help'], font=('Segoe UI', 9))
        self.tiempo_analisis_label.pack(anchor='w', pady=2)
        
        btn_analizar_todos = tk.Frame(main_archivo_frame, bg=COLORES['bg'])
        btn_analizar_todos.pack(pady=15)
        
        self.btn_analizar_todos = tk.Button(btn_analizar_todos, text="📊 ANALIZAR TODOS LOS ARCHIVOS", command=self.analizar_archivos_todos,
                                             bg='#27ae60', fg='white', font=('Segoe UI', 12, 'bold'),
                                             padx=30, pady=10, cursor='hand2')
        self.btn_analizar_todos.pack()
        
        # Pestaña resultados
        tab_res = tk.Frame(notebook, bg=COLORES['bg'])
        notebook.add(tab_res, text="📈 Resultados")
        
        self.resultados_text = scrolledtext.ScrolledText(tab_res, font=('Consolas', 10))
        self.resultados_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.resultados_text.tag_config('pos', foreground=COLORES['positive'])
        self.resultados_text.tag_config('neg', foreground=COLORES['negative'])
        
        self.btn_exportar = tk.Button(tab_res, text="💾 EXPORTAR RESULTADOS", command=self.exportar_resultados,
                                       bg='#27ae60', fg='white', padx=20, pady=5)
        self.btn_exportar.pack(pady=5)
    
    def actualizar_estado(self):
        if self.predictor.modelo_actual:
            self.estado_label.config(text=f"Modelo activo: {self.predictor.nivel_actual} | Precisión: {self.predictor.accuracy_actual:.1%}")
        else:
            self.estado_label.config(text="No hay modelo cargado. Entrena o selecciona uno.")

    def escribir_log(self, mensaje, es_error=False):
        """Escribe en el archivo de log"""
        try:
            import os
            from datetime import datetime
            carpeta_logs = os.path.join(CARPETA_ACTUAL, "logs")
            os.makedirs(carpeta_logs, exist_ok=True)
            archivo_log = os.path.join(carpeta_logs, f"log_{datetime.now().strftime('%Y%m%d')}.txt")
            
            with open(archivo_log, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                tipo = "ERROR" if es_error else "INFO"
                f.write(f"[{timestamp}] [{tipo}] {mensaje}\n")
        except:
            pass

    def exportar_resultados(self):
        """Exporta los resultados del análisis a CSV"""
        if not self.resultados_actuales:
            messagebox.showwarning("Atención", "No hay resultados para exportar")
            return
        
        archivo = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if archivo:
            df = pd.DataFrame(self.resultados_actuales['resultados'])
            
            # Asegurar que las columnas estén en orden
            columnas = ['texto_original', 'sentimiento', 'confianza', 'prob_positiva', 'prob_negativa', 'idioma']
            for col in columnas:
                if col not in df.columns:
                    df[col] = ''
            
            df.to_csv(archivo, index=False, encoding='utf-8-sig')
            messagebox.showinfo("Éxito", f"✅ Resultados exportados a:\n{archivo}")
            
            if self.ventana_log:
                self.ventana_log.agregar(f"Resultados exportados a {archivo}")
    
    def abrir_log(self):
        if self.ventana_log is None:
            self.ventana_log = VentanaLog(self.root)
        else:
            self.ventana_log.crear_ventana()
    
    def seleccionar_modelo_inicio(self):
        modelos = self.entrenador.listar_modelos()
        if any(m['existe'] for m in modelos.values()):
            VentanaSelectorModelo(self.root, modelos, self.cargar_modelo_seleccionado)
    
    def cargar_modelo_seleccionado(self, nivel):
        if nivel and nivel != "nuevo":
            self.predictor.cargar_modelo(nivel)
            self.actualizar_estado()
            if self.ventana_log:
                self.ventana_log.agregar(f"Modelo cargado: {nivel}")
    
    def cambiar_modelo(self):
        modelos = self.entrenador.listar_modelos()
        VentanaSelectorModelo(self.root, modelos, self.cargar_modelo_seleccionado)
    
    def exportar_modelo(self):
        if not self.predictor.modelo_actual:
            messagebox.showwarning("Atención", "No hay modelo activo")
            return
        
        nombre = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Modelo PKL", "*.pkl")])
        if nombre:
            datos = {
                'models': self.predictor.modelo_actual,
                'vectorizer': self.predictor.vectorizer_actual,
                'accuracy': self.predictor.accuracy_actual,
                'nivel': self.predictor.nivel_actual
            }
            joblib.dump(datos, nombre)
            messagebox.showinfo("Éxito", f"Modelo exportado a {nombre}")
    
    def mostrar_info(self):
        if not self.predictor.modelo_actual:
            messagebox.showinfo("Info", "No hay modelo cargado")
            return
        info = f"Nivel: {self.predictor.nivel_actual}\nPrecisión: {self.predictor.accuracy_actual:.1%}\nModelos: {', '.join(self.predictor.modelo_actual.keys())}"
        messagebox.showinfo("Información del Modelo", info)
    
    def mostrar_ayuda(self):
        VentanaAyuda(self.root)
    
    def mostrar_acerca(self):
        messagebox.showinfo("Acerca de", "Amazon Review Analyzer v6.0\n\nClasificador de reseñas con Ensemble Learning\n4 niveles de precisión\nCancelación en tiempo real\nMúltiples datasets\n\n© 2026")
    
    # ============================================
    # SECCIÓN 1: GESTIÓN DE DATASETS
    # ============================================
    
    def añadir_datasets(self):
        archivos = filedialog.askopenfilenames(
            title="Selecciona datasets",
            filetypes=[("CSV files", "*.csv"), ("Todos", "*.*")]
        )
        if not archivos:
            return
        
        for archivo in archivos:
            self.dataset_manager.añadir_dataset(archivo)
        
        self.actualizar_lista_datasets()
        self.guardar_dataset_combinado()
    
    def limpiar_datasets(self):
        if messagebox.askyesno("Confirmar", "¿Eliminar todos los datasets de la lista?"):
            self.dataset_manager.limpiar_todos()
            self.actualizar_lista_datasets()
            self.guardar_dataset_combinado()
    
    def eliminar_dataset_seleccionado(self):
        seleccion = self.lista_datasets.curselection()
        if seleccion:
            indice = seleccion[0]
            self.dataset_manager.eliminar_dataset(indice)
            self.actualizar_lista_datasets()
            self.guardar_dataset_combinado()
        else:
            messagebox.showwarning("Atención", "Selecciona un dataset para eliminar")
    
    def actualizar_lista_datasets(self):
        self.lista_datasets.delete(0, tk.END)
        
        for info in self.dataset_manager.info_datasets:
            texto = f"{info['nombre']} ({info['tamaño_mb']:.1f} MB, ~{info['filas_estimadas']:,} reseñas)"
            self.lista_datasets.insert(tk.END, texto)
        
        resumen = self.dataset_manager.obtener_resumen()
        self.resumen_datasets_label.config(text=resumen)
        
        nivel = self.nivel_var.get()
        tiempo_info = self.dataset_manager.calcular_tiempo_entrenamiento(nivel)
        
        if tiempo_info and tiempo_info['resenas'] > 0:
            tiempo_str = self.dataset_manager.formatear_tiempo(tiempo_info['segundos'])
            self.tiempo_entrenamiento_label.config(
                text=f"⏱️ Tiempo estimado de entrenamiento ({nivel}): {tiempo_str}",
                fg=COLORES['help']
            )
        else:
            self.tiempo_entrenamiento_label.config(text="")
    
    def guardar_dataset_combinado(self):
        if self.dataset_manager.datasets:
            def combinar():
                self.dataset_manager.combinar_datasets(self.agregar_log_si_ventana)
                self.dataset_manager.guardar_combinado(ARCHIVO_DATASET)
                self.root.after(0, self.actualizar_lista_datasets)
                self.root.after(0, lambda: self.dataset_label.config(
                    text=f"✅ Dataset combinado: {self.dataset_manager.total_filas:,} reseñas",
                    fg=COLORES['positive']
                ))
            threading.Thread(target=combinar).start()
        else:
            self.dataset_label.config(text="⚠️ No hay datasets cargados", fg=COLORES['warning'])
    
    # ============================================
    # SECCIÓN 2: ENTRENAMIENTO
    # ============================================
    
    def iniciar_entrenamiento(self):
        if self.proceso_activo:
            messagebox.showwarning("Atención", "Ya hay un proceso en curso")
            return
        if not os.path.exists(ARCHIVO_DATASET):
            messagebox.showwarning("Atención", "Primero carga un dataset")
            return
        
        nivel = self.nivel_var.get()
        
        def callback(timeout, modo, limite_resenas):
            if timeout is None and modo is None:
                return
            
            self.proceso_activo = True
            self.btn_entrenar.config(state='disabled')
            self.btn_analizar_texto.config(state='disabled')
            self.btn_analizar_todos.config(state='disabled')
            
            self.queue = Queue()
            
            archivo_nombre_path = os.path.join(CARPETA_DATASETS, "nombre_original.txt")
            if os.path.exists(archivo_nombre_path):
                with open(archivo_nombre_path, "r") as f:
                    nombre_real = f.read().strip()
            else:
                nombre_real = os.path.basename(ARCHIVO_DATASET)
            
            self.proceso_entrenamiento = Process(
                target=entrenar_proceso,
                args=(self.queue, CARPETA_MODELOS, nivel, ARCHIVO_DATASET, 
                      modo or "mezclar", limite_resenas, timeout // 60 if timeout else None,
                      nombre_real)
            )
            self.proceso_entrenamiento.start()
            
            self.ventana_progreso = VentanaProgreso(self.root, f"Entrenando {nivel}", self.cancelar_entrenamiento)
            self.ventana_progreso.agregar_log("🚀 Iniciando proceso de entrenamiento...")
            self.comprobar_progreso()
        
        VentanaOpciones(self.root, callback, "entrenamiento")
    
    def comprobar_progreso(self):
        if hasattr(self, 'queue') and self.queue is not None:
            try:
                while not self.queue.empty():
                    mensaje = self.queue.get_nowait()
                    tipo = mensaje[0]
                    
                    if tipo == 'progreso':
                        _, porcentaje, texto = mensaje
                        if self.ventana_progreso:
                            self.ventana_progreso.actualizar_progreso(porcentaje, texto)
                    elif tipo == 'log':
                        _, texto = mensaje
                        if self.ventana_progreso:
                            self.ventana_progreso.agregar_log(texto)
                    elif tipo == 'completado':
                        _, accuracy = mensaje
                        self.fin_entrenamiento(accuracy, self.nivel_var.get())
                        return
                    elif tipo == 'error':
                        _, error = mensaje
                        self.error_entrenamiento(error)
                        return
            except:
                pass
        
        if self.ventana_progreso is not None:
            try:
                if not self.ventana_progreso.ventana.winfo_exists():
                    self.limpiar_despues_cierre()
                    return
            except:
                self.limpiar_despues_cierre()
                return
        
        if hasattr(self, 'proceso_entrenamiento') and self.proceso_entrenamiento is not None:
            if self.proceso_entrenamiento.is_alive():
                self.root.after(500, self.comprobar_progreso)
            else:
                self.limpiar_despues_cierre()
    
    def limpiar_despues_cierre(self):
        if hasattr(self, 'proceso_entrenamiento') and self.proceso_entrenamiento is not None:
            if self.proceso_entrenamiento.is_alive():
                self.proceso_entrenamiento.terminate()
                self.proceso_entrenamiento.join(timeout=2)
            self.proceso_entrenamiento = None
        
        if hasattr(self, 'queue') and self.queue is not None:
            try:
                self.queue.close()
            except:
                pass
            self.queue = None
        
        self.proceso_activo = False
        self.btn_entrenar.config(state='normal')
        self.btn_analizar_texto.config(state='normal')
        self.btn_analizar_todos.config(state='normal')
        self.ventana_progreso = None
    
    def cancelar_entrenamiento(self):
        if hasattr(self, 'proceso_entrenamiento') and self.proceso_entrenamiento is not None:
            if self.proceso_entrenamiento.is_alive():
                self.proceso_entrenamiento.terminate()
                self.proceso_entrenamiento.join(timeout=2)
            self.proceso_entrenamiento = None
        
        if hasattr(self, 'queue') and self.queue is not None:
            try:
                self.queue.close()
            except:
                pass
            self.queue = None
        
        self.proceso_activo = False
        self.btn_entrenar.config(state='normal')
        self.btn_analizar_texto.config(state='normal')
        self.btn_analizar_todos.config(state='normal')
        
        if self.ventana_progreso:
            self.ventana_progreso.agregar_log("⏹️ Entrenamiento cancelado")
            self.ventana_progreso.cerrar()
        self.ventana_progreso = None
    
    def fin_entrenamiento(self, acc, nivel):
        if hasattr(self, 'proceso_entrenamiento') and self.proceso_entrenamiento is not None:
            if self.proceso_entrenamiento.is_alive():
                self.proceso_entrenamiento.terminate()
                self.proceso_entrenamiento.join(timeout=2)
            self.proceso_entrenamiento = None
        
        if hasattr(self, 'queue') and self.queue is not None:
            self.queue.close()
            self.queue = None
        
        self.proceso_activo = False
        self.btn_entrenar.config(state='normal')
        self.btn_analizar_texto.config(state='normal')
        self.btn_analizar_todos.config(state='normal')
        
        if self.ventana_progreso:
            if acc > 0:
                self.ventana_progreso.actualizar_progreso(100, f"Completado: {acc:.1%}")
                self.ventana_progreso.agregar_log(f"✅ Entrenamiento completado. Precisión: {acc:.1%}")
                self.predictor.cargar_modelo(nivel)
                self.actualizar_estado()
                if self.ventana_log:
                    self.ventana_log.agregar(f"Modelo {nivel} entrenado. Precisión: {acc:.1%}")
                messagebox.showinfo("Éxito", f"Modelo {nivel} entrenado.\nPrecisión: {acc:.1%}")
            else:
                self.ventana_progreso.agregar_log("Entrenamiento cancelado")
            self.ventana_progreso.ventana.after(2000, self.ventana_progreso.cerrar)
        
        self.ventana_progreso = None
    
    def error_entrenamiento(self, error):
        if hasattr(self, 'proceso_entrenamiento') and self.proceso_entrenamiento is not None:
            if self.proceso_entrenamiento.is_alive():
                self.proceso_entrenamiento.terminate()
                self.proceso_entrenamiento.join(timeout=2)
            self.proceso_entrenamiento = None
        
        if hasattr(self, 'queue') and self.queue is not None:
            self.queue.close()
            self.queue = None
        
        self.proceso_activo = False
        self.btn_entrenar.config(state='normal')
        self.btn_analizar_texto.config(state='normal')
        self.btn_analizar_todos.config(state='normal')
        
        if self.ventana_progreso:
            self.ventana_progreso.cerrar()
        
        messagebox.showerror("Error", f"Error en entrenamiento:\n{error}")
    
    # ============================================
    # SECCIÓN 3: ANÁLISIS
    # ============================================
    
    def analizar_texto(self):
        if self.proceso_activo:
            messagebox.showwarning("Atención", "Ya hay un proceso en curso")
            return
        if not self.predictor.modelo_actual:
            messagebox.showwarning("Atención", "Primero entrena o carga un modelo")
            return
        
        texto = self.texto_resenas.get('1.0', tk.END).strip()
        if not texto:
            messagebox.showwarning("Atención", "Ingresa al menos una reseña")
            return
        
        reseñas = [r.strip() for r in texto.split('\n') if r.strip()]
        self._iniciar_analisis(reseñas, "texto")
    
    def añadir_archivos_analisis(self):
        archivos = filedialog.askopenfilenames(
            title="Selecciona archivos CSV para analizar",
            filetypes=[("CSV files", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if not archivos:
            return
        
        for archivo in archivos:
            if archivo not in self.archivos_analisis:
                self.archivos_analisis.append(archivo)
        
        self.actualizar_lista_analisis()
    
    def limpiar_archivos_analisis(self):
        if messagebox.askyesno("Confirmar", "¿Eliminar todos los archivos de la lista?"):
            self.archivos_analisis = []
            self.actualizar_lista_analisis()
    
    def eliminar_archivo_analisis(self):
        seleccion = self.lista_analisis.curselection()
        if seleccion:
            indice = seleccion[0]
            self.archivos_analisis.pop(indice)
            self.actualizar_lista_analisis()
        else:
            messagebox.showwarning("Atención", "Selecciona un archivo para eliminar")
    
    def actualizar_lista_analisis(self):
        self.lista_analisis.delete(0, tk.END)
        
        total_filas = 0
        for archivo in self.archivos_analisis:
            nombre = os.path.basename(archivo)
            try:
                with open(archivo, 'r', encoding='utf-8') as f:
                    primeras_lineas = f.read(1024 * 1024)
                    estimado = primeras_lineas.count('\n')
                    total_filas += estimado * 10
                self.lista_analisis.insert(tk.END, f"{nombre} (~{estimado * 10:,} reseñas)")
            except:
                self.lista_analisis.insert(tk.END, f"{nombre} (error al estimar)")
        
        if self.archivos_analisis:
            self.resumen_analisis_label.config(
                text=f"📁 {len(self.archivos_analisis)} archivos, ~{total_filas:,} reseñas totales",
                fg=COLORES['positive']
            )
            tiempo_info = self.dataset_manager.calcular_tiempo_analisis(total_filas)
            if tiempo_info:
                tiempo_str = self.dataset_manager.formatear_tiempo(tiempo_info['segundos'])
                self.tiempo_analisis_label.config(
                    text=f"⏱️ Tiempo estimado de análisis: {tiempo_str}",
                    fg=COLORES['help']
                )
        else:
            self.resumen_analisis_label.config(text="⚪ No hay archivos para analizar", fg=COLORES['secondary'])
            self.tiempo_analisis_label.config(text="")
    
    def fin_analisis(self, resultados):
        """Finaliza el análisis - Solo cierra la ventana de progreso, NO el programa"""
        try:
            # Guardar resultados
            self.resultados_actuales = resultados
            
            # Mostrar resultados en la pestaña
            self.mostrar_resultados()
            
            # Cambiar a pestaña de resultados
            try:
                self.notebook.select(2)
            except:
                pass
            
            # Log
            if self.ventana_log:
                self.ventana_log.agregar(f"Análisis completado: {resultados['total']:,} reseñas")
            self.escribir_log(f"Análisis completado: {resultados['total']} reseñas")
            
        except Exception as e:
            self.escribir_log(f"Error en fin_analisis: {str(e)}", es_error=True)
        
        finally:
            # IMPORTANTE: Solo limpiar la ventana de progreso, NO cerrar el programa
            self.proceso_activo = False
            self.btn_entrenar.config(state='normal')
            self.btn_analizar_texto.config(state='normal')
            self.btn_analizar_todos.config(state='normal')
            
            # Cerrar ventana de progreso de forma segura (después de 2 segundos)
            if self.ventana_progreso:
                try:
                    # Mostrar mensaje final en la ventana de progreso antes de cerrar
                    self.ventana_progreso.actualizar_progreso(100, "¡Análisis completado!")
                    self.ventana_progreso.agregar_log(f"\n✅ Resumen final:")
                    self.ventana_progreso.agregar_log(f"   Total: {resultados['total']} reseñas")
                    self.ventana_progreso.agregar_log(f"   Positivas: {resultados['positivas']} ({resultados['porcentaje_positivas']:.1f}%)")
                    
                    if 'archivos' in resultados:
                        self.ventana_progreso.agregar_log(f"\n📁 Por archivo:")
                        for arch in resultados['archivos']:
                            total_a = arch.get('total', 0)
                            pos_a = arch.get('positivas', 0)
                            self.ventana_progreso.agregar_log(f"   {arch.get('archivo', 'desconocido')}: {total_a} reseñas, {pos_a} positivas ({pos_a/total_a*100:.1f}%)")
                    
                    # Programar cierre de la ventana de progreso (NO del programa)
                    self.root.after(2000, self.cerrar_ventana_progreso_seguro)
                except Exception as e:
                    self.escribir_log(f"Error al mostrar resumen final: {e}", es_error=True)
                    self.cerrar_ventana_progreso_seguro()
    
    def cerrar_ventana_progreso_seguro(self):
        """Cierra SOLO la ventana de progreso, NO el programa principal"""
        try:
            if self.ventana_progreso:
                self.ventana_progreso.cerrar()
                self.ventana_progreso = None
        except Exception as e:
            self.escribir_log(f"Error al cerrar ventana de progreso: {e}", es_error=True)
            self.ventana_progreso = None
    
    def analizar_archivos_todos(self):
        if self.proceso_activo:
            messagebox.showwarning("Atención", "Ya hay un proceso en curso")
            return
        if not self.predictor.modelo_actual:
            messagebox.showwarning("Atención", "Primero entrena o carga un modelo")
            return
        if not self.archivos_analisis:
            messagebox.showwarning("Atención", "No hay archivos para analizar")
            return
        
        self.proceso_activo = True
        self.btn_entrenar.config(state='disabled')
        self.btn_analizar_texto.config(state='disabled')
        self.btn_analizar_todos.config(state='disabled')
        
        self.ventana_progreso = VentanaProgreso(self.root, "Analizando múltiples archivos", self.cancelar_analisis)
        self.ventana_progreso.agregar_log(f"📊 Analizando {len(self.archivos_analisis)} archivos...")
        
        # Mostrar qué archivos se van a analizar
        for archivo in self.archivos_analisis:
            self.ventana_progreso.agregar_log(f"   📄 {os.path.basename(archivo)}")
        
        def ejecutar():
            import time
            inicio_total = time.time()
            todos_resultados = []
            total_archivos = len(self.archivos_analisis)
            
            for i, archivo in enumerate(self.archivos_analisis):
                if self.predictor.cancelar:
                    self.ventana_progreso.agregar_log("⏹️ Análisis cancelado por el usuario")
                    break
                
                nombre = os.path.basename(archivo)
                self.ventana_progreso.agregar_log(f"\n{'='*50}")
                self.ventana_progreso.agregar_log(f"📄 Procesando archivo {i+1}/{total_archivos}: {nombre}")
                self.ventana_progreso.agregar_log(f"{'='*50}")
                
                try:
                    # Cargar archivo
                    self.ventana_progreso.agregar_log(f"   📖 Leyendo archivo...")
                    df = pd.read_csv(archivo)
                    self.ventana_progreso.agregar_log(f"   ✅ {len(df):,} filas cargadas")
                    
                    # Detectar columna de texto
                    columna = None
                    for c in df.columns:
                        if any(p in c.lower() for p in ['review', 'text', 'resena', 'comment']):
                            columna = c
                            break
                    if columna is None:
                        columna = df.columns[0]
                        self.ventana_progreso.agregar_log(f"   ⚠️ Usando columna: {columna}")
                    else:
                        self.ventana_progreso.agregar_log(f"   📝 Columna de texto: {columna}")
                    
                    textos = df[columna].dropna().tolist()
                    total_textos = len(textos)
                    self.ventana_progreso.agregar_log(f"   📝 {total_textos:,} reseñas válidas")
                    
                    if total_textos == 0:
                        self.ventana_progreso.agregar_log(f"   ⚠️ No hay reseñas válidas en este archivo")
                        continue
                    
                    inicio_archivo = time.time()
                    
                    # Analizar reseñas una por una
                    resultados_archivo = []
                    positivas_archivo = 0
                    negativas_archivo = 0
                    
                    for j, texto in enumerate(textos):
                        if self.predictor.cancelar:
                            break
                        
                        # Actualizar progreso cada 100 reseñas
                        if (j + 1) % 100 == 0 or j == total_textos - 1:
                            porcentaje_global = (i + (j + 1) / total_textos) / total_archivos * 100
                            transcurrido = time.time() - inicio_total
                            
                            # Calcular tiempo restante
                            if j > 0:
                                tiempo_por_resena = transcurrido / (j + 1)
                                restante = tiempo_por_resena * (total_textos - j - 1)
                            else:
                                restante = 0
                            
                            self.ventana_progreso.actualizar_progreso(
                                porcentaje_global,
                                f"Archivo {i+1}/{total_archivos}: {nombre}\nAnalizando {j+1}/{total_textos}"
                            )
                            
                            try:
                                self.ventana_progreso.label_tiempo.config(
                                    text=f"⏱️ Tiempo transcurrido: {self._formatear_tiempo(transcurrido)}\n"
                                         f"⏰ Tiempo restante: {self._formatear_tiempo(restante)}"
                                )
                            except:
                                pass
                            
                            self.ventana_progreso.ventana.update()
                        
                        # Predecir reseña
                        try:
                            prediccion = self.predictor.predecir(texto)
                            if prediccion:
                                prediccion['texto_original'] = texto
                                resultados_archivo.append(prediccion)
                                if prediccion['sentimiento'] == 'Positiva':
                                    positivas_archivo += 1
                                else:
                                    negativas_archivo += 1
                        except Exception as e:
                            self.ventana_progreso.agregar_log(f"   ⚠️ Error en reseña {j+1}: {str(e)[:50]}")
                    
                    tiempo_archivo = time.time() - inicio_archivo
                    
                    self.ventana_progreso.agregar_log(f"\n   📊 Resumen del archivo:")
                    self.ventana_progreso.agregar_log(f"      ✅ Total: {len(resultados_archivo)} reseñas")
                    self.ventana_progreso.agregar_log(f"      👍 Positivas: {positivas_archivo}")
                    self.ventana_progreso.agregar_log(f"      👎 Negativas: {negativas_archivo}")
                    self.ventana_progreso.agregar_log(f"      ⏱️ Tiempo: {self._formatear_tiempo(tiempo_archivo)}")
                    
                    todos_resultados.append({
                        'archivo': nombre,
                        'total': len(resultados_archivo),
                        'positivas': positivas_archivo,
                        'negativas': negativas_archivo,
                        'tiempo': tiempo_archivo,
                        'resultados': resultados_archivo
                    })
                    
                except Exception as e:
                    self.ventana_progreso.agregar_log(f"   ❌ Error en archivo {nombre}: {str(e)}")
                    import traceback
                    self.ventana_progreso.agregar_log(f"   {traceback.format_exc()[:200]}")
            
            # Combinar resultados
            self.ventana_progreso.agregar_log(f"\n{'='*50}")
            self.ventana_progreso.agregar_log("📊 COMBINANDO RESULTADOS")
            self.ventana_progreso.agregar_log(f"{'='*50}")
            
            if todos_resultados:
                total_resenas = sum(r['total'] for r in todos_resultados)
                total_positivas = sum(r['positivas'] for r in todos_resultados)
                
                if total_resenas > 0:
                    confianza_promedio = sum(
                        sum(rr['confianza'] for rr in r['resultados']) / len(r['resultados']) if r['resultados'] else 0
                        for r in todos_resultados
                    ) / len(todos_resultados)
                else:
                    confianza_promedio = 0
                
                resultados_combinados = {
                    'total': total_resenas,
                    'positivas': total_positivas,
                    'negativas': total_resenas - total_positivas,
                    'porcentaje_positivas': total_positivas / total_resenas * 100 if total_resenas > 0 else 0,
                    'confianza_promedio': confianza_promedio,
                    'resultados': [r for res in todos_resultados for r in res['resultados']],
                    'archivos': [{'archivo': r['archivo'], 'total': r['total'], 'positivas': r['positivas'], 'tiempo': r['tiempo']} for r in todos_resultados]
                }
                
                self.ventana_progreso.agregar_log(f"✅ Total general: {total_resenas:,} reseñas")
                self.ventana_progreso.agregar_log(f"✅ Positivas: {total_positivas} ({resultados_combinados['porcentaje_positivas']:.1f}%)")
            else:
                resultados_combinados = {
                    'total': 0,
                    'positivas': 0,
                    'negativas': 0,
                    'porcentaje_positivas': 0,
                    'confianza_promedio': 0,
                    'resultados': [],
                    'archivos': []
                }
                self.ventana_progreso.agregar_log("⚠️ No se obtuvieron resultados")
            
            self.root.after(0, lambda: self.fin_analisis(resultados_combinados))
        
        threading.Thread(target=ejecutar).start()
    
    def cancelar_analisis(self):
        self.predictor.cancelar_analisis()
    
    
    def error_analisis(self, error):
        """Muestra el error y NO cierra el programa"""
        self.escribir_log(f"Error en análisis: {error}", es_error=True)
        
        self.proceso_activo = False
        self.btn_entrenar.config(state='normal')
        self.btn_analizar_texto.config(state='normal')
        self.btn_analizar_todos.config(state='normal')
        
        # Cerrar ventana de progreso si existe
        if self.ventana_progreso:
            self.ventana_progreso.cerrar()
            self.ventana_progreso = None
        
        # Mostrar error en un messagebox (no cierra el programa)
        messagebox.showerror("Error en análisis", f"Se produjo un error:\n\n{error}\n\nRevisa el archivo de log para más detalles.")
    
    def _formatear_tiempo(self, segundos):
        try:
            segundos = float(segundos)
            if segundos < 60:
                return f"{segundos:.0f} segundos"
            elif segundos < 3600:
                return f"{segundos/60:.1f} minutos"
            else:
                return f"{segundos/3600:.1f} horas"
        except:
            return "0 segundos"
    
    def mostrar_resultados(self, s=None):
        if s is None:
            s = self.resultados_actuales
        if not s:
            return
        
        self.resultados_text.delete('1.0', tk.END)
        self.resultados_text.insert(tk.END, "="*60 + "\n")
        self.resultados_text.insert(tk.END, "RESULTADOS DEL ANALISIS\n")
        self.resultados_text.insert(tk.END, "="*60 + "\n\n")
        self.resultados_text.insert(tk.END, f"Total de reseñas: {s['total']:,}\n")
        self.resultados_text.insert(tk.END, f"Positivas: {s['positivas']:,} ({s['porcentaje_positivas']:.1f}%)\n", 'pos')
        self.resultados_text.insert(tk.END, f"Negativas: {s['negativas']:,} ({100 - s['porcentaje_positivas']:.1f}%)\n", 'neg')
        
        # Gráfico de barras
        barra = 30
        pos = int(s['porcentaje_positivas'] / 100 * barra)
        neg = barra - pos
        self.resultados_text.insert(tk.END, f"\nVISUALIZACION:\n")
        self.resultados_text.insert(tk.END, f"   Positivas: ", 'pos')
        self.resultados_text.insert(tk.END, f"{'█' * pos} {s['porcentaje_positivas']:.0f}%\n")
        self.resultados_text.insert(tk.END, f"   Negativas: ", 'neg')
        self.resultados_text.insert(tk.END, f"{'█' * neg} {100 - s['porcentaje_positivas']:.0f}%\n\n")
        
        # Resumen por archivo (con todo: nombre, total, positivas, negativas, tiempo)
        if 'archivos' in s:
            self.resultados_text.insert(tk.END, "RESUMEN POR ARCHIVO:\n")
            for arch in s['archivos']:
                total_a = arch.get('total', 0)
                pos_a = arch.get('positivas', 0)
                neg_a = arch.get('negativas', total_a - pos_a)
                tiempo_a = arch.get('tiempo', 0)
                
                self.resultados_text.insert(tk.END, f"\n   Archivo: {arch.get('archivo', 'desconocido')}\n")
                self.resultados_text.insert(tk.END, f"      Total: {total_a} reseñas\n")
                self.resultados_text.insert(tk.END, f"      Positivas: {pos_a} ({pos_a/total_a*100:.1f}%)\n" if total_a > 0 else "      Positivas: 0\n", 'pos')
                self.resultados_text.insert(tk.END, f"      Negativas: {neg_a} ({neg_a/total_a*100:.1f}%)\n" if total_a > 0 else "      Negativas: 0\n", 'neg')
                self.resultados_text.insert(tk.END, f"      Tiempo: {self._formatear_tiempo(tiempo_a)}\n")
            self.resultados_text.insert(tk.END, "\n")
        
        # Primeras reseñas
        self.resultados_text.insert(tk.END, "PRIMERAS RESENAS ANALIZADAS:\n")
        for i, r in enumerate(s['resultados'][:5]):
            texto_corto = r['texto_original'][:100]
            self.resultados_text.insert(tk.END, f"\n{i+1}. {texto_corto}...\n")
            color = 'pos' if r['sentimiento'] == 'Positiva' else 'neg'
            self.resultados_text.insert(tk.END, f"   -> {r['sentimiento']} ", color)
            self.resultados_text.insert(tk.END, f"(confianza: {r['confianza']:.1f}%, idioma: {r['idioma']})\n")
        
        self.resultados_text.insert(tk.END, "\n" + "="*60 + "\n")
        self.resultados_text.insert(tk.END, "Analisis completado\n", 'pos')
    
    def gestionar_cache(self):
        """Abre la ventana de gestión de caché"""
        from tkinter import ttk
        import shutil
        
        carpeta_cache = os.path.join(CARPETA_MODELOS, "cache")
        ventana = tk.Toplevel(self.root)
        ventana.title("Gestión de Caché")
        ventana.geometry("700x500")
        ventana.configure(bg=COLORES['bg'])
        ventana.transient(self.root)
        ventana.grab_set()
        
        tk.Label(ventana, text="🗑️ Gestión de Caché", font=('Segoe UI', 14, 'bold'),
                bg=COLORES['bg'], fg=COLORES['accent']).pack(pady=10)
        
        # Frame para información
        info_frame = tk.Frame(ventana, bg=COLORES['bg'])
        info_frame.pack(fill='x', padx=20, pady=5)
        
        # Calcular tamaño total y listar archivos
        tamaño_total = 0
        archivos_cache = []
        
        if os.path.exists(carpeta_cache):
            for archivo in os.listdir(carpeta_cache):
                ruta = os.path.join(carpeta_cache, archivo)
                if os.path.isfile(ruta):
                    tamaño = os.path.getsize(ruta)
                    tamaño_total += tamaño
                    archivos_cache.append((archivo, tamaño, ruta))
        
        tamaño_mb = tamaño_total / (1024 * 1024)
        if tamaño_mb < 1024:
            texto_tamaño = f"{tamaño_mb:.2f} MB"
        else:
            texto_tamaño = f"{tamaño_mb/1024:.2f} GB"
        
        tk.Label(info_frame, text=f"📊 Espacio ocupado por caché: {texto_tamaño}",
                bg=COLORES['bg'], fg=COLORES['fg'], font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        
        tk.Label(info_frame, text=f"📄 Número de archivos: {len(archivos_cache)}",
                bg=COLORES['bg'], fg=COLORES['secondary']).pack(anchor='w')
        
        # Separador
        ttk.Separator(ventana, orient='horizontal').pack(fill='x', padx=20, pady=10)
        
        # Frame para la lista de archivos
        list_frame = tk.Frame(ventana, bg=COLORES['bg'])
        list_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        tk.Label(list_frame, text="📁 Archivos de caché:", bg=COLORES['bg'], fg=COLORES['fg'],
                font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        
        # Treeview para mostrar archivos
        tree_frame = tk.Frame(list_frame, bg=COLORES['bg'])
        tree_frame.pack(fill='both', expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(tree_frame)
        scrollbar.pack(side='right', fill='y')
        
        tree = ttk.Treeview(tree_frame, columns=('tamaño', 'fecha'), show='tree', yscrollcommand=scrollbar.set)
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=tree.yview)
        
        tree.heading('#0', text='Archivo')
        tree.column('#0', width=350)
        tree.heading('tamaño', text='Tamaño')
        tree.column('tamaño', width=100, anchor='center')
        tree.heading('fecha', text='Fecha modificación')
        tree.column('fecha', width=150, anchor='center')
        
        # Añadir archivos a la lista
        from datetime import datetime
        for archivo, tamaño, ruta in sorted(archivos_cache):
            if tamaño < 1024 * 1024:
                tamaño_str = f"{tamaño / 1024:.1f} KB"
            else:
                tamaño_str = f"{tamaño / (1024 * 1024):.1f} MB"
            
            # Obtener fecha de modificación
            fecha_mod = os.path.getmtime(ruta)
            fecha_str = datetime.fromtimestamp(fecha_mod).strftime("%Y-%m-%d %H:%M:%S")
            
            tree.insert('', 'end', text=archivo, values=(tamaño_str, fecha_str), tags=(ruta,))
        
        # Frame para botones
        btn_frame = tk.Frame(ventana, bg=COLORES['bg'])
        btn_frame.pack(fill='x', padx=20, pady=10)
        
        def eliminar_toda_cache():
            if messagebox.askyesno("Confirmar", f"¿Eliminar TODA la caché ({texto_tamaño})?\n\nEsta acción no se puede deshacer."):
                if os.path.exists(carpeta_cache):
                    shutil.rmtree(carpeta_cache)
                    os.makedirs(carpeta_cache, exist_ok=True)
                messagebox.showinfo("Completado", "Caché eliminada correctamente")
                ventana.destroy()
                self.gestionar_cache()  # Recargar ventana
        
        def eliminar_seleccionado():
            seleccion = tree.selection()
            if not seleccion:
                messagebox.showwarning("Atención", "Selecciona un archivo")
                return
            
            item = seleccion[0]
            ruta = tree.item(item, 'tags')[0]
            nombre = tree.item(item, 'text')
            
            if messagebox.askyesno("Confirmar", f"¿Eliminar {nombre}?"):
                try:
                    # Si es archivo .npz, eliminar también su .pkl asociado
                    if nombre.endswith('_X.npz'):
                        pkl_file = ruta.replace('_X.npz', '.pkl')
                        if os.path.exists(pkl_file):
                            os.remove(pkl_file)
                    os.remove(ruta)
                    messagebox.showinfo("Completado", "Archivo eliminado")
                    ventana.destroy()
                    self.gestionar_cache()  # Recargar
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo eliminar: {e}")
        
        tk.Button(btn_frame, text="🗑️ ELIMINAR SELECCIONADO", command=eliminar_seleccionado,
                 bg=COLORES['accent'], fg='white', padx=15, pady=5).pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="🗑️ ELIMINAR TODA LA CACHÉ", command=eliminar_toda_cache,
                 bg='#e74c3c', fg='white', padx=15, pady=5).pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="CERRAR", command=ventana.destroy,
                 bg=COLORES['border'], fg=COLORES['fg'], padx=15, pady=5).pack(side='right', padx=5)


if __name__ == "__main__":
    root = tk.Tk()
    app = Aplicacion(root)
    root.mainloop()
