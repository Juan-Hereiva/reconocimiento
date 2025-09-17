"""
Ventana de configuración avanzada para el sistema de reconocimiento facial
"""
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import json
from pathlib import Path
import sys

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config

class ConfigWindow:
    """Ventana de configuración avanzada"""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.window.title("Configuración Avanzada")
        self.window.geometry("600x700")
        self.window.configure(bg='#34495e')
        
        # Variables de configuración
        self.config_vars = {}
        self.init_config_vars()
        
        # Crear interfaz
        self.create_widgets()
        
        # Centrar ventana
        self.center_window()
    
    def init_config_vars(self):
        """Inicializar variables de configuración"""
        # Reconocimiento facial
        self.config_vars['face_threshold'] = tk.DoubleVar(value=Config.FACE_RECOGNITION_THRESHOLD)
        self.config_vars['face_crop_size'] = tk.IntVar(value=Config.FACE_CROP_SIZE)
        
        # Anti-spoofing
        self.config_vars['antispoof_threshold'] = tk.DoubleVar(value=Config.ANTI_SPOOF_THRESHOLD)
        self.config_vars['antispoof_input_size'] = tk.IntVar(value=Config.ANTI_SPOOF_INPUT_SIZE)
        
        # Detección de vida
        self.config_vars['ear_threshold'] = tk.DoubleVar(value=Config.EAR_THRESHOLD)
        self.config_vars['blink_frames'] = tk.IntVar(value=Config.BLINK_CONSECUTIVE_FRAMES)
        self.config_vars['liveness_window'] = tk.DoubleVar(value=Config.LIVENESS_TIME_WINDOW)
        
        # Cámara
        self.config_vars['camera_width'] = tk.IntVar(value=Config.CAMERA_WIDTH)
        self.config_vars['camera_height'] = tk.IntVar(value=Config.CAMERA_HEIGHT)
        self.config_vars['camera_fps'] = tk.IntVar(value=Config.CAMERA_FPS)
        
        # MediaPipe
        self.config_vars['mp_confidence'] = tk.DoubleVar(value=Config.MP_FACE_CONFIDENCE)
        self.config_vars['mp_max_faces'] = tk.IntVar(value=Config.MP_MAX_NUM_FACES)
        
        # Visualización
        self.config_vars['bbox_thickness'] = tk.IntVar(value=Config.BBOX_THICKNESS)
        self.config_vars['font_scale'] = tk.DoubleVar(value=Config.FONT_SCALE)
    
    def create_widgets(self):
        """Crear widgets de la interfaz"""
        # Frame principal con scroll
        main_canvas = tk.Canvas(self.window, bg='#34495e')
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Título
        title_label = tk.Label(scrollable_frame, text="Configuración Avanzada del Sistema", 
                              font=('Arial', 16, 'bold'), bg='#34495e', fg='white')
        title_label.pack(pady=10)
        
        # Crear secciones
        self.create_recognition_section(scrollable_frame)
        self.create_antispoof_section(scrollable_frame)
        self.create_liveness_section(scrollable_frame)
        self.create_camera_section(scrollable_frame)
        self.create_mediapipe_section(scrollable_frame)
        self.create_display_section(scrollable_frame)
        self.create_button_section(scrollable_frame)
        
        # Empaquetar canvas y scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Vincular scroll del mouse
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def create_recognition_section(self, parent):
        """Crear sección de reconocimiento facial"""
        frame = ttk.LabelFrame(parent, text="Reconocimiento Facial", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Umbral de reconocimiento
        self.create_scale_with_label(frame, "Umbral de Reconocimiento:", 
                                   self.config_vars['face_threshold'], 0.1, 1.0, 
                                   "Más bajo = menos estricto")
        
        # Tamaño de recorte de rostro
        self.create_spinbox_with_label(frame, "Tamaño de Rostro (píxeles):", 
                                     self.config_vars['face_crop_size'], 64, 512, 
                                     "Tamaño para embeddings FaceNet")
    
    def create_antispoof_section(self, parent):
        """Crear sección de anti-spoofing"""
        frame = ttk.LabelFrame(parent, text="Anti-Spoofing", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Umbral anti-spoofing
        self.create_scale_with_label(frame, "Umbral Anti-Spoofing:", 
                                   self.config_vars['antispoof_threshold'], 0.1, 1.0, 
                                   "Confianza mínima para rostro real")
        
        # Tamaño de entrada
        self.create_spinbox_with_label(frame, "Tamaño de Entrada (píxeles):", 
                                     self.config_vars['antispoof_input_size'], 64, 256, 
                                     "Tamaño para modelo anti-spoofing")
    
    def create_liveness_section(self, parent):
        """Crear sección de detección de vida"""
        frame = ttk.LabelFrame(parent, text="Detección de Vida", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Umbral EAR
        self.create_scale_with_label(frame, "Umbral EAR (Parpadeo):", 
                                   self.config_vars['ear_threshold'], 0.1, 0.5, 
                                   "Más bajo = más sensible al parpadeo")
        
        # Frames consecutivos
        self.create_spinbox_with_label(frame, "Frames Consecutivos:", 
                                     self.config_vars['blink_frames'], 1, 10, 
                                     "Frames para confirmar parpadeo")
        
        # Ventana de tiempo
        self.create_scale_with_label(frame, "Ventana de Tiempo (segundos):", 
                                   self.config_vars['liveness_window'], 1.0, 10.0, 
                                   "Tiempo para verificar vida")
    
    def create_camera_section(self, parent):
        """Crear sección de cámara"""
        frame = ttk.LabelFrame(parent, text="Configuración de Cámara", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Resolución
        res_frame = ttk.Frame(frame)
        res_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(res_frame, text="Resolución:").pack(side=tk.LEFT)
        
        ttk.Label(res_frame, text="Ancho:").pack(side=tk.LEFT, padx=(20, 5))
        width_spinbox = ttk.Spinbox(res_frame, from_=320, to=1920, width=8, 
                                   textvariable=self.config_vars['camera_width'])
        width_spinbox.pack(side=tk.LEFT)
        
        ttk.Label(res_frame, text="Alto:").pack(side=tk.LEFT, padx=(10, 5))
        height_spinbox = ttk.Spinbox(res_frame, from_=240, to=1080, width=8, 
                                    textvariable=self.config_vars['camera_height'])
        height_spinbox.pack(side=tk.LEFT)
        
        # FPS
        self.create_spinbox_with_label(frame, "FPS:", 
                                     self.config_vars['camera_fps'], 15, 60, 
                                     "Frames por segundo")
        
        # Presets de resolución
        preset_frame = ttk.Frame(frame)
        preset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(preset_frame, text="Presets:").pack(side=tk.LEFT)
        
        presets = [
            ("320x240", 320, 240),
            ("640x480", 640, 480),
            ("1280x720", 1280, 720),
            ("1920x1080", 1920, 1080)
        ]
        
        for name, w, h in presets:
            btn = ttk.Button(preset_frame, text=name, width=10,
                           command=lambda w=w, h=h: self.set_resolution(w, h))
            btn.pack(side=tk.LEFT, padx=2)
    
    def create_mediapipe_section(self, parent):
        """Crear sección de MediaPipe"""
        frame = ttk.LabelFrame(parent, text="MediaPipe", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Confianza de detección
        self.create_scale_with_label(frame, "Confianza de Detección:", 
                                   self.config_vars['mp_confidence'], 0.1, 1.0, 
                                   "Umbral para detectar rostros")
        
        # Número máximo de rostros
        self.create_spinbox_with_label(frame, "Máximo de Rostros:", 
                                     self.config_vars['mp_max_faces'], 1, 5, 
                                     "Rostros a detectar simultáneamente")
    
    def create_display_section(self, parent):
        """Crear sección de visualización"""
        frame = ttk.LabelFrame(parent, text="Visualización", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Grosor de bounding box
        self.create_spinbox_with_label(frame, "Grosor de Recuadro:", 
                                     self.config_vars['bbox_thickness'], 1, 5, 
                                     "Grosor de línea del recuadro")
        
        # Escala de fuente
        self.create_scale_with_label(frame, "Tamaño de Fuente:", 
                                   self.config_vars['font_scale'], 0.3, 2.0, 
                                   "Tamaño del texto")
        
        # Colores
        color_frame = ttk.Frame(frame)
        color_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(color_frame, text="Colores:").pack(side=tk.LEFT)
        
        self.color_allowed = Config.BBOX_COLOR_ALLOWED
        self.color_denied = Config.BBOX_COLOR_DENIED
        
        self.color_allowed_btn = tk.Button(color_frame, text="Permitido", 
                                          bg=self.rgb_to_hex(self.color_allowed),
                                          command=lambda: self.choose_color('allowed'))
        self.color_allowed_btn.pack(side=tk.LEFT, padx=(10, 5))
        
        self.color_denied_btn = tk.Button(color_frame, text="Denegado", 
                                         bg=self.rgb_to_hex(self.color_denied),
                                         command=lambda: self.choose_color('denied'))
        self.color_denied_btn.pack(side=tk.LEFT, padx=5)
    
    def create_button_section(self, parent):
        """Crear sección de botones"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=20)
        
        # Botones principales
        button_frame = ttk.Frame(frame)
        button_frame.pack()
        
        ttk.Button(button_frame, text="Aplicar", 
                  command=self.apply_config).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Guardar", 
                  command=self.save_config).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Cargar", 
                  command=self.load_config).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Restaurar", 
                  command=self.restore_defaults).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Cerrar", 
                  command=self.window.destroy).pack(side=tk.LEFT, padx=5)
    
    def create_scale_with_label(self, parent, label_text, variable, from_, to, description=""):
        """Crear escala con etiqueta y descripción"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        # Etiqueta principal
        label_frame = ttk.Frame(frame)
        label_frame.pack(fill=tk.X)
        
        ttk.Label(label_frame, text=label_text).pack(side=tk.LEFT)
        value_label = ttk.Label(label_frame, text=f"{variable.get():.2f}")
        value_label.pack(side=tk.RIGHT)
        
        # Escala
        scale = ttk.Scale(frame, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL)
        scale.pack(fill=tk.X, pady=(0, 2))
        
        # Actualizar etiqueta de valor
        def update_label(*args):
            value_label.config(text=f"{variable.get():.2f}")
        variable.trace('w', update_label)
        
        # Descripción
        if description:
            desc_label = ttk.Label(frame, text=description, font=('Arial', 8), foreground='gray')
            desc_label.pack(anchor=tk.W)
    
    def create_spinbox_with_label(self, parent, label_text, variable, from_, to, description=""):
        """Crear spinbox con etiqueta y descripción"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        # Etiqueta y spinbox
        label_frame = ttk.Frame(frame)
        label_frame.pack(fill=tk.X)
        
        ttk.Label(label_frame, text=label_text).pack(side=tk.LEFT)
        
        spinbox = ttk.Spinbox(label_frame, from_=from_, to=to, width=10, 
                             textvariable=variable)
        spinbox.pack(side=tk.RIGHT)
        
        # Descripción
        if description:
            desc_label = ttk.Label(frame, text=description, font=('Arial', 8), foreground='gray')
            desc_label.pack(anchor=tk.W)
    
    def set_resolution(self, width, height):
        """Establecer resolución preset"""
        self.config_vars['camera_width'].set(width)
        self.config_vars['camera_height'].set(height)
    
    def choose_color(self, color_type):
        """Elegir color"""
        if color_type == 'allowed':
            current_color = self.color_allowed
            button = self.color_allowed_btn
        else:
            current_color = self.color_denied
            button = self.color_denied_btn
        
        # Convertir BGR a RGB para colorchooser
        current_rgb = (current_color[2], current_color[1], current_color[0])
        
        color = colorchooser.askcolor(initialcolor=current_rgb, title="Elegir Color")
        
        if color[0]:  # Si se seleccionó un color
            rgb = tuple(int(c) for c in color[0])
            bgr = (rgb[2], rgb[1], rgb[0])  # Convertir a BGR para OpenCV
            
            if color_type == 'allowed':
                self.color_allowed = bgr
            else:
                self.color_denied = bgr
            
            button.config(bg=self.rgb_to_hex(rgb))
    
    def rgb_to_hex(self, color):
        """Convertir color BGR/RGB a hex"""
        if len(color) == 3:
            # Asumir RGB si viene de colorchooser, BGR si viene de config
            if isinstance(color[0], float):
                return f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"
            else:
                # BGR a RGB para display
                return f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
        return "#000000"
    
    def apply_config(self):
        """Aplicar configuración actual"""
        try:
            # Actualizar Config con valores actuales
            Config.FACE_RECOGNITION_THRESHOLD = self.config_vars['face_threshold'].get()
            Config.FACE_CROP_SIZE = self.config_vars['face_crop_size'].get()
            Config.ANTI_SPOOF_THRESHOLD = self.config_vars['antispoof_threshold'].get()
            Config.ANTI_SPOOF_INPUT_SIZE = self.config_vars['antispoof_input_size'].get()
            Config.EAR_THRESHOLD = self.config_vars['ear_threshold'].get()
            Config.BLINK_CONSECUTIVE_FRAMES = self.config_vars['blink_frames'].get()
            Config.LIVENESS_TIME_WINDOW = self.config_vars['liveness_window'].get()
            Config.CAMERA_WIDTH = self.config_vars['camera_width'].get()
            Config.CAMERA_HEIGHT = self.config_vars['camera_height'].get()
            Config.CAMERA_FPS = self.config_vars['camera_fps'].get()
            Config.MP_FACE_CONFIDENCE = self.config_vars['mp_confidence'].get()
            Config.MP_MAX_NUM_FACES = self.config_vars['mp_max_faces'].get()
            Config.BBOX_THICKNESS = self.config_vars['bbox_thickness'].get()
            Config.FONT_SCALE = self.config_vars['font_scale'].get()
            Config.BBOX_COLOR_ALLOWED = self.color_allowed
            Config.BBOX_COLOR_DENIED = self.color_denied
            
            messagebox.showinfo("Éxito", "Configuración aplicada correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error aplicando configuración: {e}")
    
    def save_config(self):
        """Guardar configuración a archivo"""
        try:
            config_data = {}
            
            # Recopilar todos los valores
            for key, var in self.config_vars.items():
                config_data[key] = var.get()
            
            # Agregar colores
            config_data['color_allowed'] = list(self.color_allowed)
            config_data['color_denied'] = list(self.color_denied)
            
            # Guardar a archivo
            config_file = Config.BASE_DIR / "user_config.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            messagebox.showinfo("Éxito", f"Configuración guardada en {config_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error guardando configuración: {e}")
    
    def load_config(self):
        """Cargar configuración desde archivo"""
        try:
            config_file = Config.BASE_DIR / "user_config.json"
            
            if not config_file.exists():
                messagebox.showwarning("Advertencia", "No se encontró archivo de configuración")
                return
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Aplicar valores cargados
            for key, value in config_data.items():
                if key in self.config_vars:
                    self.config_vars[key].set(value)
                elif key == 'color_allowed':
                    self.color_allowed = tuple(value)
                    self.color_allowed_btn.config(bg=self.rgb_to_hex(value))
                elif key == 'color_denied':
                    self.color_denied = tuple(value)
                    self.color_denied_btn.config(bg=self.rgb_to_hex(value))
            
            messagebox.showinfo("Éxito", "Configuración cargada correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando configuración: {e}")
    
    def restore_defaults(self):
        """Restaurar valores por defecto"""
        if messagebox.askyesno("Confirmar", "¿Restaurar configuración por defecto?"):
            # Reinicializar variables con valores por defecto
            self.init_config_vars()
            
            # Restaurar colores
            self.color_allowed = Config.BBOX_COLOR_ALLOWED
            self.color_denied = Config.BBOX_COLOR_DENIED
            self.color_allowed_btn.config(bg=self.rgb_to_hex(self.color_allowed))
            self.color_denied_btn.config(bg=self.rgb_to_hex(self.color_denied))
            
            messagebox.showinfo("Éxito", "Configuración restaurada a valores por defecto")
    
    def center_window(self):
        """Centrar ventana en pantalla"""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')


if __name__ == "__main__":
    # Prueba independiente
    config_window = ConfigWindow()
    config_window.window.mainloop()