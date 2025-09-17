"""
Interfaz gráfica principal para el sistema de reconocimiento facial
"""
# ===== IMPORTS =====
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import sys
import time
from pathlib import Path

# Agregar el directorio actual al path para imports
sys.path.append(str(Path(__file__).parent))

from config.config import Config
from models.face_detector import FaceDetector
from models.face_recognizer import FaceRecognizer
from models.anti_spoof import AntiSpoofDetector
from models.liveness_detector import LivenessDetector
from utils.camera_utils import CameraManager
from utils.database_manager import DatabaseManager

# ===== CLASE PRINCIPAL =====
class FaceRecognitionGUI:
    """Interfaz gráfica para el sistema de reconocimiento facial"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sistema de Reconocimiento Facial con Detección de Vida")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Variables de estado
        self.is_running = False
        self.current_frame = None
        self.processed_frame = None
        self.camera_thread = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Inicializar variables de interfaz PRIMERO
        self.status_text = tk.StringVar(value="Iniciando sistema...")
        self.person_count_text = tk.StringVar(value="Personas: 0")
        self.fps_text = tk.StringVar(value="FPS: 0.0")
        
        # Crear interfaz
        self.create_widgets()
        
        # Inicializar componentes del sistema DESPUÉS de crear la interfaz
        self.init_system_components()
        
        # Configurar eventos
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Cargar base de datos existente
        self.load_database()
    
    def init_system_components(self):
        """Inicializar componentes del sistema"""
        try:
            self.status_text.set("Creando directorios...")
            Config.create_directories()
            
            self.status_text.set("Inicializando detector facial...")
            self.face_detector = FaceDetector()
            
            self.status_text.set("Inicializando reconocedor facial...")
            self.face_recognizer = FaceRecognizer()
            
            self.status_text.set("Inicializando detector anti-spoofing...")
            self.anti_spoof_detector = AntiSpoofDetector(str(Config.ANTI_SPOOF_MODEL_PATH))
            
            self.status_text.set("Inicializando detector de vida...")
            self.liveness_detector = LivenessDetector()
            
            self.status_text.set("Inicializando gestor de cámara...")
            self.camera_manager = CameraManager()
            
            self.status_text.set("Inicializando gestor de base de datos...")
            self.database_manager = DatabaseManager(str(Config.FACE_DATABASE_PATH))
            
            self.status_text.set("Sistema inicializado correctamente")
            
        except Exception as e:
            error_msg = f"Error inicializando sistema: {e}"
            self.status_text.set(error_msg)
            print(f"Error detallado: {e}")
            messagebox.showerror("Error de Inicialización", 
                               f"{error_msg}\n\nVerifique que todas las dependencias estén instaladas.")
            
            # Inicializar con valores por defecto para evitar más errores
            try:
                if not hasattr(self, 'face_detector'):
                    self.face_detector = None
                if not hasattr(self, 'face_recognizer'):
                    self.face_recognizer = None
                if not hasattr(self, 'anti_spoof_detector'):
                    self.anti_spoof_detector = None
                if not hasattr(self, 'liveness_detector'):
                    self.liveness_detector = None
                if not hasattr(self, 'camera_manager'):
                    self.camera_manager = None
                if not hasattr(self, 'database_manager'):
                    self.database_manager = None
            except:
                pass
    
    def create_widgets(self):
        """Crear widgets de la interfaz"""
        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo - Video
        self.create_video_panel(main_frame)
        
        # Panel derecho - Controles
        self.create_control_panel(main_frame)
        
        # Panel inferior - Estado
        self.create_status_panel(main_frame)
    
    def create_video_panel(self, parent):
        """Crear panel de video"""
        video_frame = ttk.LabelFrame(parent, text="Video en Tiempo Real", padding=10)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Canvas para mostrar video
        self.video_canvas = tk.Canvas(video_frame, width=640, height=480, bg='black')
        self.video_canvas.pack()
        
        # Controles de video
        video_controls = ttk.Frame(video_frame)
        video_controls.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(video_controls, text="Iniciar Cámara", 
                                      command=self.toggle_camera)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.capture_button = ttk.Button(video_controls, text="Capturar", 
                                       command=self.capture_frame, state=tk.DISABLED)
        self.capture_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Información de video
        video_info = ttk.Frame(video_frame)
        video_info.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(video_info, textvariable=self.fps_text).pack(side=tk.LEFT)
        ttk.Label(video_info, textvariable=self.person_count_text).pack(side=tk.RIGHT)
    
    def create_control_panel(self, parent):
        """Crear panel de controles"""
        control_frame = ttk.LabelFrame(parent, text="Panel de Control", padding=10)
        control_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # Configurar grid
        parent.grid_columnconfigure(0, weight=2)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        
        # Sección de registro
        self.create_registration_section(control_frame)
        
        # Sección de base de datos
        self.create_database_section(control_frame)
        
        # Sección de configuración
        self.create_settings_section(control_frame)
        
        # Sección de pruebas
        self.create_testing_section(control_frame)
    
    def create_registration_section(self, parent):
        """Crear sección de registro"""
        reg_frame = ttk.LabelFrame(parent, text="Registrar Persona", padding=10)
        reg_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Nombre de persona
        ttk.Label(reg_frame, text="Nombre:").pack(anchor=tk.W)
        self.name_entry = ttk.Entry(reg_frame, width=30)
        self.name_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Botones de registro
        reg_buttons = ttk.Frame(reg_frame)
        reg_buttons.pack(fill=tk.X)
        
        ttk.Button(reg_buttons, text="Registro Interactivo", 
                  command=self.start_interactive_registration).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(reg_buttons, text="Desde Carpeta", 
                  command=self.register_from_folder).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(reg_buttons, text="Desde Imagen", 
                  command=self.register_from_image).pack(fill=tk.X)
    
    def create_database_section(self, parent):
        """Crear sección de base de datos"""
        db_frame = ttk.LabelFrame(parent, text="Base de Datos", padding=10)
        db_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Lista de personas
        list_frame = ttk.Frame(db_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(list_frame, text="Personas Registradas:").pack(anchor=tk.W)
        
        # Listbox con scrollbar
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        self.person_listbox = tk.Listbox(listbox_frame, height=6)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.person_listbox.yview)
        self.person_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.person_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Botones de base de datos
        db_buttons = ttk.Frame(db_frame)
        db_buttons.pack(fill=tk.X)
        
        ttk.Button(db_buttons, text="Actualizar Lista", 
                  command=self.refresh_person_list).pack(fill=tk.X, pady=(0, 2))
        
        ttk.Button(db_buttons, text="Eliminar Seleccionado", 
                  command=self.delete_selected_person).pack(fill=tk.X, pady=(0, 2))
        
        ttk.Button(db_buttons, text="Ver Estadísticas", 
                  command=self.show_database_stats).pack(fill=tk.X, pady=(0, 2))
        
        ttk.Button(db_buttons, text="Mejorar Registro", 
                  command=self.improve_registration).pack(fill=tk.X)
    
    def create_settings_section(self, parent):
        """Crear sección de configuración"""
        settings_frame = ttk.LabelFrame(parent, text="Configuración", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Umbrales
        ttk.Label(settings_frame, text="Umbral Reconocimiento:").pack(anchor=tk.W)
        self.recognition_threshold = tk.DoubleVar(value=0.35)  # Reducido para ser menos estricto
        threshold_scale = ttk.Scale(settings_frame, from_=0.2, to=0.8, 
                                   variable=self.recognition_threshold, 
                                   orient=tk.HORIZONTAL)
        threshold_scale.pack(fill=tk.X, pady=(0, 5))
        
        # Mostrar valor actual
        self.threshold_label = tk.Label(settings_frame, text=f"Valor actual: {self.recognition_threshold.get():.2f}", 
                                       bg='#2c3e50', fg='white')
        self.threshold_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Actualizar etiqueta cuando cambie el valor
        def update_threshold_label(*args):
            self.threshold_label.config(text=f"Valor actual: {self.recognition_threshold.get():.2f}")
        self.recognition_threshold.trace('w', update_threshold_label)
        
        # Umbral EAR para detección de vida
        ttk.Label(settings_frame, text="Umbral Parpadeo (EAR):").pack(anchor=tk.W)
        self.ear_threshold = tk.DoubleVar(value=Config.EAR_THRESHOLD)
        ear_scale = ttk.Scale(settings_frame, from_=0.15, to=0.35, 
                             variable=self.ear_threshold, orient=tk.HORIZONTAL)
        ear_scale.pack(fill=tk.X, pady=(0, 5))
        
        # Checkbox para anti-spoofing
        self.enable_antispoof = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Habilitar Anti-Spoofing", 
                       variable=self.enable_antispoof).pack(anchor=tk.W, pady=(0, 5))
        
        # Checkbox para detección de vida
        self.enable_liveness = tk.BooleanVar(value=False)  # Deshabilitado por defecto
        ttk.Checkbutton(settings_frame, text="Habilitar Detección de Vida", 
                       variable=self.enable_liveness).pack(anchor=tk.W, pady=(0, 5))
        
        # Modo simple sin parpadeo
        self.simple_liveness = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Modo Simple (solo movimiento)", 
                       variable=self.simple_liveness).pack(anchor=tk.W, pady=(0, 5))
        
        # Botón para reiniciar detector de vida
        ttk.Button(settings_frame, text="Reiniciar Detector Vida", 
                  command=self.reset_liveness).pack(fill=tk.X, pady=(0, 2))
        
        # Botón para modo debug
        ttk.Button(settings_frame, text="Mostrar Debug Ojos", 
                  command=self.toggle_eye_debug).pack(fill=tk.X, pady=(0, 2))
        
        # Botón para debug de reconocimiento
        ttk.Button(settings_frame, text="Mostrar Valores Confianza", 
                  command=self.toggle_confidence_debug).pack(fill=tk.X)
        
        # Variables para debug
        self.show_eye_debug = False
        self.show_confidence_debug = False
    
    def create_testing_section(self, parent):
        """Crear sección de pruebas"""
        test_frame = ttk.LabelFrame(parent, text="Pruebas del Sistema", padding=10)
        test_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Configuración avanzada
        ttk.Button(test_frame, text="Configuración Avanzada", 
                  command=self.open_config_window).pack(fill=tk.X, pady=(0, 2))
        
        # Estadísticas
        ttk.Button(test_frame, text="Estadísticas y Monitoreo", 
                  command=self.open_stats_window).pack(fill=tk.X, pady=(0, 2))
        
        ttk.Button(test_frame, text="Probar Cámara", 
                  command=self.test_camera).pack(fill=tk.X, pady=(0, 2))
        
        ttk.Button(test_frame, text="Probar Detección", 
                  command=self.test_detection).pack(fill=tk.X, pady=(0, 2))
        
        ttk.Button(test_frame, text="Probar Todo", 
                  command=self.test_all).pack(fill=tk.X)
    
    def create_status_panel(self, parent):
        """Crear panel de estado"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        # Barra de estado
        self.status_bar = ttk.Label(status_frame, textvariable=self.status_text, 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X)
    
    def toggle_camera(self):
        """Alternar estado de la cámara"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Iniciar cámara"""
        try:
            if not self.camera_manager:
                messagebox.showerror("Error", "El gestor de cámara no está inicializado")
                return
                
            if not self.camera_manager.open_camera():
                messagebox.showerror("Error", "No se pudo abrir la cámara")
                return
            
            self.is_running = True
            self.start_button.config(text="Detener Cámara")
            self.capture_button.config(state=tk.NORMAL)
            
            # Iniciar hilo de procesamiento
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            self.status_text.set("Cámara iniciada")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error iniciando cámara: {e}")
            self.status_text.set(f"Error: {e}")
    
    def stop_camera(self):
        """Detener cámara"""
        self.is_running = False
        self.camera_manager.close_camera()
        self.start_button.config(text="Iniciar Cámara")
        self.capture_button.config(state=tk.DISABLED)
        self.status_text.set("Cámara detenida")
    
    def camera_loop(self):
        """Loop principal de procesamiento de cámara"""
        while self.is_running:
            try:
                frame = self.camera_manager.read_frame()
                if frame is None:
                    continue
                
                # Procesar frame
                processed_frame = self.process_frame(frame)
                
                # Actualizar FPS
                self.update_fps()
                
                # Mostrar frame en la interfaz
                self.update_video_display(processed_frame)
                
                # Pequeña pausa para no saturar la CPU
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error en camera_loop: {e}")
                break
    
    def process_frame(self, frame):
        """Procesar frame con reconocimiento facial"""
        try:
            # Verificar que los componentes estén inicializados
            if not self.face_detector:
                cv2.putText(frame, "Detector facial no inicializado", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return frame
            
            # Detectar rostro
            bbox, landmarks = self.face_detector.detect_face_and_landmarks(frame)
            
            if bbox is None:
                # No hay rostro
                cv2.putText(frame, "No se detecta rostro", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return frame
            
            # Extraer rostro
            face_crop = self.face_detector.crop_face(frame, bbox, size=Config.FACE_CROP_SIZE)
            if face_crop is None:
                return frame
            
            # Reconocimiento facial
            person_name, confidence = None, 0.0
            if self.face_recognizer:
                person_name, confidence = self.face_recognizer.identify_person(
                    face_crop, threshold=self.recognition_threshold.get()
                )
                
                # Debug de confianza
                if self.show_confidence_debug:
                    # Mostrar valores de confianza para todas las personas
                    debug_y = 100
                    cv2.putText(frame, f"Umbral actual: {self.recognition_threshold.get():.2f}", 
                               (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    if person_name:
                        cv2.putText(frame, f"Reconocido: {person_name} ({confidence:.3f})", 
                                   (10, debug_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        print(f"✅ RECONOCIDO: {person_name} con confianza {confidence:.3f} (umbral: {self.recognition_threshold.get():.2f})")
                    else:
                        cv2.putText(frame, f"No reconocido (confianza máxima: {confidence:.3f})", 
                                   (10, debug_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        print(f"❌ NO RECONOCIDO: confianza máxima {confidence:.3f} < umbral {self.recognition_threshold.get():.2f}")
                    
                    # Mostrar todas las personas en la DB
                    if hasattr(self, 'face_recognizer') and self.face_recognizer.face_database:
                        cv2.putText(frame, f"Personas en DB: {len(self.face_recognizer.face_database)}", 
                                   (10, debug_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Anti-spoofing si está habilitado
            is_real = True
            spoof_confidence = 1.0
            if self.enable_antispoof.get() and self.anti_spoof_detector:
                anti_spoof_crop = self.face_detector.crop_face(
                    frame, bbox, size=Config.ANTI_SPOOF_INPUT_SIZE
                )
                is_real, spoof_confidence = self.anti_spoof_detector.detect_spoofing(anti_spoof_crop)
            
            # Detección de vida si está habilitada
            is_alive = True
            liveness_info = {}
            if self.enable_liveness.get() and self.liveness_detector:
                # Actualizar umbral si cambió
                if hasattr(self, 'ear_threshold'):
                    self.liveness_detector.ear_threshold = self.ear_threshold.get()
                
                if self.simple_liveness.get():
                    # Modo simple - solo detectar movimiento
                    movement = self.liveness_detector.detect_head_movement(bbox)
                    has_movement = movement > 20.0
                    is_alive = has_movement or self.liveness_detector.blink_counter > 0
                    
                    if self.show_eye_debug:
                        cv2.putText(frame, f"Movimiento: {movement:.1f}", 
                                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"Vivo (simple): {'SI' if is_alive else 'NO'}", 
                                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # Modo complejo con análisis de parpadeo
                    left_eye, right_eye = self.face_detector.get_eye_landmarks(
                        landmarks, frame.shape, Config.LEFT_EYE_INDICES, Config.RIGHT_EYE_INDICES
                    )
                    
                    # Debug: mostrar información de ojos
                    if self.show_eye_debug:
                        # Dibujar puntos de ojos solo si tenemos landmarks válidos
                        if left_eye and len(left_eye) >= 4:
                            for point in left_eye:
                                cv2.circle(frame, point, 2, (255, 255, 0), -1)
                        if right_eye and len(right_eye) >= 4:
                            for point in right_eye:
                                cv2.circle(frame, point, 2, (0, 255, 255), -1)
                        
                        # Calcular EAR para debug solo si hay puntos válidos
                        if left_eye and right_eye and len(left_eye) >= 4 and len(right_eye) >= 4:
                            left_ear = self.liveness_detector.simplified_ear(left_eye)
                            right_ear = self.liveness_detector.simplified_ear(right_eye)
                            avg_ear = (left_ear + right_ear) / 2 if (left_ear > 0 and right_ear > 0) else 0
                            
                            # Mostrar valores EAR
                            cv2.putText(frame, f"EAR: {avg_ear:.3f} (Umbral: {self.liveness_detector.ear_threshold:.3f})", 
                                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, f"Parpadeos: {self.liveness_detector.blink_counter}", 
                                       (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            print(f"DEBUG - EAR: {avg_ear:.3f}, Umbral: {self.liveness_detector.ear_threshold:.3f}, Parpadeos: {self.liveness_detector.blink_counter}")
                        else:
                            cv2.putText(frame, "Sin landmarks de ojos válidos", 
                                       (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Verificar vida solo si tenemos landmarks válidos
                    if left_eye and right_eye and len(left_eye) >= 4 and len(right_eye) >= 4:
                        is_alive, liveness_info = self.liveness_detector.check_liveness(left_eye, right_eye, bbox)
                    else:
                        # Fallback a modo simple si no hay landmarks
                        movement = self.liveness_detector.detect_head_movement(bbox)
                        is_alive = movement > 20.0
                        liveness_info = {"fallback": True, "movement": movement}
            
            # Determinar acceso
            access_granted = (person_name is not None and is_real and is_alive)
            
            # Crear etiquetas
            if person_name:
                name_label = f"{person_name} ({confidence:.2f})"
            else:
                name_label = "Desconocido"
            
            # Agregar información de estado
            status_parts = []
            if self.enable_antispoof.get():
                status_parts.append(f"Real: {spoof_confidence:.2f}")
            if self.enable_liveness.get():
                status_parts.append(f"Vivo: {'Sí' if is_alive else 'No'}")
            
            if status_parts:
                status_label = " | ".join(status_parts)
            else:
                status_label = ""
            
            # Determinar color
            if access_granted:
                color = (0, 255, 0)  # Verde
                access_text = "ACCESO PERMITIDO"
            else:
                color = (0, 0, 255)  # Rojo
                access_text = "ACCESO DENEGADO"
            
            # Dibujar información
            frame = self.face_detector.draw_face_box(frame, bbox, name_label, color)
            
            # Agregar información adicional
            y_offset = bbox[1] + bbox[3] + 30
            if status_label:
                cv2.putText(frame, status_label, (bbox[0], y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
            
            cv2.putText(frame, access_text, (bbox[0], y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return frame
            
        except Exception as e:
            print(f"Error procesando frame: {e}")
            cv2.putText(frame, f"Error: {str(e)[:50]}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
    
    def update_fps(self):
        """Actualizar cálculo de FPS"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.fps_text.set(f"FPS: {self.fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def update_video_display(self, frame):
        """Actualizar display de video en la interfaz"""
        try:
            # Redimensionar frame para canvas
            frame_resized = cv2.resize(frame, (640, 480))
            
            # Convertir BGR a RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convertir a ImageTk
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Actualizar canvas en hilo principal
            self.root.after(0, self._update_canvas, photo)
            
        except Exception as e:
            print(f"Error actualizando display: {e}")
    
    def _update_canvas(self, photo):
        """Actualizar canvas (debe ejecutarse en hilo principal)"""
        self.video_canvas.delete("all")
        self.video_canvas.create_image(320, 240, image=photo)
        self.video_canvas.image = photo  # Mantener referencia
    
    def capture_frame(self):
        """Capturar frame actual"""
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Captura", f"Frame guardado como {filename}")
    
    def start_interactive_registration(self):
        """Iniciar registro interactivo"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Ingrese un nombre")
            return
        
        if name.lower() in self.face_recognizer.face_database:
            messagebox.showerror("Error", f"La persona '{name}' ya existe")
            return
        
        # Crear ventana de registro
        self.create_registration_window(name)
    
    def create_registration_window(self, person_name):
        """Crear ventana de registro interactivo"""
        reg_window = tk.Toplevel(self.root)
        reg_window.title(f"Registrando: {person_name}")
        reg_window.geometry("800x600")
        reg_window.configure(bg='#2c3e50')
        
        # Variables
        captured_faces = []
        max_captures = 10
        
        # Canvas para video
        canvas = tk.Canvas(reg_window, width=640, height=480, bg='black')
        canvas.pack(pady=10)
        
        # Información
        info_label = tk.Label(reg_window, text=f"Capturando rostros para {person_name}", 
                            font=('Arial', 12), bg='#2c3e50', fg='white')
        info_label.pack()
        
        capture_label = tk.Label(reg_window, text=f"Capturas: 0/{max_captures}", 
                               font=('Arial', 10), bg='#2c3e50', fg='white')
        capture_label.pack()
        
        # Botones
        button_frame = tk.Frame(reg_window, bg='#2c3e50')
        button_frame.pack(pady=10)
        
        def capture_face():
            if len(captured_faces) < max_captures and self.current_frame is not None:
                bbox, _ = self.face_detector.detect_face_and_landmarks(self.current_frame)
                if bbox is not None:
                    face_crop = self.face_detector.crop_face(self.current_frame, bbox)
                    if face_crop is not None:
                        captured_faces.append(face_crop)
                        capture_label.config(text=f"Capturas: {len(captured_faces)}/{max_captures}")
                        
                        if len(captured_faces) >= 3:
                            finish_button.config(state=tk.NORMAL)
        
        def finish_registration():
            if len(captured_faces) < 3:
                messagebox.showerror("Error", "Se necesitan al menos 3 capturas")
                return
            
            try:
                # Registrar persona
                success = self.face_recognizer.add_person_to_database(
                    person_name, captured_faces, method='average'
                )
                
                if success:
                    # Agregar a database manager
                    embeddings = self.face_recognizer.face_database[person_name]
                    metadata = {
                        'captures': len(captured_faces),
                        'registration_method': 'interactive_gui'
                    }
                    
                    self.database_manager.add_person(person_name, embeddings, metadata)
                    self.database_manager.save_database()
                    
                    messagebox.showinfo("Éxito", f"Persona '{person_name}' registrada exitosamente")
                    self.refresh_person_list()
                    reg_window.destroy()
                else:
                    messagebox.showerror("Error", "Error registrando persona")
            
            except Exception as e:
                messagebox.showerror("Error", f"Error en registro: {e}")
        
        capture_button = tk.Button(button_frame, text="Capturar (ESPACIO)", 
                                 command=capture_face, bg='#3498db', fg='white')
        capture_button.pack(side=tk.LEFT, padx=5)
        
        finish_button = tk.Button(button_frame, text="Finalizar", 
                                command=finish_registration, state=tk.DISABLED,
                                bg='#27ae60', fg='white')
        finish_button.pack(side=tk.LEFT, padx=5)
        
        cancel_button = tk.Button(button_frame, text="Cancelar", 
                                command=reg_window.destroy, bg='#e74c3c', fg='white')
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Vincular tecla ESPACIO
        def on_space(event):
            capture_face()
        
        reg_window.bind('<space>', on_space)
        reg_window.focus_set()
        
        # Loop de actualización del video en ventana de registro
        def update_reg_video():
            if reg_window.winfo_exists() and self.current_frame is not None:
                # Detectar rostro y dibujar
                frame_copy = self.current_frame.copy()
                bbox, _ = self.face_detector.detect_face_and_landmarks(frame_copy)
                
                if bbox is not None:
                    frame_copy = self.face_detector.draw_face_box(
                        frame_copy, bbox, f"Listo para capturar", (0, 255, 0)
                    )
                else:
                    cv2.putText(frame_copy, "Posiciónese frente a la cámara", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Mostrar en canvas
                frame_resized = cv2.resize(frame_copy, (640, 480))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                
                canvas.delete("all")
                canvas.create_image(320, 240, image=photo)
                canvas.image = photo
                
                reg_window.after(30, update_reg_video)
        
        # Iniciar actualización
        if self.is_running:
            update_reg_video()
        else:
            messagebox.showwarning("Advertencia", "Inicie la cámara primero")
            reg_window.destroy()
    
    def open_config_window(self):
        """Abrir ventana de configuración"""
        try:
            from gui.config_window import ConfigWindow
            ConfigWindow(self.root)
        except ImportError:
            # Si no se puede importar, usar la versión local
            try:
                import sys
                from pathlib import Path
                sys.path.append(str(Path(__file__).parent / "gui"))
                from config_window import ConfigWindow
                ConfigWindow(self.root)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo abrir ventana de configuración: {e}")
    
    def open_stats_window(self):
        """Abrir ventana de estadísticas"""
        try:
            from gui.stats_window import StatsWindow
            StatsWindow(self.root, self)
        except ImportError:
            try:
                import sys
                from pathlib import Path
                sys.path.append(str(Path(__file__).parent / "gui"))
                from stats_window import StatsWindow
                StatsWindow(self.root, self)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo abrir ventana de estadísticas: {e}")
    
    def register_from_folder(self):
        """Registrar persona desde carpeta"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Ingrese un nombre")
            return
        
        folder_path = filedialog.askdirectory(title="Seleccionar carpeta con imágenes")
        if not folder_path:
            return
        
        try:
            # Mostrar ventana de progreso
            progress_window = self.create_progress_window("Registrando persona...")
            
            def register_task():
                try:
                    from utils.image_processing import ImageProcessor, ImageValidator
                    
                    processor = ImageProcessor()
                    validator = ImageValidator()
                    
                    # Buscar imágenes
                    folder_path_obj = Path(folder_path)
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
                    image_files = []
                    
                    for ext in image_extensions:
                        image_files.extend(folder_path_obj.glob(f"*{ext}"))
                        image_files.extend(folder_path_obj.glob(f"*{ext.upper()}"))
                    
                    if not image_files:
                        self.root.after(0, lambda: messagebox.showerror("Error", "No se encontraron imágenes"))
                        progress_window.destroy()
                        return
                    
                    # Procesar imágenes
                    valid_faces = []
                    total_images = len(image_files)
                    
                    for i, image_path in enumerate(image_files):
                        try:
                            # Actualizar progreso
                            progress = (i / total_images) * 100
                            self.root.after(0, lambda p=progress: self.update_progress(progress_window, p, f"Procesando imagen {i+1}/{total_images}"))
                            
                            # Cargar y procesar imagen
                            image = processor.load_image(str(image_path))
                            if image is None:
                                continue
                            
                            # Detectar rostro
                            bbox, _ = self.face_detector.detect_face_and_landmarks(image)
                            if bbox is None:
                                continue
                            
                            # Extraer rostro
                            face_crop = self.face_detector.crop_face(image, bbox, size=Config.FACE_CROP_SIZE)
                            if face_crop is None:
                                continue
                            
                            # Verificar calidad
                            is_good, _ = validator.check_image_quality(face_crop)
                            if is_good:
                                valid_faces.append(face_crop)
                                
                        except Exception as e:
                            print(f"Error procesando {image_path}: {e}")
                            continue
                    
                    # Registrar si hay suficientes rostros
                    if len(valid_faces) >= 3:
                        success = self.face_recognizer.add_person_to_database(
                            name, valid_faces, method='average'
                        )
                        
                        if success:
                            embeddings = self.face_recognizer.face_database[name]
                            metadata = {
                                'total_images': total_images,
                                'valid_faces': len(valid_faces),
                                'source_folder': str(folder_path),
                                'registration_method': 'folder_gui'
                            }
                            
                            self.database_manager.add_person(name, embeddings, metadata)
                            self.database_manager.save_database()
                            
                            self.root.after(0, lambda: messagebox.showinfo("Éxito", f"Persona '{name}' registrada con {len(valid_faces)} rostros válidos"))
                            self.root.after(0, self.refresh_person_list)
                        else:
                            self.root.after(0, lambda: messagebox.showerror("Error", "Error registrando embeddings"))
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Error", f"Se necesitan al menos 3 rostros válidos, solo se encontraron {len(valid_faces)}"))
                    
                    progress_window.destroy()
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Error en registro: {e}"))
                    progress_window.destroy()
            
            # Ejecutar en hilo separado
            thread = threading.Thread(target=register_task)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error registrando desde carpeta: {e}")
    
    def register_from_image(self):
        """Registrar persona desde imagen única"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Ingrese un nombre")
            return
        
        image_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not image_path:
            return
        
        try:
            from utils.image_processing import ImageProcessor, ImageValidator
            
            processor = ImageProcessor()
            validator = ImageValidator()
            
            # Cargar imagen
            image = processor.load_image(image_path)
            if image is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen")
                return
            
            # Detectar rostro
            bbox, _ = self.face_detector.detect_face_and_landmarks(image)
            if bbox is None:
                messagebox.showerror("Error", "No se detectó rostro en la imagen")
                return
            
            # Extraer rostro
            face_crop = self.face_detector.crop_face(image, bbox, size=Config.FACE_CROP_SIZE)
            if face_crop is None:
                messagebox.showerror("Error", "No se pudo extraer rostro")
                return
            
            # Verificar calidad
            is_good, metrics = validator.check_image_quality(face_crop)
            if not is_good:
                result = messagebox.askyesno("Advertencia", 
                    "La imagen es de baja calidad. ¿Continuar de todos modos?")
                if not result:
                    return
            
            # Registrar
            success = self.face_recognizer.add_person_to_database(
                name, [face_crop], method='average'
            )
            
            if success:
                embeddings = self.face_recognizer.face_database[name]
                metadata = {
                    'source_image': image_path,
                    'registration_method': 'single_image_gui',
                    'quality_metrics': metrics
                }
                
                self.database_manager.add_person(name, embeddings, metadata)
                self.database_manager.save_database()
                
                messagebox.showinfo("Éxito", f"Persona '{name}' registrada desde imagen única\n\nNOTA: El registro con una sola imagen es menos confiable")
                self.refresh_person_list()
            else:
                messagebox.showerror("Error", "Error registrando embedding")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error registrando desde imagen: {e}")
    
    def create_progress_window(self, title="Procesando..."):
        """Crear ventana de progreso"""
        progress_window = tk.Toplevel(self.root)
        progress_window.title(title)
        progress_window.geometry("400x120")
        progress_window.configure(bg='#34495e')
        progress_window.resizable(False, False)
        
        # Centrar ventana
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Título
        title_label = tk.Label(progress_window, text=title, 
                              font=('Arial', 12), bg='#34495e', fg='white')
        title_label.pack(pady=10)
        
        # Barra de progreso
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, 
                                     maximum=100, length=300)
        progress_bar.pack(pady=5)
        
        # Etiqueta de estado
        status_label = tk.Label(progress_window, text="Iniciando...", 
                               bg='#34495e', fg='white')
        status_label.pack(pady=5)
        
        progress_window.progress_var = progress_var
        progress_window.status_label = status_label
        
        return progress_window
    
    def update_progress(self, window, value, text=""):
        """Actualizar ventana de progreso"""
        try:
            if window.winfo_exists():
                window.progress_var.set(value)
                if text:
                    window.status_label.config(text=text)
                window.update()
        except:
            pass
    
    def load_database(self):
        """Cargar base de datos existente"""
        try:
            for person_name in self.database_manager.list_persons():
                person_data = self.database_manager.get_person(person_name)
                if person_data:
                    embeddings, metadata = person_data
                    self.face_recognizer.face_database[person_name] = embeddings
            
            self.refresh_person_list()
            self.status_text.set(f"Base de datos cargada: {len(self.face_recognizer.face_database)} personas")
            
        except Exception as e:
            self.status_text.set(f"Error cargando base de datos: {e}")
    
    def refresh_person_list(self):
        """Actualizar lista de personas"""
        self.person_listbox.delete(0, tk.END)
        persons = self.database_manager.list_persons()
        
        for person in persons:
            self.person_listbox.insert(tk.END, person)
        
        self.person_count_text.set(f"Personas: {len(persons)}")
    
    def delete_selected_person(self):
        """Eliminar persona seleccionada"""
        selection = self.person_listbox.curselection()
        if not selection:
            messagebox.showwarning("Advertencia", "Seleccione una persona")
            return
        
        person_name = self.person_listbox.get(selection[0])
        
        if messagebox.askyesno("Confirmar", f"¿Eliminar a '{person_name}'?"):
            try:
                self.database_manager.remove_person(person_name)
                self.face_recognizer.remove_person(person_name)
                self.database_manager.save_database()
                self.refresh_person_list()
                messagebox.showinfo("Éxito", f"Persona '{person_name}' eliminada")
            except Exception as e:
                messagebox.showerror("Error", f"Error eliminando persona: {e}")
    
    def improve_registration(self):
        """Mejorar registro de persona seleccionada"""
        selection = self.person_listbox.curselection()
        if not selection:
            messagebox.showwarning("Advertencia", "Seleccione una persona para mejorar")
            return
        
        person_name = self.person_listbox.get(selection[0])
        
        result = messagebox.askyesno("Mejorar Registro", 
                                   f"¿Agregar más imágenes para mejorar el registro de '{person_name}'?\n\n"
                                   "Esto ayudará a que el reconocimiento sea más consistente.")
        
        if result:
            # Usar el nombre existente para el registro interactivo
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, person_name)
            
            # Primero eliminar el registro actual
            try:
                self.database_manager.remove_person(person_name)
                self.face_recognizer.remove_person(person_name)
            except:
                pass
            
            # Iniciar nuevo registro
            self.create_registration_window(person_name)
    
    def show_database_stats(self):
        """Mostrar estadísticas de la base de datos"""
        try:
            stats = self.database_manager.get_statistics()
            
            stats_text = f"""Estadísticas de la Base de Datos:
            
Total de personas: {stats.get('total_persons', 0)}
Total de embeddings: {stats.get('total_embeddings', 0)}
Dimensiones de embedding: {stats.get('embedding_dimensions', 'N/A')}
Personas con múltiples embeddings: {stats.get('persons_with_multiple_embeddings', 0)}
Promedio de embeddings por persona: {stats.get('average_embeddings_per_person', 0):.1f}
Tamaño de base de datos: {stats.get('database_size_mb', 0):.2f} MB"""
            
            messagebox.showinfo("Estadísticas", stats_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error obteniendo estadísticas: {e}")
    
    def test_camera(self):
        """Probar cámara"""
        try:
            from utils.camera_utils import detect_available_cameras
            cameras = detect_available_cameras()
            
            if cameras:
                camera_text = f"Cámaras disponibles: {cameras}"
                messagebox.showinfo("Prueba de Cámara", camera_text)
            else:
                messagebox.showerror("Prueba de Cámara", "No se encontraron cámaras")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error probando cámara: {e}")
    
    def test_detection(self):
        """Probar detección facial"""
        if not self.is_running:
            messagebox.showwarning("Advertencia", "Inicie la cámara primero")
            return
        
        messagebox.showinfo("Prueba de Detección", 
                           "Observe el video. Si ve un recuadro verde alrededor de su rostro, la detección funciona correctamente.")
    
    def test_all(self):
        """Probar todo el sistema"""
        try:
            # Verificar componentes
            issues = []
            
            if len(self.face_recognizer.face_database) == 0:
                issues.append("- No hay personas registradas")
            
            if not Config.ANTI_SPOOF_MODEL_PATH.exists():
                issues.append("- Modelo anti-spoofing no encontrado")
            
            if not self.is_running:
                issues.append("- Cámara no está iniciada")
            
            if issues:
                result_text = "Problemas encontrados:\n" + "\n".join(issues)
                messagebox.showwarning("Prueba del Sistema", result_text)
            else:
                messagebox.showinfo("Prueba del Sistema", 
                                   "✅ Todos los componentes funcionando correctamente")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en prueba del sistema: {e}")
    
    def reset_liveness(self):
        """Reiniciar detector de vida"""
        if self.liveness_detector:
            self.liveness_detector.reset()
            # Actualizar umbral si cambió
            if hasattr(self, 'ear_threshold'):
                self.liveness_detector.ear_threshold = self.ear_threshold.get()
            self.status_text.set("Detector de vida reiniciado")
            print(f"Detector reiniciado con umbral EAR: {self.ear_threshold.get() if hasattr(self, 'ear_threshold') else 'default'}")
    
    def toggle_confidence_debug(self):
        """Alternar modo debug para confianza"""
        self.show_confidence_debug = not self.show_confidence_debug
        status = "habilitado" if self.show_confidence_debug else "deshabilitado"
        self.status_text.set(f"Debug de confianza {status}")
        print(f"Debug de confianza: {status}")
    
    def toggle_eye_debug(self):
        """Alternar modo debug para ojos"""
        self.show_eye_debug = not self.show_eye_debug
        status = "habilitado" if self.show_eye_debug else "deshabilitado"
        self.status_text.set(f"Debug de ojos {status}")
        print(f"Debug de ojos: {status}")
    
    def on_closing(self):
        """Manejar cierre de ventana"""
        if self.is_running:
            self.stop_camera()
        
        try:
            self.database_manager.save_database()
        except:
            pass
        
        self.root.destroy()
    
    def run(self):
        """Ejecutar la aplicación"""
        # Actualizar frame actual para registro
        def update_current_frame():
            if self.is_running:
                self.current_frame = self.camera_manager.read_frame()
            self.root.after(50, update_current_frame)
        
        update_current_frame()
        self.root.mainloop()


# ===== CÓDIGO PRINCIPAL =====
if __name__ == "__main__":
    app = FaceRecognitionGUI()
    app.run()