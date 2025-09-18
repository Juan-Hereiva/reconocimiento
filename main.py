"""
Sistema principal de reconocimiento facial con detección de vida
"""
import cv2
import numpy as np
import time
import logging
import sys
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

# Configurar logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    """Sistema completo de reconocimiento facial"""
    
    def __init__(self):
        """Inicializar sistema"""
        logger.info("Inicializando sistema de reconocimiento facial...")
        
        # Crear directorios necesarios
        Config.create_directories()
        
        # Inicializar componentes
        self.face_detector = FaceDetector(
            confidence=Config.MP_FACE_CONFIDENCE,
            max_num_faces=Config.MP_MAX_NUM_FACES
        )
        
        self.face_recognizer = FaceRecognizer()
        
        self.anti_spoof_detector = AntiSpoofDetector(
            model_path=str(Config.ANTI_SPOOF_MODEL_PATH)
        )
        
        self.liveness_detector = LivenessDetector(
            ear_threshold=Config.EAR_THRESHOLD,
            consecutive_frames=Config.BLINK_CONSECUTIVE_FRAMES,
            time_window=Config.LIVENESS_TIME_WINDOW
        )
        
        self.camera_manager = CameraManager(
            camera_index=Config.CAMERA_INDEX,
            width=Config.CAMERA_WIDTH,
            height=Config.CAMERA_HEIGHT,
            fps=Config.CAMERA_FPS
        )
        
        self.database_manager = DatabaseManager(str(Config.FACE_DATABASE_PATH))
        
        # Estado del sistema
        self.is_running = False
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
        # Cargar base de datos existente en el reconocedor
        self._load_face_database()
        
        logger.info("Sistema inicializado exitosamente")
    
    def _load_face_database(self):
        """Cargar base de datos de rostros en el reconocedor"""
        try:
            for person_name in self.database_manager.list_persons():
                person_data = self.database_manager.get_person(person_name)
                if person_data:
                    embeddings, metadata = person_data
                    self.face_recognizer.face_database[person_name] = embeddings
            
            logger.info(f"Cargadas {len(self.face_recognizer.face_database)} personas en el reconocedor")
            
        except Exception as e:
            logger.error(f"Error cargando base de datos: {e}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesar frame completo
        
        Args:
            frame: Frame BGR de entrada
            
        Returns:
            np.ndarray: Frame procesado con anotaciones
        """
        try:
            # Detectar rostro y landmarks
            bbox, landmarks = self.face_detector.detect_face_and_landmarks(frame)
            
            if bbox is None:
                # No hay rostro detectado
                cv2.putText(frame, "No se detecta rostro", (20, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return frame
            
            # Extraer rostro para reconocimiento
            face_crop = self.face_detector.crop_face(
                frame, bbox, size=Config.FACE_CROP_SIZE
            )
            
            if face_crop is None:
                return frame
            
            # Reconocimiento facial
            person_name, confidence = self.face_recognizer.identify_person(
                face_crop, threshold=Config.FACE_RECOGNITION_THRESHOLD
            )
            
            # Detección anti-spoofing
            # Detección anti-spoofing
            anti_spoof_crop = self.face_detector.crop_face(
                frame, bbox, size=Config.ANTI_SPOOF_INPUT_SIZE
            )

            if anti_spoof_crop is not None and anti_spoof_crop.size > 0:
                is_real_face, spoof_confidence = self.anti_spoof_detector.detect_spoofing(
                    anti_spoof_crop
                )
            else:
                is_real_face = False
                spoof_confidence = 0.0
                logger.warning("anti_spoof_crop no válido: no se pudo aplicar anti-spoofing")
            
            # Detección de vida
            left_eye, right_eye = self.face_detector.get_eye_landmarks(
                landmarks, frame.shape, Config.LEFT_EYE_INDICES, Config.RIGHT_EYE_INDICES
            )
            
            is_alive, liveness_results = self.liveness_detector.check_liveness(
                left_eye, right_eye, bbox
            )
            
            # Decisión final de acceso
            is_known_person = person_name is not None
            is_real_person = is_real_face
            
            access_granted = is_known_person and is_real_person and is_alive
            
            # Crear etiquetas para mostrar
            if person_name:
                name_label = f"{person_name} ({confidence:.2f})"
            else:
                name_label = "Desconocido"
            
            status_label = f"Real: {spoof_confidence:.2f} | Vivo: {is_alive}"
            
            # Determinar color del bounding box
            if access_granted:
                bbox_color = Config.BBOX_COLOR_ALLOWED
                access_text = "ACCESO PERMITIDO"
            else:
                bbox_color = Config.BBOX_COLOR_DENIED
                access_text = "ACCESO DENEGADO"
            
            # Dibujar bounding box y etiquetas
            frame = self.face_detector.draw_face_box(
                frame, bbox, name_label, bbox_color, Config.BBOX_THICKNESS
            )
            
            # Agregar información adicional
            y_offset = bbox[1] + bbox[3] + 30
            cv2.putText(frame, status_label, (bbox[0], y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE, 
                       bbox_color, Config.FONT_THICKNESS)
            
            cv2.putText(frame, access_text, (bbox[0], y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE, 
                       bbox_color, Config.FONT_THICKNESS)
            
            # Log de evento si hay acceso
            if access_granted:
                logger.info(f"Acceso permitido: {person_name} (conf: {confidence:.2f})")
            elif is_known_person:
                logger.warning(f"Acceso denegado para {person_name}: real={is_real_person}, vivo={is_alive}")
            
            return frame
            
        except Exception as e:
            logger.error(f"Error procesando frame: {e}")
            cv2.putText(frame, "Error en procesamiento", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
    
    def update_fps(self):
        """Actualizar cálculo de FPS"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:  # Actualizar cada segundo
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def draw_system_info(self, frame: np.ndarray) -> np.ndarray:
        """
        Dibujar información del sistema en el frame
        
        Args:
            frame: Frame BGR
            
        Returns:
            np.ndarray: Frame con información del sistema
        """
        try:
            h, w = frame.shape[:2]
            
            # Información de FPS
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, fps_text, (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Información de personas registradas
            num_persons = len(self.face_recognizer.face_database)
            persons_text = f"Personas: {num_persons}"
            cv2.putText(frame, persons_text, (w - 150, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Estado del detector de vida
            liveness_status = self.liveness_detector.get_status()
            blinks_text = f"Parpadeos: {liveness_status['total_blinks']}"
            cv2.putText(frame, blinks_text, (w - 150, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Instrucciones
            cv2.putText(frame, "Presiona 'q' para salir", (20, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Presiona 'r' para reiniciar liveness", (20, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error dibujando información del sistema: {e}")
            return frame
    
    def run(self):
        """Ejecutar sistema principal"""
        logger.info("Iniciando sistema de reconocimiento facial...")
        
        try:
            # Abrir cámara
            if not self.camera_manager.open_camera():
                logger.error("No se pudo abrir la cámara")
                return False
            
            self.is_running = True
            logger.info("Sistema ejecutándose. Presiona 'q' para salir.")
            
            while self.is_running:
                # Leer frame
                frame = self.camera_manager.read_frame()
                if frame is None:
                    logger.warning("Frame vacío, continuando...")
                    continue
                
                # Procesar frame
                processed_frame = self.process_frame(frame)
                
                # Agregar información del sistema
                processed_frame = self.draw_system_info(processed_frame)
                
                # Actualizar FPS
                self.update_fps()
                
                # Mostrar frame
                cv2.imshow("Sistema de Reconocimiento Facial", processed_frame)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Saliendo del sistema...")
                    break
                elif key == ord('r'):
                    logger.info("Reiniciando detector de vida...")
                    self.liveness_detector.reset()
                elif key == ord('s'):
                    # Guardar screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    logger.info(f"Screenshot guardado: {screenshot_path}")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Interrupción por teclado")
            return True
            
        except Exception as e:
            logger.error(f"Error en ejecución del sistema: {e}")
            return False
            
        finally:
            self.stop()
    
    def stop(self):
        """Detener sistema"""
        try:
            self.is_running = False
            
            # Cerrar cámara
            self.camera_manager.close_camera()
            
            # Cerrar ventanas
            cv2.destroyAllWindows()
            
            # Guardar base de datos
            self.database_manager.save_database()
            
            logger.info("Sistema detenido correctamente")
            
        except Exception as e:
            logger.error(f"Error deteniendo sistema: {e}")
    
    def add_person_interactive(self):
        """Agregar persona de forma interactiva"""
        try:
            print("\n=== Agregar Nueva Persona ===")
            person_name = input("Ingrese el nombre de la persona: ").strip()
            
            if not person_name:
                print("Nombre inválido")
                return False
            
            if person_name.lower() in self.face_recognizer.face_database:
                print(f"La persona '{person_name}' ya existe en la base de datos")
                return False
            
            # Abrir cámara para captura
            if not self.camera_manager.open_camera():
                print("No se pudo abrir la cámara")
                return False
            
            captured_faces = []
            print(f"\nCapturando rostros para {person_name}...")
            print("Presiona ESPACIO para capturar, 'q' para finalizar")
            
            while len(captured_faces) < 10:  # Máximo 10 capturas
                frame = self.camera_manager.read_frame()
                if frame is None:
                    continue
                
                # Detectar rostro
                bbox, landmarks = self.face_detector.detect_face_and_landmarks(frame)

                if bbox is None:
                    cv2.putText(frame, "No se detecta rostro", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    return frame

                
                if bbox is not None:
                    # Dibujar bounding box
                    frame = self.face_detector.draw_face_box(
                        frame, bbox, f"Capturas: {len(captured_faces)}/10", (0, 255, 0)
                    )
                    
                    # Mostrar instrucciones
                    cv2.putText(frame, "ESPACIO: Capturar | Q: Finalizar", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "No se detecta rostro", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow("Captura de Rostro", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and bbox is not None:
                    # Capturar rostro
                    face_crop = self.face_detector.crop_face(frame, bbox)
                    if face_crop is not None:
                        captured_faces.append(face_crop)
                        print(f"Rostro capturado {len(captured_faces)}/10")
                
                elif key == ord('q'):
                    break
            
            cv2.destroyWindow("Captura de Rostro")
            
            if len(captured_faces) < 3:
                print("Se necesitan al menos 3 capturas para registrar una persona")
                return False
            
            # Generar embeddings
            print("Generando embeddings...")
            success = self.face_recognizer.add_person_to_database(
                person_name, captured_faces, method='average'
            )
            
            if success:
                # Agregar a database manager
                embeddings = self.face_recognizer.face_database[person_name]
                metadata = {
                    'captures': len(captured_faces),
                    'registration_method': 'interactive'
                }
                
                self.database_manager.add_person(person_name, embeddings, metadata)
                self.database_manager.save_database()
                
                print(f"Persona '{person_name}' registrada exitosamente con {len(captured_faces)} capturas")
                return True
            else:
                print("Error registrando persona")
                return False
                
        except Exception as e:
            logger.error(f"Error en registro interactivo: {e}")
            print(f"Error: {e}")
            return False
    
    def list_registered_persons(self):
        """Listar personas registradas"""
        try:
            persons = self.database_manager.list_persons()
            
            if not persons:
                print("No hay personas registradas")
                return
            
            print("\n=== Personas Registradas ===")
            for i, person_name in enumerate(persons, 1):
                person_data = self.database_manager.get_person(person_name)
                if person_data:
                    _, metadata = person_data
                    added_date = metadata.get('added_date', 'Desconocido')
                    num_embeddings = metadata.get('num_embeddings', 'Desconocido')
                    print(f"{i}. {person_name} (Embeddings: {num_embeddings}, Fecha: {added_date})")
            
        except Exception as e:
            logger.error(f"Error listando personas: {e}")
            print(f"Error: {e}")
    
    def show_menu(self):
        """Mostrar menú interactivo"""
        while True:
            print("\n" + "="*50)
            print("SISTEMA DE RECONOCIMIENTO FACIAL")
            print("="*50)
            print("1. Ejecutar sistema de reconocimiento")
            print("2. Agregar nueva persona")
            print("3. Listar personas registradas")
            print("4. Probar cámara")
            print("5. Ver estadísticas de la base de datos")
            print("6. Salir")
            print("="*50)
            
            try:
                choice = input("Seleccione una opción (1-6): ").strip()
                
                if choice == '1':
                    self.run()
                elif choice == '2':
                    self.add_person_interactive()
                elif choice == '3':
                    self.list_registered_persons()
                elif choice == '4':
                    self.camera_manager.test_camera()
                elif choice == '5':
                    stats = self.database_manager.get_statistics()
                    print("\n=== Estadísticas de la Base de Datos ===")
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                elif choice == '6':
                    print("¡Hasta luego!")
                    break
                else:
                    print("Opción inválida")
                    
            except KeyboardInterrupt:
                print("\n¡Hasta luego!")
                break
            except Exception as e:
                logger.error(f"Error en menú: {e}")
                print(f"Error: {e}")

def main():
    """Función principal"""
    try:
        # Crear sistema
        system = FaceRecognitionSystem()
        
        # Mostrar menú
        system.show_menu()
        
    except Exception as e:
        logger.error(f"Error en función principal: {e}")
        print(f"Error crítico: {e}")
    
    finally:
        # Limpiar recursos
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
