"""
Script para probar el sistema de reconocimiento facial
"""
import os
import sys
import cv2
import numpy as np
import time
import argparse
import logging
from pathlib import Path

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.face_detector import FaceDetector
from models.face_recognizer import FaceRecognizer
from models.anti_spoof import AntiSpoofDetector
from models.liveness_detector import LivenessDetector
from utils.camera_utils import CameraManager, detect_available_cameras
from utils.database_manager import DatabaseManager
from utils.image_processing import ImageProcessor

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemTester:
    """Probador del sistema de reconocimiento facial"""
    
    def __init__(self):
        """Inicializar probador"""
        logger.info("Inicializando SystemTester...")
        
        # Crear directorios necesarios
        Config.create_directories()
        
        # Inicializar componentes
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.anti_spoof_detector = AntiSpoofDetector(str(Config.ANTI_SPOOF_MODEL_PATH))
        self.liveness_detector = LivenessDetector()
        self.database_manager = DatabaseManager(str(Config.FACE_DATABASE_PATH))
        self.image_processor = ImageProcessor()
        
        # Cargar base de datos
        self._load_face_database()
        
        logger.info("SystemTester inicializado")
    
    def _load_face_database(self):
        """Cargar base de datos de rostros"""
        try:
            for person_name in self.database_manager.list_persons():
                person_data = self.database_manager.get_person(person_name)
                if person_data:
                    embeddings, metadata = person_data
                    self.face_recognizer.face_database[person_name] = embeddings
            
            logger.info(f"Cargadas {len(self.face_recognizer.face_database)} personas")
        except Exception as e:
            logger.error(f"Error cargando base de datos: {e}")
    
    def test_camera_detection(self):
        """Probar detección de cámaras disponibles"""
        try:
            print("\n=== PRUEBA DE DETECCIÓN DE CÁMARAS ===")
            
            available_cameras = detect_available_cameras(max_cameras=5)
            
            if not available_cameras:
                print("❌ No se encontraron cámaras disponibles")
                return False
            
            print(f"✅ Cámaras encontradas: {available_cameras}")
            
            # Probar cada cámara
            for camera_idx in available_cameras:
                print(f"\nProbando cámara {camera_idx}...")
                
                camera = CameraManager(camera_idx)
                if camera.open_camera():
                    info = camera.get_camera_info()
                    print(f"  ✅ Cámara {camera_idx}: {info['width']}x{info['height']} @ {info['fps']}fps")
                    
                    # Capturar frame de prueba
                    frame = camera.read_frame()
                    if frame is not None:
                        print(f"  ✅ Captura exitosa: {frame.shape}")
                    else:
                        print(f"  ❌ Error en captura")
                    
                    camera.close_camera()
                else:
                    print(f"  ❌ No se pudo abrir cámara {camera_idx}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error probando cámaras: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def test_face_detection(self, camera_index: int = 0, duration: int = 10):
        """
        Probar detección facial
        
        Args:
            camera_index: Índice de la cámara
            duration: Duración de la prueba en segundos
        """
        try:
            print(f"\n=== PRUEBA DE DETECCIÓN FACIAL (Cámara {camera_index}) ===")
            
            camera = CameraManager(camera_index)
            if not camera.open_camera():
                print("❌ No se pudo abrir la cámara")
                return False
            
            start_time = time.time()
            detection_count = 0
            frame_count = 0
            
            print(f"Probando detección por {duration} segundos...")
            print("Presiona 'q' para salir anticipadamente")
            
            while time.time() - start_time < duration:
                frame = camera.read_frame()
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Detectar rostro
                bbox, landmarks = self.face_detector.detect_face_and_landmarks(frame)
                
                if bbox is not None:
                    detection_count += 1
                    
                    # Dibujar bounding box
                    frame = self.face_detector.draw_face_box(
                        frame, bbox, f"Detección #{detection_count}", (0, 255, 0)
                    )
                
                # Mostrar estadísticas
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
                
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Detecciones: {detection_count}/{frame_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Tasa: {detection_rate:.1f}%", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Prueba de Detección Facial", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            camera.close_camera()
            cv2.destroyAllWindows()
            
            # Resultados
            final_fps = frame_count / (time.time() - start_time)
            final_detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
            
            print(f"\n📊 RESULTADOS:")
            print(f"  Frames procesados: {frame_count}")
            print(f"  Detecciones exitosas: {detection_count}")
            print(f"  FPS promedio: {final_fps:.1f}")
            print(f"  Tasa de detección: {final_detection_rate:.1f}%")
            
            if final_detection_rate > 70:
                print("✅ Detección facial funcionando correctamente")
                return True
            else:
                print("⚠️ Baja tasa de detección, verificar iluminación y posición")
                return False
            
        except Exception as e:
            logger.error(f"Error probando detección facial: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def test_face_recognition(self, test_image_path: str = None):
        """
        Probar reconocimiento facial
        
        Args:
            test_image_path: Ruta de imagen de prueba (opcional)
        """
        try:
            print(f"\n=== PRUEBA DE RECONOCIMIENTO FACIAL ===")
            
            if len(self.face_recognizer.face_database) == 0:
                print("❌ No hay personas registradas en la base de datos")
                print("Use el script enroll_faces.py para registrar personas primero")
                return False
            
            print(f"Personas registradas: {list(self.face_recognizer.face_database.keys())}")
            
            if test_image_path:
                # Probar con imagen específica
                return self._test_recognition_on_image(test_image_path)
            else:
                # Probar con cámara
                return self._test_recognition_on_camera()
            
        except Exception as e:
            logger.error(f"Error probando reconocimiento: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def _test_recognition_on_image(self, image_path: str) -> bool:
        """Probar reconocimiento en imagen"""
        try:
            print(f"Probando reconocimiento en imagen: {image_path}")
            
            # Cargar imagen
            image = self.image_processor.load_image(image_path)
            if image is None:
                print(f"❌ No se pudo cargar la imagen: {image_path}")
                return False
            
            # Detectar rostro
            bbox, landmarks = self.face_detector.detect_face_and_landmarks(image)
            if bbox is None:
                print("❌ No se detectó rostro en la imagen")
                return False
            
            # Extraer rostro
            face_crop = self.face_detector.crop_face(image, bbox)
            if face_crop is None:
                print("❌ No se pudo extraer rostro")
                return False
            
            # Reconocer
            person_name, confidence = self.face_recognizer.identify_person(
                face_crop, threshold=Config.FACE_RECOGNITION_THRESHOLD
            )
            
            # Mostrar resultado
            result_image = self.face_detector.draw_face_box(
                image, bbox, 
                f"{person_name or 'Desconocido'} ({confidence:.2f})",
                (0, 255, 0) if person_name else (0, 0, 255)
            )
            
            cv2.imshow("Resultado de Reconocimiento", result_image)
            print(f"Resultado: {person_name or 'Desconocido'} (confianza: {confidence:.2f})")
            print("Presiona cualquier tecla para continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            logger.error(f"Error en reconocimiento de imagen: {e}")
            return False
    
    def _test_recognition_on_camera(self) -> bool:
        """Probar reconocimiento con cámara"""
        try:
            print("Probando reconocimiento con cámara...")
            print("Presiona 'q' para salir")
            
            camera = CameraManager()
            if not camera.open_camera():
                print("❌ No se pudo abrir la cámara")
                return False
            
            recognition_count = 0
            frame_count = 0
            
            while True:
                frame = camera.read_frame()
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Detectar rostro
                bbox, landmarks = self.face_detector.detect_face_and_landmarks(frame)
                
                if bbox is not None:
                    # Extraer rostro
                    face_crop = self.face_detector.crop_face(frame, bbox)
                    
                    if face_crop is not None:
                        # Reconocer
                        person_name, confidence = self.face_recognizer.identify_person(
                            face_crop, threshold=Config.FACE_RECOGNITION_THRESHOLD
                        )
                        
                        if person_name:
                            recognition_count += 1
                        
                        # Dibujar resultado
                        label = f"{person_name or 'Desconocido'} ({confidence:.2f})"
                        color = (0, 255, 0) if person_name else (0, 0, 255)
                        frame = self.face_detector.draw_face_box(frame, bbox, label, color)
                
                # Mostrar estadísticas
                recognition_rate = (recognition_count / frame_count) * 100 if frame_count > 0 else 0
                cv2.putText(frame, f"Reconocimientos: {recognition_count}/{frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Tasa: {recognition_rate:.1f}%", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Prueba de Reconocimiento", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            camera.close_camera()
            cv2.destroyAllWindows()
            
            print(f"\n📊 RESULTADOS:")
            print(f"  Frames con rostro: {frame_count}")
            print(f"  Reconocimientos exitosos: {recognition_count}")
            print(f"  Tasa de reconocimiento: {recognition_rate:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en reconocimiento con cámara: {e}")
            return False
    
    def test_anti_spoofing(self, camera_index: int = 0):
        """
        Probar detección anti-spoofing
        
        Args:
            camera_index: Índice de la cámara
        """
        try:
            print(f"\n=== PRUEBA DE ANTI-SPOOFING ===")
            
            # Verificar si existe modelo entrenado
            if not Config.ANTI_SPOOF_MODEL_PATH.exists():
                print("⚠️ Modelo anti-spoofing no encontrado")
                print("Entrenar modelo con: python scripts/train_antispoof.py")
                return False
            
            print("Probando detección anti-spoofing...")
            print("Prueba mostrando tu rostro real, luego una foto de tu rostro")
            print("Presiona 'q' para salir")
            
            camera = CameraManager(camera_index)
            if not camera.open_camera():
                print("❌ No se pudo abrir la cámara")
                return False
            
            real_detections = 0
            fake_detections = 0
            total_detections = 0
            
            while True:
                frame = camera.read_frame()
                if frame is None:
                    continue
                
                # Detectar rostro
                bbox, landmarks = self.face_detector.detect_face_and_landmarks(frame)
                
                if bbox is not None:
                    # Extraer rostro para anti-spoofing
                    face_crop = self.face_detector.crop_face(
                        frame, bbox, size=Config.ANTI_SPOOF_INPUT_SIZE
                    )
                    
                    if face_crop is not None:
                        # Detectar spoofing
                        is_real, confidence = self.anti_spoof_detector.detect_spoofing(face_crop)
                        
                        total_detections += 1
                        if is_real:
                            real_detections += 1
                        else:
                            fake_detections += 1
                        
                        # Dibujar resultado
                        status = "REAL" if is_real else "FAKE"
                        color = (0, 255, 0) if is_real else (0, 0, 255)
                        label = f"{status} ({confidence:.2f})"
                        
                        frame = self.face_detector.draw_face_box(frame, bbox, label, color)
                
                # Mostrar estadísticas
                if total_detections > 0:
                    real_rate = (real_detections / total_detections) * 100
                    fake_rate = (fake_detections / total_detections) * 100
                    
                    cv2.putText(frame, f"Real: {real_detections} ({real_rate:.1f}%)", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Fake: {fake_detections} ({fake_rate:.1f}%)", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("Prueba Anti-Spoofing", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            camera.close_camera()
            cv2.destroyAllWindows()
            
            print(f"\n📊 RESULTADOS:")
            print(f"  Total detecciones: {total_detections}")
            print(f"  Clasificadas como reales: {real_detections}")
            print(f"  Clasificadas como falsas: {fake_detections}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error probando anti-spoofing: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def test_liveness_detection(self, camera_index: int = 0):
        """
        Probar detección de vida
        
        Args:
            camera_index: Índice de la cámara
        """
        try:
            print(f"\n=== PRUEBA DE DETECCIÓN DE VIDA ===")
            print("Parpadea varias veces para probar la detección de vida")
            print("Presiona 'r' para reiniciar, 'q' para salir")
            
            camera = CameraManager(camera_index)
            if not camera.open_camera():
                print("❌ No se pudo abrir la cámara")
                return False
            
            while True:
                frame = camera.read_frame()
                if frame is None:
                    continue
                
                # Detectar rostro y landmarks
                bbox, landmarks = self.face_detector.detect_face_and_landmarks(frame)
                
                if bbox is not None and landmarks is not None:
                    # Extraer landmarks de ojos
                    left_eye, right_eye = self.face_detector.get_eye_landmarks(
                        landmarks, frame.shape, 
                        Config.LEFT_EYE_INDICES, Config.RIGHT_EYE_INDICES
                    )
                    
                    # Verificar vida
                    is_alive, liveness_results = self.liveness_detector.check_liveness(
                        left_eye, right_eye, bbox
                    )
                    
                    # Dibujar resultado
                    status = "VIVO" if is_alive else "NO VIVO"
                    color = (0, 255, 0) if is_alive else (0, 0, 255)
                    
                    frame = self.face_detector.draw_face_box(frame, bbox, status, color)
                    
                    # Mostrar información detallada
                    y_offset = bbox[3] + 20
                    cv2.putText(frame, f"Parpadeos: {liveness_results['total_blinks']}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.putText(frame, f"Movimiento: {'Si' if liveness_results['has_movement'] else 'No'}", 
                               (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.putText(frame, f"Patrón natural: {'Si' if liveness_results['natural_pattern'] else 'No'}", 
                               (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Prueba de Detección de Vida", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.liveness_detector.reset()
                    print("Detector de vida reiniciado")
            
            camera.close_camera()
            cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            logger.error(f"Error probando detección de vida: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def test_full_system(self, camera_index: int = 0):
        """
        Probar sistema completo
        
        Args:
            camera_index: Índice de la cámara
        """
        try:
            print(f"\n=== PRUEBA DEL SISTEMA COMPLETO ===")
            
            if len(self.face_recognizer.face_database) == 0:
                print("❌ No hay personas registradas")
                return False
            
            print("Probando sistema completo...")
            print("El sistema debe: detectar rostro, reconocer persona, verificar anti-spoofing y vida")
            print("Presiona 'q' para salir, 'r' para reiniciar liveness")
            
            camera = CameraManager(camera_index)
            if not camera.open_camera():
                print("❌ No se pudo abrir la cámara")
                return False
            
            access_granted_count = 0
            access_denied_count = 0
            
            while True:
                frame = camera.read_frame()
                if frame is None:
                    continue
                
                # Detectar rostro
                bbox, landmarks = self.face_detector.detect_face_and_landmarks(frame)
                
                if bbox is not None:
                    # Reconocimiento facial
                    face_crop = self.face_detector.crop_face(frame, bbox, size=Config.FACE_CROP_SIZE)
                    person_name, confidence = self.face_recognizer.identify_person(
                        face_crop, threshold=Config.FACE_RECOGNITION_THRESHOLD
                    )
                    
                    # Anti-spoofing
                    anti_spoof_crop = self.face_detector.crop_face(
                        frame, bbox, size=Config.ANTI_SPOOF_INPUT_SIZE
                    )
                    is_real, spoof_confidence = self.anti_spoof_detector.detect_spoofing(anti_spoof_crop)
                    
                    # Detección de vida
                    left_eye, right_eye = self.face_detector.get_eye_landmarks(
                        landmarks, frame.shape, Config.LEFT_EYE_INDICES, Config.RIGHT_EYE_INDICES
                    )
                    is_alive, liveness_results = self.liveness_detector.check_liveness(
                        left_eye, right_eye, bbox
                    )
                    
                    # Decisión final
                    is_known = person_name is not None
                    is_real_person = is_real and spoof_confidence > Config.ANTI_SPOOF_THRESHOLD
                    access_granted = is_known and is_real_person and is_alive
                    
                    if access_granted:
                        access_granted_count += 1
                    else:
                        access_denied_count += 1
                    
                    # Dibujar resultado
                    if access_granted:
                        status = f"ACCESO PERMITIDO - {person_name}"
                        color = (0, 255, 0)
                    else:
                        status = "ACCESO DENEGADO"
                        color = (0, 0, 255)
                    
                    frame = self.face_detector.draw_face_box(frame, bbox, status, color)
                    
                    # Información detallada
                    y_offset = bbox[3] + 20
                    cv2.putText(frame, f"Persona: {person_name or 'Desconocido'} ({confidence:.2f})", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Anti-spoof: {spoof_confidence:.2f}", 
                               (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Vivo: {is_alive}", 
                               (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Estadísticas generales
                total_attempts = access_granted_count + access_denied_count
                if total_attempts > 0:
                    success_rate = (access_granted_count / total_attempts) * 100
                    cv2.putText(frame, f"Accesos: {access_granted_count}/{total_attempts} ({success_rate:.1f}%)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow("Prueba del Sistema Completo", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.liveness_detector.reset()
            
            camera.close_camera()
            cv2.destroyAllWindows()
            
            print(f"\n📊 RESULTADOS FINALES:")
            print(f"  Accesos permitidos: {access_granted_count}")
            print(f"  Accesos denegados: {access_denied_count}")
            print(f"  Total intentos: {total_attempts}")
            if total_attempts > 0:
                print(f"  Tasa de éxito: {success_rate:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error probando sistema completo: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def test_database_operations(self):
        """Probar operaciones de base de datos"""
        try:
            print(f"\n=== PRUEBA DE OPERACIONES DE BASE DE DATOS ===")
            
            # Información básica
            info = self.database_manager.get_database_info()
            print(f"Personas en BD: {info['total_persons']}")
            print(f"Embeddings totales: {info['total_embeddings']}")
            
            # Estadísticas detalladas
            stats = self.database_manager.get_statistics()
            print(f"\n📊 ESTADÍSTICAS DETALLADAS:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Validar integridad
            is_valid, issues = self.database_manager.validate_database()
            if is_valid:
                print("\n✅ Base de datos válida")
            else:
                print(f"\n⚠️ Problemas en base de datos: {len(issues)}")
                for issue in issues:
                    print(f"  - {issue}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error probando base de datos: {e}")
            print(f"❌ Error: {e}")
            return False

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Probar sistema de reconocimiento facial")
    parser.add_argument("--test", "-t", type=str, choices=[
        'cameras', 'detection', 'recognition', 'antispoof', 'liveness', 'full', 'database', 'all'
    ], default='all', help="Tipo de prueba a ejecutar")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Índice de cámara")
    parser.add_argument("--image", "-i", type=str, help="Imagen de prueba para reconocimiento")
    parser.add_argument("--duration", "-d", type=int, default=10, help="Duración de pruebas en segundos")
    
    args = parser.parse_args()
    
    try:
        # Crear probador
        tester = SystemTester()
        
        tests_to_run = []
        if args.test == 'all':
            tests_to_run = ['cameras', 'detection', 'recognition', 'antispoof', 'liveness', 'database', 'full']
        else:
            tests_to_run = [args.test]
        
        results = {}
        
        for test_name in tests_to_run:
            print(f"\n{'='*60}")
            print(f"EJECUTANDO PRUEBA: {test_name.upper()}")
            print(f"{'='*60}")
            
            if test_name == 'cameras':
                results[test_name] = tester.test_camera_detection()
            elif test_name == 'detection':
                results[test_name] = tester.test_face_detection(args.camera, args.duration)
            elif test_name == 'recognition':
                results[test_name] = tester.test_face_recognition(args.image)
            elif test_name == 'antispoof':
                results[test_name] = tester.test_anti_spoofing(args.camera)
            elif test_name == 'liveness':
                results[test_name] = tester.test_liveness_detection(args.camera)
            elif test_name == 'database':
                results[test_name] = tester.test_database_operations()
            elif test_name == 'full':
                results[test_name] = tester.test_full_system(args.camera)
        
        # Resumen final
        print(f"\n{'='*60}")
        print("RESUMEN DE PRUEBAS")
        print(f"{'='*60}")
        
        for test_name, success in results.items():
            status = "✅ ÉXITO" if success else "❌ FALLO"
            print(f"{test_name.capitalize()}: {status}")
        
        total_tests = len(results)
        successful_tests = sum(results.values())
        print(f"\nResultado general: {successful_tests}/{total_tests} pruebas exitosas")
        
        if successful_tests == total_tests:
            print("🎉 ¡Todas las pruebas fueron exitosas!")
        else:
            print("⚠️ Algunas pruebas fallaron, revisar logs para más detalles")
    
    except Exception as e:
        logger.error(f"Error en función principal: {e}")
        print(f"❌ Error crítico: {e}")

if __name__ == "__main__":
    main()