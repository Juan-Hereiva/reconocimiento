"""
Detector de vida (Liveness Detection) usando análisis de parpadeo y movimiento
"""
import cv2
import numpy as np
import time
from collections import deque
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class LivenessDetector:
    """Detector de vida usando múltiples métodos"""
    
    def __init__(self, ear_threshold: float = 0.25, 
                 consecutive_frames: int = 3,
                 time_window: float = 5.0,
                 movement_threshold: float = 20.0):
        """
        Inicializar detector de vida
        
        Args:
            ear_threshold: Umbral para Eye Aspect Ratio
            consecutive_frames: Frames consecutivos para confirmar parpadeo
            time_window: Ventana de tiempo para análisis (segundos)
            movement_threshold: Umbral para detección de movimiento
        """
     
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.time_window = time_window
        self.movement_threshold = movement_threshold
        
        # Estado del detector
        self.blink_counter = 0
        self.frame_counter = 0
        self.ear_history = deque(maxlen=int(30 * time_window))  # 30 FPS * ventana
        self.blink_times = deque(maxlen=10)  # Últimos 10 parpadeos
        self.face_positions = deque(maxlen=30)  # Posiciones recientes del rostro
        
        # Estados de parpadeo
        self.eye_closed_frames = 0
        self.is_blinking = False
           # Estado de frames inválidos (para tolerancia)
        self.bad_frames = 0
        self.min_bad_frames = 5  # puedes ajustar, p. ej. 5 frames (~0.16s a 30fps)


        logger.info("LivenessDetector inicializado")
    
    def calculate_ear(self, eye_landmarks: List[Tuple[int, int]]) -> float:
        """
        Calcular Eye Aspect Ratio (EAR)
        
        Args:
            eye_landmarks: Lista de puntos del ojo [(x, y), ...]
            
        Returns:
            float: Valor EAR
        """
        try:
            if len(eye_landmarks) < 6:
                return 0.0
            
            # Convertir a numpy array
            points = np.array(eye_landmarks)
            
            # Calcular distancias verticales
            vertical_1 = np.linalg.norm(points[1] - points[5])
            vertical_2 = np.linalg.norm(points[2] - points[4])
            
            # Calcular distancia horizontal
            horizontal = np.linalg.norm(points[0] - points[3])
            
            # Calcular EAR
            if horizontal == 0:
                return 0.0
                
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
            
        except Exception as e:
            logger.error(f"Error calculando EAR: {e}")
            return 0.0
    
    def simplified_ear(self, eye_landmarks: List[Tuple[int, int]]) -> float:
        """
        Versión simplificada de EAR para landmarks limitados
        
        Args:
            eye_landmarks: Lista de puntos del ojo
            
        Returns:
            float: Valor EAR simplificado
        """
        try:
            if len(eye_landmarks) < 4:
                return 0.0
            
            # Tomar puntos aproximados (esquinas y puntos verticales)
            left_corner = eye_landmarks[0]
            right_corner = eye_landmarks[-1]
            top_point = max(eye_landmarks, key=lambda p: p[1])
            bottom_point = min(eye_landmarks, key=lambda p: p[1])
            
            # Calcular distancias
            horizontal = np.linalg.norm(np.array(right_corner) - np.array(left_corner))
            vertical = np.linalg.norm(np.array(top_point) - np.array(bottom_point))
            
            if horizontal == 0:
                return 0.0
                
            ear = vertical / horizontal
            return ear
            
        except Exception as e:
            logger.error(f"Error en EAR simplificado: {e}")
            return 0.0
    
    def detect_blink(self, left_eye: List[Tuple[int, int]], 
                    right_eye: List[Tuple[int, int]]) -> bool:
        """
        Detectar parpadeo usando EAR
        
        Args:
            left_eye: Landmarks del ojo izquierdo
            right_eye: Landmarks del ojo derecho
            
        Returns:
            bool: True si se detectó un parpadeo
        """
        try:
            # Calcular EAR para ambos ojos
            left_ear = self.simplified_ear(left_eye) if left_eye else 0.0
            right_ear = self.simplified_ear(right_eye) if right_eye else 0.0
            
            # Promedio de ambos ojos
            avg_ear = (left_ear + right_ear) / 2.0 if (left_ear > 0 and right_ear > 0) else 0.0
            
            # Agregar a historial
            self.ear_history.append(avg_ear)
            
            # Verificar si el ojo está cerrado
            if avg_ear < self.ear_threshold:
                self.eye_closed_frames += 1
            else:
                # Si el ojo se abrió después de estar cerrado
                if self.eye_closed_frames >= self.consecutive_frames:
                    # Parpadeo detectado
                    self.blink_counter += 1
                    self.blink_times.append(time.time())
                    logger.debug(f"Parpadeo detectado! Total: {self.blink_counter}")
                    self.eye_closed_frames = 0
                    return True
                
                self.eye_closed_frames = 0
            
            return False
            
        except Exception as e:
            logger.error(f"Error detectando parpadeo: {e}")
            return False
    
    def detect_head_movement(self, face_bbox: Tuple[int, int, int, int]) -> float:
        """
        Detectar movimiento de cabeza
        
        Args:
            face_bbox: Bounding box del rostro (x1, y1, x2, y2)
            
        Returns:
            float: Cantidad de movimiento detectado
        """
        try:
            # Calcular centro del rostro
            x1, y1, x2, y2 = face_bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_position = (center_x, center_y)
            
            # Agregar posición actual
            self.face_positions.append(current_position)
            
            if len(self.face_positions) < 2:
                return 0.0
            
            # Calcular movimiento total en la ventana
            total_movement = 0.0
            for i in range(1, len(self.face_positions)):
                prev_pos = self.face_positions[i-1]
                curr_pos = self.face_positions[i]
                
                distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                 (curr_pos[1] - prev_pos[1])**2)
                total_movement += distance
            
            return total_movement
            
        except Exception as e:
            logger.error(f"Error detectando movimiento: {e}")
            return 0.0
    
    def analyze_blink_pattern(self) -> Tuple[bool, dict]:
        """
        Analizar patrón de parpadeo para detectar naturalidad
        
        Returns:
            tuple: (is_natural, statistics)
        """
        try:
            current_time = time.time()
            
            # Filtrar parpadeos recientes
            recent_blinks = [t for t in self.blink_times 
                           if current_time - t <= self.time_window]
            
            if len(recent_blinks) < 2:
                return False, {"blinks": len(recent_blinks), "reason": "insuficientes_parpadeos"}
            
            # Calcular estadísticas de parpadeo
            blink_intervals = []
            for i in range(1, len(recent_blinks)):
                interval = recent_blinks[i] - recent_blinks[i-1]
                blink_intervals.append(interval)
            
            avg_interval = np.mean(blink_intervals)
            std_interval = np.std(blink_intervals)
            
            # Criterios de naturalidad
            # Frecuencia normal: 15-20 parpadeos por minuto (3-4 segundos entre parpadeos)
            natural_frequency = 1.5 <= avg_interval <= 6.0
            
            # Variabilidad natural en intervalos
            natural_variability = std_interval > 0.2 and std_interval < avg_interval
            
            # Número razonable de parpadeos
            reasonable_count = 2 <= len(recent_blinks) <= 10
            
            is_natural = natural_frequency and natural_variability and reasonable_count
            
            stats = {
                "blinks": len(recent_blinks),
                "avg_interval": avg_interval,
                "std_interval": std_interval,
                "natural_frequency": natural_frequency,
                "natural_variability": natural_variability,
                "reasonable_count": reasonable_count
            }
            
            return is_natural, stats
            
        except Exception as e:
            logger.error(f"Error analizando patrón de parpadeo: {e}")
            return False, {"error": str(e)}
    
    def check_liveness(self, left_eye: List[Tuple[int, int]], 
                      right_eye: List[Tuple[int, int]], 
                      face_bbox: Tuple[int, int, int, int]) -> Tuple[bool, dict]:
        """
        Verificación completa de vida
        
        Args:
            left_eye: Landmarks del ojo izquierdo
            right_eye: Landmarks del ojo derecho
            face_bbox: Bounding box del rostro
            
        Returns:
            tuple: (is_alive, detailed_results)
        """
        try:
            # Detectar parpadeo
            blink_detected = self.detect_blink(left_eye, right_eye)
            
            # Detectar movimiento
            movement = self.detect_head_movement(face_bbox)
            has_movement = movement > self.movement_threshold
            
            # Analizar patrón de parpadeo
            natural_pattern, blink_stats = self.analyze_blink_pattern()
            
                        # Criterios de vida
            enough_blinks = self.blink_counter >= 2
            recent_activity = len([t for t in self.blink_times 
                                if time.time() - t <= self.time_window]) > 0

            # Decisión final con tolerancia
            if (enough_blinks and natural_pattern) or (recent_activity and has_movement):
                self.bad_frames = 0
                is_alive = True
            else:
                self.bad_frames += 1
                is_alive = self.bad_frames < self.min_bad_frames

            results = {
                "is_alive": is_alive,
                "blink_detected": blink_detected,
                "total_blinks": self.blink_counter,
                "has_movement": has_movement,
                "movement_amount": movement,
                "natural_pattern": natural_pattern,
                "blink_statistics": blink_stats,
                "recent_activity": recent_activity,
                "bad_frames": self.bad_frames
            }
            
            return is_alive, results
            
        except Exception as e:
            logger.error(f"Error en verificación de vida: {e}")
            return False, {"error": str(e)}
    
    def reset(self):
        """Reiniciar estado del detector"""
        self.blink_counter = 0
        self.frame_counter = 0
        self.ear_history.clear()
        self.blink_times.clear()
        self.face_positions.clear()
        self.eye_closed_frames = 0
        self.is_blinking = False
        self.bad_frames = 0  # <- IMPORTANTE
        logger.info("LivenessDetector reiniciado")
    
    def get_status(self) -> dict:
        """
        Obtener estado actual del detector
        
        Returns:
            dict: Estado actual
        """
        current_time = time.time()
        recent_blinks = len([t for t in self.blink_times 
                           if current_time - t <= self.time_window])
        
        return {
            "total_blinks": self.blink_counter,
            "recent_blinks": recent_blinks,
            "eye_closed_frames": self.eye_closed_frames,
            "ear_history_length": len(self.ear_history),
            "face_positions_tracked": len(self.face_positions),
            "time_window": self.time_window,
            "ear_threshold": self.ear_threshold
        }


class DepthBasedLiveness:
    """Detector de vida basado en información de profundidad (si está disponible)"""
    
    def __init__(self, depth_threshold: float = 100.0):
        """
        Inicializar detector basado en profundidad
        
        Args:
            depth_threshold: Umbral de variación de profundidad
        """
        self.depth_threshold = depth_threshold
        logger.info("DepthBasedLiveness inicializado")
    
    def analyze_depth(self, depth_image: np.ndarray, 
                     face_bbox: Tuple[int, int, int, int]) -> Tuple[bool, dict]:
        """
        Analizar imagen de profundidad para detectar objeto 3D real
        
        Args:
            depth_image: Imagen de profundidad
            face_bbox: Bounding box del rostro
            
        Returns:
            tuple: (is_real_3d, analysis_results)
        """
        try:
            x1, y1, x2, y2 = face_bbox
            
            # Extraer región del rostro en imagen de profundidad
            face_depth = depth_image[y1:y2, x1:x2]
            
            if face_depth.size == 0:
                return False, {"error": "región_vacía"}
            
            # Calcular estadísticas de profundidad
            depth_mean = np.mean(face_depth)
            depth_std = np.std(face_depth)
            depth_range = np.max(face_depth) - np.min(face_depth)
            
            # Un rostro real debe tener variación de profundidad significativa
            has_depth_variation = depth_range > self.depth_threshold
            has_depth_structure = depth_std > 10.0  # Variabilidad mínima
            
            # Verificar que no sea una superficie plana
            is_real_3d = has_depth_variation and has_depth_structure
            
            results = {
                "is_real_3d": is_real_3d,
                "depth_mean": float(depth_mean),
                "depth_std": float(depth_std),
                "depth_range": float(depth_range),
                "has_depth_variation": has_depth_variation,
                "has_depth_structure": has_depth_structure
            }
            
            return is_real_3d, results
            
        except Exception as e:
            logger.error(f"Error en análisis de profundidad: {e}")
            return False, {"error": str(e)}