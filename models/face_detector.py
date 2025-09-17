"""
Detector de rostros usando MediaPipe - Versión Corregida
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, confidence=0.7, max_num_faces=1):
        """
        Inicializar el detector de rostros
        """
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        logger.info(f"FaceDetector inicializado con confianza={confidence}")
    
    def detect_face_and_landmarks(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[object]]:
        """
        Detectar rostro y landmarks en el frame
        """
        try:
            # Obtener dimensiones del frame
            h, w, c = frame.shape
            
            # Convertir BGR a RGB para MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe
            results = self.face_mesh.process(rgb)
            
            if not results.multi_face_landmarks:
                return None, None
            
            # Tomar el primer rostro detectado
            landmarks = results.multi_face_landmarks[0]
            
            # Calcular bounding box desde landmarks
            bbox = self._calculate_bbox(landmarks, (h, w))
            
            return bbox, landmarks
            
        except Exception as e:
            logger.error(f"Error en detección de rostro: {e}")
            return None, None
    
    def _calculate_bbox(self, landmarks, frame_shape) -> Tuple[int, int, int, int]:
        """
        Calcular bounding box desde landmarks
        """
        h, w = frame_shape
        
        # Extraer coordenadas x, y de todos los landmarks
        xs = [landmark.x for landmark in landmarks.landmark]
        ys = [landmark.y for landmark in landmarks.landmark]
        
        # Convertir a coordenadas de pixel
        x_coords = [int(x * w) for x in xs]
        y_coords = [int(y * h) for y in ys]
        
        # Calcular bounding box con margen
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        # Agregar margen del 10%
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.1)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        return (x1, y1, x2, y2)
    
    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                  margin: float = 0.2, size: int = 160) -> np.ndarray:
        """
        Recortar rostro con margen adicional
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Calcular margen
            face_width = x2 - x1
            face_height = y2 - y1
            margin_w = int(face_width * margin)
            margin_h = int(face_height * margin)
            
            # Aplicar margen con límites
            xa = max(0, x1 - margin_w)
            xb = min(w, x2 + margin_w)
            ya = max(0, y1 - margin_h)
            yb = min(h, y2 + margin_h)
            
            # Recortar rostro
            face_crop = frame[ya:yb, xa:xb]
            
            # Verificar que el recorte es válido
            if face_crop.size == 0:
                logger.warning("Recorte de rostro vacío")
                return None
            
            # Redimensionar
            face_resized = cv2.resize(face_crop, (size, size))
            
            return face_resized
            
        except Exception as e:
            logger.error(f"Error al recortar rostro: {e}")
            return None
    
    def draw_face_box(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                     label: str = "", color: Tuple[int, int, int] = (0, 255, 0), 
                     thickness: int = 2) -> np.ndarray:
        """
        Dibujar bounding box y etiqueta en el frame
        """
        try:
            frame_copy = frame.copy()
            x1, y1, x2, y2 = bbox
            
            # Dibujar rectángulo
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Dibujar etiqueta si se proporciona
            if label:
                # Calcular tamaño del texto
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                text_thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
                
                # Dibujar fondo del texto
                cv2.rectangle(frame_copy, 
                            (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), 
                            color, -1)
                
                # Dibujar texto
                cv2.putText(frame_copy, label, (x1, y1 - 5), 
                          font, font_scale, (255, 255, 255), text_thickness)
            
            return frame_copy
            
        except Exception as e:
            logger.error(f"Error al dibujar bounding box: {e}")
            return frame
    
    def get_eye_landmarks(self, landmarks, frame_shape, 
                         left_eye_indices: List[int], 
                         right_eye_indices: List[int]) -> Tuple[List, List]:
        """
        Extraer coordenadas de landmarks de ojos - VERSIÓN MEJORADA
        """
        h, w = frame_shape[:2]
        
        def extract_eye_points(indices):
            points = []
            for idx in indices:
                if idx < len(landmarks.landmark):
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    points.append((x, y))
            return points
        
        # Usar índices simplificados de MediaPipe Face Mesh
        # Estos son más confiables
        left_eye_simple = [33, 7, 163, 144, 145, 153]  # 6 puntos clave
        right_eye_simple = [362, 382, 381, 380, 374, 373]  # 6 puntos clave
        
        try:
            left_eye = extract_eye_points(left_eye_simple)
            right_eye = extract_eye_points(right_eye_simple)
            
            # Validar que tenemos suficientes puntos
            if len(left_eye) >= 4 and len(right_eye) >= 4:
                return left_eye, right_eye
            else:
                logger.warning(f"Pocos landmarks de ojos: izq={len(left_eye)}, der={len(right_eye)}")
                return [], []
                
        except Exception as e:
            logger.error(f"Error extrayendo landmarks de ojos: {e}")
            return [], []
    
    def __del__(self):
        """Limpiar recursos"""
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
        except:
            pass