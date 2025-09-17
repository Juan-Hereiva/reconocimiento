"""
Utilidades para manejo de cámara
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging
import time

logger = logging.getLogger(__name__)

class CameraManager:
    """Gestor de cámara con funcionalidades avanzadas"""
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Inicializar gestor de cámara
        
        Args:
            camera_index: Índice de la cámara
            width: Ancho de captura
            height: Alto de captura
            fps: Frames por segundo
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_opened = False
        
        # Configuración de captura
        self.auto_exposure = True
        self.brightness = 0.5
        self.contrast = 0.5
        
        logger.info(f"CameraManager inicializado: índice={camera_index}, resolución={width}x{height}")
    
    def open_camera(self) -> bool:
        """
        Abrir cámara
        
        Returns:
            bool: True si se abrió exitosamente
        """
        try:
            if self.is_opened:
                logger.warning("La cámara ya está abierta")
                return True
            
            # Abrir captura de video
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"No se pudo abrir la cámara {self.camera_index}")
                return False
            
            # Configurar propiedades
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verificar configuración
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Cámara configurada: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.is_opened = True
            return True
            
        except Exception as e:
            logger.error(f"Error abriendo cámara: {e}")
            return False
    
    def close_camera(self):
        """Cerrar cámara"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            self.is_opened = False
            logger.info("Cámara cerrada")
            
        except Exception as e:
            logger.error(f"Error cerrando cámara: {e}")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Leer frame de la cámara
        
        Returns:
            np.ndarray: Frame BGR o None si hay error
        """
        try:
            if not self.is_opened or self.cap is None:
                return None
            
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logger.warning("No se pudo leer frame de la cámara")
                return None
            
            return frame
            
        except Exception as e:
            logger.error(f"Error leyendo frame: {e}")
            return None
    
    def set_camera_properties(self, brightness: Optional[float] = None, 
                            contrast: Optional[float] = None,
                            exposure: Optional[float] = None) -> bool:
        """
        Configurar propiedades de la cámara
        
        Args:
            brightness: Brillo (0.0 - 1.0)
            contrast: Contraste (0.0 - 1.0)
            exposure: Exposición (-8.0 - 1.0, None para automático)
            
        Returns:
            bool: True si se configuró exitosamente
        """
        try:
            if not self.is_opened or self.cap is None:
                logger.warning("Cámara no está abierta")
                return False
            
            success = True
            
            if brightness is not None:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
                self.brightness = brightness
                logger.debug(f"Brillo configurado: {brightness}")
            
            if contrast is not None:
                self.cap.set(cv2.CAP_PROP_CONTRAST, contrast)
                self.contrast = contrast
                logger.debug(f"Contraste configurado: {contrast}")
            
            if exposure is not None:
                # Deshabilitar auto exposición
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                self.auto_exposure = False
                logger.debug(f"Exposición manual configurada: {exposure}")
            else:
                # Habilitar auto exposición
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto mode
                self.auto_exposure = True
                logger.debug("Auto exposición habilitada")
            
            return success
            
        except Exception as e:
            logger.error(f"Error configurando propiedades de cámara: {e}")
            return False
    
    def get_camera_info(self) -> dict:
        """
        Obtener información de la cámara
        
        Returns:
            dict: Información de la cámara
        """
        if not self.is_opened or self.cap is None:
            return {"error": "Cámara no está abierta"}
        
        try:
            info = {
                "index": self.camera_index,
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
                "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
                "auto_exposure": bool(self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error obteniendo información de cámara: {e}")
            return {"error": str(e)}
    
    def test_camera(self, duration: int = 5) -> bool:
        """
        Probar funcionamiento de la cámara
        
        Args:
            duration: Duración de la prueba en segundos
            
        Returns:
            bool: True si la prueba fue exitosa
        """
        try:
            if not self.open_camera():
                return False
            
            start_time = time.time()
            frame_count = 0
            
            logger.info(f"Iniciando prueba de cámara por {duration} segundos...")
            
            while time.time() - start_time < duration:
                frame = self.read_frame()
                if frame is not None:
                    frame_count += 1
                    # Mostrar frame con información
                    fps_current = frame_count / (time.time() - start_time)
                    cv2.putText(frame, f"FPS: {fps_current:.1f}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Presiona 'q' para salir", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Prueba de Cámara", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    logger.warning("Frame vacío durante prueba")
            
            cv2.destroyAllWindows()
            
            avg_fps = frame_count / duration
            logger.info(f"Prueba completada: {frame_count} frames, FPS promedio: {avg_fps:.1f}")
            
            return frame_count > 0
            
        except Exception as e:
            logger.error(f"Error durante prueba de cámara: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry"""
        self.open_camera()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_camera()


class MultiCameraManager:
    """Gestor para múltiples cámaras"""
    
    def __init__(self):
        """Inicializar gestor de múltiples cámaras"""
        self.cameras = {}
        self.active_camera = None
        logger.info("MultiCameraManager inicializado")
    
    def add_camera(self, name: str, camera_index: int, 
                   width: int = 640, height: int = 480, fps: int = 30) -> bool:
        """
        Agregar cámara
        
        Args:
            name: Nombre identificador de la cámara
            camera_index: Índice de la cámara
            width: Ancho de captura
            height: Alto de captura
            fps: Frames por segundo
            
        Returns:
            bool: True si se agregó exitosamente
        """
        try:
            camera = CameraManager(camera_index, width, height, fps)
            self.cameras[name] = camera
            logger.info(f"Cámara agregada: {name} (índice {camera_index})")
            return True
            
        except Exception as e:
            logger.error(f"Error agregando cámara {name}: {e}")
            return False
    
    def remove_camera(self, name: str) -> bool:
        """
        Eliminar cámara
        
        Args:
            name: Nombre de la cámara
            
        Returns:
            bool: True si se eliminó exitosamente
        """
        try:
            if name in self.cameras:
                self.cameras[name].close_camera()
                del self.cameras[name]
                if self.active_camera == name:
                    self.active_camera = None
                logger.info(f"Cámara eliminada: {name}")
                return True
            else:
                logger.warning(f"Cámara no encontrada: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error eliminando cámara {name}: {e}")
            return False
    
    def switch_camera(self, name: str) -> bool:
        """
        Cambiar a otra cámara
        
        Args:
            name: Nombre de la cámara
            
        Returns:
            bool: True si se cambió exitosamente
        """
        try:
            if name not in self.cameras:
                logger.error(f"Cámara no encontrada: {name}")
                return False
            
            # Cerrar cámara activa actual
            if self.active_camera and self.active_camera in self.cameras:
                self.cameras[self.active_camera].close_camera()
            
            # Abrir nueva cámara
            if self.cameras[name].open_camera():
                self.active_camera = name
                logger.info(f"Cambiado a cámara: {name}")
                return True
            else:
                logger.error(f"No se pudo abrir cámara: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error cambiando cámara a {name}: {e}")
            return False
    
    def read_frame(self, camera_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Leer frame de cámara específica o activa
        
        Args:
            camera_name: Nombre de la cámara (opcional, usa activa por defecto)
            
        Returns:
            np.ndarray: Frame BGR o None si hay error
        """
        try:
            if camera_name is None:
                camera_name = self.active_camera
            
            if camera_name is None or camera_name not in self.cameras:
                return None
            
            return self.cameras[camera_name].read_frame()
            
        except Exception as e:
            logger.error(f"Error leyendo frame de {camera_name}: {e}")
            return None
    
    def list_cameras(self) -> List[str]:
        """
        Listar cámaras disponibles
        
        Returns:
            List[str]: Lista de nombres de cámaras
        """
        return list(self.cameras.keys())
    
    def close_all_cameras(self):
        """Cerrar todas las cámaras"""
        try:
            for camera in self.cameras.values():
                camera.close_camera()
            self.active_camera = None
            logger.info("Todas las cámaras cerradas")
            
        except Exception as e:
            logger.error(f"Error cerrando cámaras: {e}")


def detect_available_cameras(max_cameras: int = 10) -> List[int]:
    """
    Detectar cámaras disponibles en el sistema
    
    Args:
        max_cameras: Número máximo de cámaras a verificar
        
    Returns:
        List[int]: Lista de índices de cámaras disponibles
    """
    available_cameras = []
    
    logger.info(f"Detectando cámaras disponibles (máximo {max_cameras})...")
    
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                    logger.info(f"Cámara encontrada en índice {i}")
                cap.release()
            
        except Exception as e:
            logger.debug(f"Error verificando cámara {i}: {e}")
    
    logger.info(f"Cámaras disponibles: {available_cameras}")
    return available_cameras


def get_camera_resolutions(camera_index: int) -> List[Tuple[int, int]]:
    """
    Obtener resoluciones soportadas por una cámara
    
    Args:
        camera_index: Índice de la cámara
        
    Returns:
        List[Tuple[int, int]]: Lista de resoluciones (width, height)
    """
    common_resolutions = [
        (320, 240),   # QVGA
        (640, 480),   # VGA
        (800, 600),   # SVGA
        (1024, 768),  # XGA
        (1280, 720),  # HD
        (1280, 960),  # SXGA-
        (1920, 1080), # Full HD
        (2560, 1440), # QHD
        (3840, 2160), # 4K
    ]
    
    supported_resolutions = []
    
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return supported_resolutions
        
        for width, height in common_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                ret, frame = cap.read()
                if ret and frame is not None:
                    supported_resolutions.append((width, height))
        
        cap.release()
        
    except Exception as e:
        logger.error(f"Error obteniendo resoluciones para cámara {camera_index}: {e}")
    
    logger.info(f"Resoluciones soportadas por cámara {camera_index}: {supported_resolutions}")
    return supported_resolutions