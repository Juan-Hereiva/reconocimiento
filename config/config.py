"""
Configuración del sistema de reconocimiento facial
"""
import os
from pathlib import Path

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent.parent

class Config:
    # Rutas de archivos y directorios
    DATA_DIR = BASE_DIR / "data"
    ENROLLED_FACES_DIR = DATA_DIR / "enrolled_faces"
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"
    MODELS_DIR = DATA_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Archivos específicos
    FACE_DATABASE_PATH = EMBEDDINGS_DIR / "face_database.npy"
    ANTI_SPOOF_MODEL_PATH = MODELS_DIR / "anti_spoof_model.pth"
    LOG_FILE = LOGS_DIR / "system.log"
    
    # Configuración de reconocimiento facial
    FACE_RECOGNITION_THRESHOLD = 0.45
    FACE_CROP_SIZE = 160  # Tamaño para embeddings FaceNet
    ANTI_SPOOF_INPUT_SIZE = 128  # Tamaño para modelo anti-spoof
    
    # Configuración de detección de vida
    EAR_THRESHOLD = 0.25  # Umbral para detección de parpadeo
    BLINK_CONSECUTIVE_FRAMES = 3  # Frames consecutivos para confirmar parpadeo
    LIVENESS_TIME_WINDOW = 5.0  # Segundos para verificar vida
    
    # Configuración anti-spoofing - CORREGIDA
    ANTI_SPOOF_THRESHOLD = 0.4  # CAMBIADO: de 0.8 a 0.4 (menos estricto)
    ANTI_SPOOF_DEBUG = True  # NUEVO: habilitar debug por defecto
    ANTI_SPOOF_SMOOTHING = True  # NUEVO: suavizado temporal
    ANTI_SPOOF_SMOOTH_FRAMES = 3  # NUEVO: frames para promediar
    ANTI_SPOOF_FAIL_TOLERANCE = 4  # NUEVO: frames consecutivos con baja confianza antes de bloquear
    
    # Configuración de cámara
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Configuración de MediaPipe
    MP_FACE_CONFIDENCE = 0.7
    MP_MAX_NUM_FACES = 1
    
    # Configuración de visualización
    BBOX_COLOR_ALLOWED = (0, 255, 0)  # Verde para acceso permitido
    BBOX_COLOR_DENIED = (0, 0, 255)   # Rojo para acceso denegado
    BBOX_THICKNESS = 2
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    
    # Índices de landmarks para ojos (MediaPipe Face Mesh)
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # Configuración de logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def create_directories(cls):
        """Crear directorios necesarios si no existen"""
        directories = [
            cls.DATA_DIR,
            cls.ENROLLED_FACES_DIR,
            cls.EMBEDDINGS_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Configuración específica para entrenamiento - MEJORADA
class TrainingConfig:
    # Configuración de entrenamiento anti-spoofing
    BATCH_SIZE = 16  # REDUCIDO: de 32 a 16 para mejor convergencia
    LEARNING_RATE = 0.0005  # REDUCIDO: de 0.001 para evitar overfitting
    NUM_EPOCHS = 30  # REDUCIDO: de 50 para evitar overfitting
    WEIGHT_DECAY = 1e-4
    
    # Data augmentation - AJUSTADA
    AUGMENTATION_ROTATION = 5  # REDUCIDO: menos rotación extrema
    AUGMENTATION_BRIGHTNESS = 0.05  # REDUCIDO: cambios más sutiles
    AUGMENTATION_CONTRAST = 0.05  # REDUCIDO: cambios más sutiles
    
    # Validación
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 5  # REDUCIDO: parar antes si no mejora
    
    # NUEVO: Balanceado de clases
    CLASS_WEIGHTS = [1.0, 0.5]  # Peso menor para clase "fake" debido al desbalance
    USE_FOCAL_LOSS = True  # Usar focal loss para clases desbalanceadas
