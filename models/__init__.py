"""
Módulo de modelos para el sistema de reconocimiento facial
"""

from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .anti_spoof import AntiSpoofNet, AntiSpoofDetector, AntiSpoofTrainer
from .liveness_detector import LivenessDetector, DepthBasedLiveness

__all__ = [
    'FaceDetector',
    'FaceRecognizer', 
    'AntiSpoofNet',
    'AntiSpoofDetector',
    'AntiSpoofTrainer',
    'LivenessDetector',
    'DepthBasedLiveness'
]