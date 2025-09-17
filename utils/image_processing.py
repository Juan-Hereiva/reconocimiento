"""
Utilidades para procesamiento de imágenes
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Procesador de imágenes con múltiples funcionalidades"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        Cargar imagen desde archivo
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            np.ndarray: Imagen BGR o None si hay error
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Imagen no encontrada: {image_path}")
                return None
            
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"No se pudo cargar la imagen: {image_path}")
                return None
            
            return image
            
        except Exception as e:
            logger.error(f"Error cargando imagen {image_path}: {e}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, save_path: str) -> bool:
        """
        Guardar imagen a archivo
        
        Args:
            image: Imagen BGR
            save_path: Ruta donde guardar
            
        Returns:
            bool: True si se guardó exitosamente
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Guardar imagen
            success = cv2.imwrite(save_path, image)
            if success:
                logger.info(f"Imagen guardada: {save_path}")
                return True
            else:
                logger.error(f"Error guardando imagen: {save_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error guardando imagen {save_path}: {e}")
            return False
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                    maintain_aspect: bool = True) -> np.ndarray:
        """
        Redimensionar imagen
        
        Args:
            image: Imagen BGR
            target_size: (width, height) objetivo
            maintain_aspect: Mantener relación de aspecto
            
        Returns:
            np.ndarray: Imagen redimensionada
        """
        try:
            if maintain_aspect:
                h, w = image.shape[:2]
                target_w, target_h = target_size
                
                # Calcular escalado manteniendo aspecto
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Redimensionar
                resized = cv2.resize(image, (new_w, new_h))
                
                # Crear imagen con padding si es necesario
                if new_w != target_w or new_h != target_h:
                    # Crear imagen con fondo negro
                    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    
                    # Centrar imagen redimensionada
                    y_offset = (target_h - new_h) // 2
                    x_offset = (target_w - new_w) // 2
                    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                    
                    return result
                else:
                    return resized
            else:
                return cv2.resize(image, target_size)
                
        except Exception as e:
            logger.error(f"Error redimensionando imagen: {e}")
            return image
    
    @staticmethod
    def enhance_image(image: np.ndarray, brightness: float = 1.0, 
                     contrast: float = 1.0, sharpness: float = 1.0) -> np.ndarray:
        """
        Mejorar calidad de imagen
        
        Args:
            image: Imagen BGR
            brightness: Factor de brillo (1.0 = sin cambio)
            contrast: Factor de contraste (1.0 = sin cambio)
            sharpness: Factor de nitidez (1.0 = sin cambio)
            
        Returns:
            np.ndarray: Imagen mejorada
        """
        try:
            # Convertir a PIL para procesamiento
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Aplicar mejoras
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(brightness)
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(contrast)
            
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(sharpness)
            
            # Convertir de vuelta a OpenCV
            enhanced_rgb = np.array(pil_image)
            enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
            
            return enhanced_bgr
            
        except Exception as e:
            logger.error(f"Error mejorando imagen: {e}")
            return image
    
    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, 
                   tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Imagen BGR
            clip_limit: Límite de recorte
            tile_grid_size: Tamaño de la grilla de tiles
            
        Returns:
            np.ndarray: Imagen con CLAHE aplicado
        """
        try:
            # Convertir a LAB para procesar luminancia
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Aplicar CLAHE solo al canal L (luminancia)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l_channel_clahe = clahe.apply(l_channel)
            
            # Recombinar canales
            lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
            
            # Convertir de vuelta a BGR
            result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            
            return result
            
        except Exception as e:
            logger.error(f"Error aplicando CLAHE: {e}")
            return image
    
    @staticmethod
    def denoise_image(image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Reducir ruido en imagen
        
        Args:
            image: Imagen BGR
            method: Método de reducción de ruido ('bilateral', 'gaussian', 'median')
            
        Returns:
            np.ndarray: Imagen sin ruido
        """
        try:
            if method == 'bilateral':
                return cv2.bilateralFilter(image, 9, 75, 75)
            elif method == 'gaussian':
                return cv2.GaussianBlur(image, (5, 5), 0)
            elif method == 'median':
                return cv2.medianBlur(image, 5)
            else:
                logger.warning(f"Método de denoising desconocido: {method}")
                return image
                
        except Exception as e:
            logger.error(f"Error reduciendo ruido: {e}")
            return image
    
    @staticmethod
    def normalize_lighting(image: np.ndarray) -> np.ndarray:
        """
        Normalizar iluminación de la imagen
        
        Args:
            image: Imagen BGR
            
        Returns:
            np.ndarray: Imagen con iluminación normalizada
        """
        try:
            # Convertir a escala de grises para análisis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calcular estadísticas de iluminación
            mean_brightness = np.mean(gray)
            
            # Ajustar si la imagen está muy oscura o muy clara
            if mean_brightness < 100:  # Muy oscura
                gamma = 1.5
            elif mean_brightness > 180:  # Muy clara
                gamma = 0.7
            else:
                gamma = 1.0
            
            if gamma != 1.0:
                # Aplicar corrección gamma
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 
                                for i in np.arange(0, 256)]).astype("uint8")
                corrected = cv2.LUT(image, table)
                return corrected
            
            return image
            
        except Exception as e:
            logger.error(f"Error normalizando iluminación: {e}")
            return image


class DataAugmentor:
    """Aumentador de datos para entrenamiento"""
    
    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotar imagen
        
        Args:
            image: Imagen BGR
            angle: Ángulo de rotación en grados
            
        Returns:
            np.ndarray: Imagen rotada
        """
        try:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Crear matriz de rotación
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Aplicar rotación
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
            
            return rotated
            
        except Exception as e:
            logger.error(f"Error rotando imagen: {e}")
            return image
    
    @staticmethod
    def flip_image(image: np.ndarray, direction: int = 1) -> np.ndarray:
        """
        Voltear imagen
        
        Args:
            image: Imagen BGR
            direction: 1=horizontal, 0=vertical, -1=ambos
            
        Returns:
            np.ndarray: Imagen volteada
        """
        try:
            return cv2.flip(image, direction)
        except Exception as e:
            logger.error(f"Error volteando imagen: {e}")
            return image
    
    @staticmethod
    def add_noise(image: np.ndarray, noise_type: str = 'gaussian') -> np.ndarray:
        """
        Agregar ruido a imagen
        
        Args:
            image: Imagen BGR
            noise_type: Tipo de ruido ('gaussian', 'salt_pepper')
            
        Returns:
            np.ndarray: Imagen con ruido
        """
        try:
            if noise_type == 'gaussian':
                noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
                noisy = cv2.add(image, noise)
                return noisy
            elif noise_type == 'salt_pepper':
                noisy = image.copy()
                # Salt noise
                salt = np.random.random(image.shape[:2]) < 0.01
                noisy[salt] = 255
                # Pepper noise
                pepper = np.random.random(image.shape[:2]) < 0.01
                noisy[pepper] = 0
                return noisy
            else:
                return image
                
        except Exception as e:
            logger.error(f"Error agregando ruido: {e}")
            return image
    
    @staticmethod
    def change_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Cambiar brillo de imagen
        
        Args:
            image: Imagen BGR
            factor: Factor de brillo (0.5 = más oscuro, 1.5 = más claro)
            
        Returns:
            np.ndarray: Imagen con brillo modificado
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 2] *= factor
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            hsv = hsv.astype(np.uint8)
            
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        except Exception as e:
            logger.error(f"Error cambiando brillo: {e}")
            return image
    
    @staticmethod
    def augment_dataset(images: List[np.ndarray], 
                       augmentations_per_image: int = 3) -> List[np.ndarray]:
        """
        Aumentar dataset con múltiples transformaciones
        
        Args:
            images: Lista de imágenes originales
            augmentations_per_image: Número de aumentaciones por imagen
            
        Returns:
            List[np.ndarray]: Dataset aumentado
        """
        augmented_images = []
        
        for image in images:
            # Agregar imagen original
            augmented_images.append(image)
            
            # Generar aumentaciones
            for _ in range(augmentations_per_image):
                aug_image = image.copy()
                
                # Aplicar transformaciones aleatorias
                if np.random.random() > 0.5:
                    angle = np.random.uniform(-15, 15)
                    aug_image = DataAugmentor.rotate_image(aug_image, angle)
                
                if np.random.random() > 0.5:
                    aug_image = DataAugmentor.flip_image(aug_image, 1)
                
                if np.random.random() > 0.5:
                    factor = np.random.uniform(0.7, 1.3)
                    aug_image = DataAugmentor.change_brightness(aug_image, factor)
                
                if np.random.random() > 0.7:
                    aug_image = DataAugmentor.add_noise(aug_image, 'gaussian')
                
                augmented_images.append(aug_image)
        
        logger.info(f"Dataset aumentado: {len(images)} -> {len(augmented_images)} imágenes")
        return augmented_images


class ImageValidator:
    """Validador de calidad de imágenes"""
    
    @staticmethod
    def check_image_quality(image: np.ndarray) -> Tuple[bool, dict]:
        """
        Verificar calidad de imagen
        
        Args:
            image: Imagen BGR
            
        Returns:
            tuple: (is_good_quality, quality_metrics)
        """
        try:
            h, w = image.shape[:2]
            
            # Verificar resolución mínima
            min_resolution = h >= 100 and w >= 100
            
            # Verificar brillo
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            good_brightness = 50 <= brightness <= 200
            
            # Verificar contraste
            contrast = np.std(gray)
            good_contrast = contrast > 20
            
            # Verificar nitidez (usando Laplacian)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            good_sharpness = laplacian_var > 100
            
            # Verificar si no está saturada
            saturated_pixels = np.sum((image >= 250) | (image <= 5))
            total_pixels = h * w * 3
            saturation_ratio = saturated_pixels / total_pixels
            not_saturated = saturation_ratio < 0.1
            
            is_good = (min_resolution and good_brightness and 
                      good_contrast and good_sharpness and not_saturated)
            
            metrics = {
                "resolution": (w, h),
                "min_resolution": min_resolution,
                "brightness": float(brightness),
                "good_brightness": good_brightness,
                "contrast": float(contrast),
                "good_contrast": good_contrast,
                "sharpness": float(laplacian_var),
                "good_sharpness": good_sharpness,
                "saturation_ratio": float(saturation_ratio),
                "not_saturated": not_saturated,
                "overall_quality": is_good
            }
            
            return is_good, metrics
            
        except Exception as e:
            logger.error(f"Error verificando calidad: {e}")
            return False, {"error": str(e)}
    
    @staticmethod
    def filter_good_images(image_paths: List[str]) -> List[str]:
        """
        Filtrar imágenes de buena calidad
        
        Args:
            image_paths: Lista de rutas de imágenes
            
        Returns:
            List[str]: Rutas de imágenes de buena calidad
        """
        good_images = []
        
        for path in image_paths:
            image = ImageProcessor.load_image(path)
            if image is not None:
                is_good, _ = ImageValidator.check_image_quality(image)
                if is_good:
                    good_images.append(path)
                else:
                    logger.warning(f"Imagen de baja calidad descartada: {path}")
        
        logger.info(f"Filtradas {len(good_images)}/{len(image_paths)} imágenes de buena calidad")
        return good_images