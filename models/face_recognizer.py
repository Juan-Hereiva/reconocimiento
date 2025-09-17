"""
Reconocedor facial usando FaceNet
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, device='auto', model_name='vggface2'):
        """
        Inicializar el reconocedor facial
        
        Args:
            device: 'cuda', 'cpu' o 'auto'
            model_name: Modelo pre-entrenado ('vggface2' o 'casia-webface')
        """
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Usando device: {self.device}")
        
        # Cargar modelo FaceNet pre-entrenado
        try:
            self.model = InceptionResnetV1(pretrained=model_name).eval().to(self.device)
            logger.info(f"Modelo FaceNet cargado: {model_name}")
        except Exception as e:
            logger.error(f"Error cargando modelo FaceNet: {e}")
            raise
        
        # Base de datos de embeddings
        self.face_database = {}
        
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Obtener embedding de una imagen de rostro
        
        Args:
            face_image: Imagen BGR del rostro (160x160)
            
        Returns:
            np.ndarray: Embedding de 512 dimensiones
        """
        try:
            # Convertir BGR a RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Normalizar a [0, 1]
            face_normalized = face_rgb.astype(np.float32) / 255.0
            
            # Convertir a tensor PyTorch
            face_tensor = torch.tensor(face_normalized).permute(2, 0, 1).unsqueeze(0)
            face_tensor = face_tensor.to(self.device)
            
            # Obtener embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
                
            # Convertir a numpy y normalizar
            embedding = embedding.cpu().numpy().flatten()
            embedding = F.normalize(torch.tensor(embedding), p=2, dim=0).numpy()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error obteniendo embedding: {e}")
            return None
    
    def add_person_to_database(self, person_name: str, face_images: list, 
                              method: str = 'average') -> bool:
        """
        Agregar persona a la base de datos
        
        Args:
            person_name: Nombre de la persona
            face_images: Lista de imágenes de rostro
            method: 'average' o 'multiple' para múltiples embeddings
            
        Returns:
            bool: True si se agregó exitosamente
        """
        try:
            embeddings = []
            
            for face_image in face_images:
                embedding = self.get_embedding(face_image)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if not embeddings:
                logger.warning(f"No se pudieron generar embeddings para {person_name}")
                return False
            
            embeddings = np.array(embeddings)
            
            if method == 'average':
                # Promedio de todos los embeddings
                final_embedding = np.mean(embeddings, axis=0)
                self.face_database[person_name] = final_embedding
            else:
                # Guardar múltiples embeddings
                self.face_database[person_name] = embeddings
            
            logger.info(f"Persona '{person_name}' agregada con {len(embeddings)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error agregando persona {person_name}: {e}")
            return False
    
    def identify_person(self, face_image: np.ndarray, 
                       threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """
        Identificar persona en la imagen
        
        Args:
            face_image: Imagen BGR del rostro
            threshold: Umbral de similitud mínima
            
        Returns:
            tuple: (nombre_persona, similitud) o (None, mejor_similitud)
        """
        try:
            if not self.face_database:
                logger.warning("Base de datos vacía")
                return None, 0.0
            
            # Obtener embedding de la imagen
            query_embedding = self.get_embedding(face_image)
            if query_embedding is None:
                return None, 0.0
            
            best_match = None
            best_similarity = 0.0
            
            # Comparar con cada persona en la base de datos
            for person_name, stored_embedding in self.face_database.items():
                if stored_embedding.ndim == 1:
                    # Embedding único
                    similarity = cosine_similarity([query_embedding], [stored_embedding])[0, 0]
                else:
                    # Múltiples embeddings - tomar el mejor
                    similarities = cosine_similarity([query_embedding], stored_embedding)[0]
                    similarity = np.max(similarities)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    if similarity >= threshold:
                        best_match = person_name
            
            return best_match, float(best_similarity)
            
        except Exception as e:
            logger.error(f"Error en identificación: {e}")
            return None, 0.0
    
    def remove_person(self, person_name: str) -> bool:
        """
        Eliminar persona de la base de datos
        
        Args:
            person_name: Nombre de la persona a eliminar
            
        Returns:
            bool: True si se eliminó exitosamente
        """
        if person_name in self.face_database:
            del self.face_database[person_name]
            logger.info(f"Persona '{person_name}' eliminada de la base de datos")
            return True
        else:
            logger.warning(f"Persona '{person_name}' no encontrada en la base de datos")
            return False
    
    def get_database_info(self) -> Dict:
        """
        Obtener información de la base de datos
        
        Returns:
            dict: Información de personas registradas
        """
        info = {
            'total_persons': len(self.face_database),
            'persons': list(self.face_database.keys())
        }
        
        for person, embedding in self.face_database.items():
            if embedding.ndim == 1:
                info[f'{person}_embeddings'] = 1
            else:
                info[f'{person}_embeddings'] = len(embedding)
        
        return info
    
    def save_database(self, filepath: str) -> bool:
        """
        Guardar base de datos a archivo
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            bool: True si se guardó exitosamente
        """
        try:
            np.save(filepath, self.face_database)
            logger.info(f"Base de datos guardada en {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error guardando base de datos: {e}")
            return False
    
    def load_database(self, filepath: str) -> bool:
        """
        Cargar base de datos desde archivo
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            bool: True si se cargó exitosamente
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Archivo de base de datos no existe: {filepath}")
                return False
                
            self.face_database = np.load(filepath, allow_pickle=True).item()
            logger.info(f"Base de datos cargada desde {filepath}")
            logger.info(f"Personas cargadas: {list(self.face_database.keys())}")
            return True
        except Exception as e:
            logger.error(f"Error cargando base de datos: {e}")
            return False
    
    def clear_database(self):
        """Limpiar base de datos"""
        self.face_database = {}
        logger.info("Base de datos limpiada")
    
    def update_threshold(self, new_threshold: float):
        """
        Actualizar umbral de reconocimiento
        
        Args:
            new_threshold: Nuevo umbral (0.0 - 1.0)
        """
        if 0.0 <= new_threshold <= 1.0:
            logger.info(f"Umbral actualizado a {new_threshold}")
        else:
            logger.warning(f"Umbral inválido: {new_threshold}. Debe estar entre 0.0 y 1.0")

import os  # Agregar import que faltaba