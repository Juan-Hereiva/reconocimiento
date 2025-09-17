"""
Gestor de base de datos para rostros registrados
"""
import os
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gestor de base de datos de rostros"""
    
    def __init__(self, db_path: str):
        """
        Inicializar gestor de base de datos
        
        Args:
            db_path: Ruta del archivo de base de datos
        """
        self.db_path = Path(db_path)
        self.metadata_path = self.db_path.with_suffix('.json')
        
        # Estructura de datos
        self.face_embeddings = {}  # nombre -> embedding(s)
        self.metadata = {}  # información adicional
        
        # Crear directorio si no existe
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Cargar datos existentes
        self.load_database()
        
        logger.info(f"DatabaseManager inicializado: {self.db_path}")
    
    def add_person(self, person_name: str, embeddings: np.ndarray, 
                   metadata: Optional[Dict] = None) -> bool:
        """
        Agregar persona a la base de datos
        
        Args:
            person_name: Nombre de la persona
            embeddings: Embedding(s) facial(es)
            metadata: Información adicional
            
        Returns:
            bool: True si se agregó exitosamente
        """
        try:
            # Validar entrada
            if not person_name or not isinstance(embeddings, np.ndarray):
                logger.error("Nombre o embeddings inválidos")
                return False
            
            # Normalizar nombre
            person_name = person_name.strip().lower()
            
            # Agregar embeddings
            self.face_embeddings[person_name] = embeddings
            
            # Agregar metadata
            current_metadata = {
                'name': person_name,
                'added_date': datetime.now().isoformat(),
                'embedding_shape': embeddings.shape,
                'num_embeddings': len(embeddings) if embeddings.ndim > 1 else 1
            }
            
            if metadata:
                current_metadata.update(metadata)
            
            self.metadata[person_name] = current_metadata
            
            logger.info(f"Persona agregada: {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error agregando persona {person_name}: {e}")
            return False
    
    def remove_person(self, person_name: str) -> bool:
        """
        Eliminar persona de la base de datos
        
        Args:
            person_name: Nombre de la persona
            
        Returns:
            bool: True si se eliminó exitosamente
        """
        try:
            person_name = person_name.strip().lower()
            
            if person_name in self.face_embeddings:
                del self.face_embeddings[person_name]
                del self.metadata[person_name]
                logger.info(f"Persona eliminada: {person_name}")
                return True
            else:
                logger.warning(f"Persona no encontrada: {person_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error eliminando persona {person_name}: {e}")
            return False
    
    def get_person(self, person_name: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Obtener datos de una persona
        
        Args:
            person_name: Nombre de la persona
            
        Returns:
            tuple: (embeddings, metadata) o None si no existe
        """
        try:
            person_name = person_name.strip().lower()
            
            if person_name in self.face_embeddings:
                embeddings = self.face_embeddings[person_name]
                metadata = self.metadata.get(person_name, {})
                return embeddings, metadata
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error obteniendo persona {person_name}: {e}")
            return None
    
    def list_persons(self) -> List[str]:
        """
        Listar todas las personas en la base de datos
        
        Returns:
            List[str]: Lista de nombres
        """
        return list(self.face_embeddings.keys())
    
    def get_database_info(self) -> Dict:
        """
        Obtener información de la base de datos
        
        Returns:
            Dict: Información estadística
        """
        total_persons = len(self.face_embeddings)
        total_embeddings = 0
        
        for embeddings in self.face_embeddings.values():
            if embeddings.ndim > 1:
                total_embeddings += len(embeddings)
            else:
                total_embeddings += 1
        
        return {
            'total_persons': total_persons,
            'total_embeddings': total_embeddings,
            'database_path': str(self.db_path),
            'last_modified': self._get_last_modified()
        }
    
    def _get_last_modified(self) -> str:
        """Obtener fecha de última modificación"""
        try:
            if self.db_path.exists():
                timestamp = self.db_path.stat().st_mtime
                return datetime.fromtimestamp(timestamp).isoformat()
            else:
                return "nunca"
        except:
            return "desconocido"
    
    def save_database(self) -> bool:
        """
        Guardar base de datos a archivo
        
        Returns:
            bool: True si se guardó exitosamente
        """
        try:
            # Guardar embeddings (formato numpy)
            np.save(self.db_path, self.face_embeddings)
            
            # Guardar metadata (formato JSON)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Base de datos guardada: {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando base de datos: {e}")
            return False
    
    def load_database(self) -> bool:
        """
        Cargar base de datos desde archivo
        
        Returns:
            bool: True si se cargó exitosamente
        """
        try:
            # Cargar embeddings
            if self.db_path.exists():
                self.face_embeddings = np.load(self.db_path, allow_pickle=True).item()
                logger.info(f"Embeddings cargados: {len(self.face_embeddings)} personas")
            else:
                self.face_embeddings = {}
                logger.info("No se encontró archivo de embeddings, creando nuevo")
            
            # Cargar metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadata cargada: {len(self.metadata)} registros")
            else:
                self.metadata = {}
                logger.info("No se encontró archivo de metadata, creando nuevo")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cargando base de datos: {e}")
            self.face_embeddings = {}
            self.metadata = {}
            return False
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """
        Crear respaldo de la base de datos
        
        Args:
            backup_path: Ruta del respaldo (opcional)
            
        Returns:
            bool: True si se creó exitosamente
        """
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.db_path.stem}_backup_{timestamp}{self.db_path.suffix}"
                backup_path = self.db_path.parent / backup_path
            
            backup_path = Path(backup_path)
            
            # Crear backup de embeddings
            if self.db_path.exists():
                import shutil
                shutil.copy2(self.db_path, backup_path)
            
            # Crear backup de metadata
            backup_metadata_path = backup_path.with_suffix('.json')
            if self.metadata_path.exists():
                shutil.copy2(self.metadata_path, backup_metadata_path)
            
            logger.info(f"Backup creado: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creando backup: {e}")
            return False
    
    def clear_database(self) -> bool:
        """
        Limpiar base de datos completamente
        
        Returns:
            bool: True si se limpió exitosamente
        """
        try:
            self.face_embeddings = {}
            self.metadata = {}
            
            # Eliminar archivos
            if self.db_path.exists():
                self.db_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            
            logger.info("Base de datos limpiada")
            return True
            
        except Exception as e:
            logger.error(f"Error limpiando base de datos: {e}")
            return False
    
    def export_database(self, export_path: str, format: str = 'pickle') -> bool:
        """
        Exportar base de datos a otro formato
        
        Args:
            export_path: Ruta de exportación
            format: Formato ('pickle', 'json', 'csv')
            
        Returns:
            bool: True si se exportó exitosamente
        """
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'pickle':
                data = {
                    'embeddings': self.face_embeddings,
                    'metadata': self.metadata
                }
                with open(export_path, 'wb') as f:
                    pickle.dump(data, f)
                    
            elif format == 'json':
                # Solo exportar metadata (embeddings no son JSON serializable)
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                    
            elif format == 'csv':
                import pandas as pd
                df = pd.DataFrame.from_dict(self.metadata, orient='index')
                df.to_csv(export_path, index=True)
                
            else:
                logger.error(f"Formato de exportación no soportado: {format}")
                return False
            
            logger.info(f"Base de datos exportada: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando base de datos: {e}")
            return False
    
    def import_database(self, import_path: str, format: str = 'pickle') -> bool:
        """
        Importar base de datos desde otro formato
        
        Args:
            import_path: Ruta de importación
            format: Formato ('pickle', 'numpy')
            
        Returns:
            bool: True si se importó exitosamente
        """
        try:
            import_path = Path(import_path)
            
            if not import_path.exists():
                logger.error(f"Archivo de importación no existe: {import_path}")
                return False
            
            if format == 'pickle':
                with open(import_path, 'rb') as f:
                    data = pickle.load(f)
                    self.face_embeddings = data.get('embeddings', {})
                    self.metadata = data.get('metadata', {})
                    
            elif format == 'numpy':
                # Importar solo embeddings
                self.face_embeddings = np.load(import_path, allow_pickle=True).item()
                # Generar metadata básica
                for name in self.face_embeddings.keys():
                    if name not in self.metadata:
                        self.metadata[name] = {
                            'name': name,
                            'added_date': datetime.now().isoformat(),
                            'imported': True
                        }
                        
            else:
                logger.error(f"Formato de importación no soportado: {format}")
                return False
            
            logger.info(f"Base de datos importada: {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importando base de datos: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, 
                      threshold: float = 0.6, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Buscar personas similares al embedding de consulta
        
        Args:
            query_embedding: Embedding de consulta
            threshold: Umbral mínimo de similitud
            top_k: Número máximo de resultados
            
        Returns:
            List[Tuple[str, float]]: Lista de (nombre, similitud)
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            results = []
            
            for person_name, stored_embedding in self.face_embeddings.items():
                if stored_embedding.ndim == 1:
                    # Embedding único
                    similarity = cosine_similarity([query_embedding], [stored_embedding])[0, 0]
                    if similarity >= threshold:
                        results.append((person_name, float(similarity)))
                else:
                    # Múltiples embeddings - tomar el mejor
                    similarities = cosine_similarity([query_embedding], stored_embedding)[0]
                    best_sim = np.max(similarities)
                    if best_sim >= threshold:
                        results.append((person_name, float(best_sim)))
            
            # Ordenar por similitud descendente
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Retornar top_k resultados
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error en búsqueda de similares: {e}")
            return []
    
    def update_person_metadata(self, person_name: str, new_metadata: Dict) -> bool:
        """
        Actualizar metadata de una persona
        
        Args:
            person_name: Nombre de la persona
            new_metadata: Nueva metadata
            
        Returns:
            bool: True si se actualizó exitosamente
        """
        try:
            person_name = person_name.strip().lower()
            
            if person_name in self.metadata:
                self.metadata[person_name].update(new_metadata)
                self.metadata[person_name]['last_updated'] = datetime.now().isoformat()
                logger.info(f"Metadata actualizada para: {person_name}")
                return True
            else:
                logger.warning(f"Persona no encontrada para actualizar metadata: {person_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error actualizando metadata de {person_name}: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """
        Obtener estadísticas detalladas de la base de datos
        
        Returns:
            Dict: Estadísticas detalladas
        """
        try:
            stats = {
                'total_persons': len(self.face_embeddings),
                'total_embeddings': 0,
                'embedding_dimensions': None,
                'persons_with_multiple_embeddings': 0,
                'average_embeddings_per_person': 0.0,
                'database_size_mb': 0.0
            }
            
            embedding_counts = []
            
            for embeddings in self.face_embeddings.values():
                if embeddings.ndim == 1:
                    stats['total_embeddings'] += 1
                    embedding_counts.append(1)
                    if stats['embedding_dimensions'] is None:
                        stats['embedding_dimensions'] = embeddings.shape[0]
                else:
                    count = len(embeddings)
                    stats['total_embeddings'] += count
                    embedding_counts.append(count)
                    if count > 1:
                        stats['persons_with_multiple_embeddings'] += 1
                    if stats['embedding_dimensions'] is None:
                        stats['embedding_dimensions'] = embeddings.shape[1]
            
            if embedding_counts:
                stats['average_embeddings_per_person'] = np.mean(embedding_counts)
            
            # Calcular tamaño de archivo
            if self.db_path.exists():
                stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculando estadísticas: {e}")
            return {}
    
    def validate_database(self) -> Tuple[bool, List[str]]:
        """
        Validar integridad de la base de datos
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Verificar consistencia entre embeddings y metadata
            for person_name in self.face_embeddings:
                if person_name not in self.metadata:
                    issues.append(f"Persona sin metadata: {person_name}")
            
            for person_name in self.metadata:
                if person_name not in self.face_embeddings:
                    issues.append(f"Metadata sin embedding: {person_name}")
            
            # Verificar formato de embeddings
            expected_dim = None
            for person_name, embedding in self.face_embeddings.items():
                if embedding.ndim == 1:
                    current_dim = embedding.shape[0]
                elif embedding.ndim == 2:
                    current_dim = embedding.shape[1]
                else:
                    issues.append(f"Embedding con dimensiones incorrectas: {person_name}")
                    continue
                
                if expected_dim is None:
                    expected_dim = current_dim
                elif current_dim != expected_dim:
                    issues.append(f"Dimensión inconsistente en embedding de {person_name}: {current_dim} vs {expected_dim}")
            
            # Verificar que no hay nombres duplicados con diferente capitalización
            normalized_names = [name.lower() for name in self.face_embeddings.keys()]
            if len(normalized_names) != len(set(normalized_names)):
                issues.append("Nombres duplicados detectados (diferentes capitalizaciones)")
            
            is_valid = len(issues) == 0
            
            if is_valid:
                logger.info("Base de datos validada exitosamente")
            else:
                logger.warning(f"Problemas encontrados en la base de datos: {len(issues)}")
                for issue in issues:
                    logger.warning(f"  - {issue}")
            
            return is_valid, issues
            
        except Exception as e:
            error_msg = f"Error durante validación: {e}"
            logger.error(error_msg)
            return False, [error_msg]
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto save"""
        self.save_database()
    
    def __len__(self):
        """Número de personas en la base de datos"""
        return len(self.face_embeddings)
    
    def __contains__(self, person_name: str):
        """Verificar si una persona está en la base de datos"""
        return person_name.strip().lower() in self.face_embeddings
    
    def __str__(self):
        """Representación string de la base de datos"""
        info = self.get_database_info()
        return f"DatabaseManager({info['total_persons']} personas, {info['total_embeddings']} embeddings)"