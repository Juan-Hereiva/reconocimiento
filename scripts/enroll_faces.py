"""
Script para registrar rostros desde carpetas de imágenes
"""
import os
import sys
import cv2
import argparse
import logging
from pathlib import Path

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.face_detector import FaceDetector
from models.face_recognizer import FaceRecognizer
from utils.database_manager import DatabaseManager
from utils.image_processing import ImageProcessor, ImageValidator

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceEnroller:
    """Registrador de rostros desde imágenes"""
    
    def __init__(self):
        """Inicializar registrador"""
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.database_manager = DatabaseManager(str(Config.FACE_DATABASE_PATH))
        self.image_processor = ImageProcessor()
        self.image_validator = ImageValidator()
        
        logger.info("FaceEnroller inicializado")
    
    def enroll_person_from_folder(self, person_name: str, folder_path: str, 
                                 max_images: int = 20) -> bool:
        """
        Registrar persona desde carpeta de imágenes
        
        Args:
            person_name: Nombre de la persona
            folder_path: Ruta de la carpeta con imágenes
            max_images: Número máximo de imágenes a procesar
            
        Returns:
            bool: True si se registró exitosamente
        """
        try:
            folder_path = Path(folder_path)
            
            if not folder_path.exists():
                logger.error(f"Carpeta no existe: {folder_path}")
                return False
            
            # Buscar archivos de imagen
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(folder_path.glob(f"*{ext}"))
                image_files.extend(folder_path.glob(f"*{ext.upper()}"))
            
            if not image_files:
                logger.error(f"No se encontraron imágenes en: {folder_path}")
                return False
            
            logger.info(f"Encontradas {len(image_files)} imágenes para {person_name}")
            
            # Filtrar imágenes de buena calidad
            good_images = self.image_validator.filter_good_images([str(f) for f in image_files])
            
            if not good_images:
                logger.error("No se encontraron imágenes de buena calidad")
                return False
            
            # Limitar número de imágenes
            if len(good_images) > max_images:
                good_images = good_images[:max_images]
                logger.info(f"Limitando a {max_images} imágenes")
            
            # Procesar imágenes
            valid_faces = []
            processed_count = 0
            
            for image_path in good_images:
                try:
                    # Cargar imagen
                    image = self.image_processor.load_image(image_path)
                    if image is None:
                        continue
                    
                    # Mejorar calidad si es necesario
                    image = self.image_processor.normalize_lighting(image)
                    image = self.image_processor.denoise_image(image, method='bilateral')
                    
                    # Detectar rostro
                    bbox, landmarks = self.face_detector.detect_face_and_landmarks(image)
                    
                    if bbox is None:
                        logger.warning(f"No se detectó rostro en: {image_path}")
                        continue
                    
                    # Extraer y verificar rostro
                    face_crop = self.face_detector.crop_face(image, bbox, size=Config.FACE_CROP_SIZE)
                    
                    if face_crop is None:
                        logger.warning(f"No se pudo extraer rostro de: {image_path}")
                        continue
                    
                    # Verificar calidad del rostro
                    is_good, metrics = self.image_validator.check_image_quality(face_crop)
                    
                    if not is_good:
                        logger.warning(f"Rostro de baja calidad en: {image_path}")
                        continue
                    
                    valid_faces.append(face_crop)
                    processed_count += 1
                    
                    logger.info(f"Procesado {processed_count}/{len(good_images)}: {Path(image_path).name}")
                    
                except Exception as e:
                    logger.error(f"Error procesando {image_path}: {e}")
                    continue
            
            if len(valid_faces) < 3:
                logger.error(f"Se necesitan al menos 3 rostros válidos, solo se encontraron {len(valid_faces)}")
                return False
            
            logger.info(f"Rostros válidos extraídos: {len(valid_faces)}")
            
            # Registrar en el sistema
            success = self.face_recognizer.add_person_to_database(
                person_name, valid_faces, method='average'
            )
            
            if success:
                # Agregar a database manager
                embeddings = self.face_recognizer.face_database[person_name]
                metadata = {
                    'total_images': len(image_files),
                    'valid_faces': len(valid_faces),
                    'source_folder': str(folder_path),
                    'registration_method': 'folder_batch'
                }
                
                self.database_manager.add_person(person_name, embeddings, metadata)
                self.database_manager.save_database()
                
                logger.info(f"Persona '{person_name}' registrada exitosamente")
                return True
            else:
                logger.error("Error registrando embeddings")
                return False
                
        except Exception as e:
            logger.error(f"Error registrando persona {person_name}: {e}")
            return False
    
    def enroll_from_directory_structure(self, root_dir: str) -> dict:
        """
        Registrar múltiples personas desde estructura de directorios
        Estructura esperada: root_dir/person_name/images...
        
        Args:
            root_dir: Directorio raíz
            
        Returns:
            dict: Resultados del registro
        """
        try:
            root_path = Path(root_dir)
            
            if not root_path.exists():
                logger.error(f"Directorio raíz no existe: {root_path}")
                return {"success": False, "error": "Directorio no existe"}
            
            results = {
                "success": True,
                "total_persons": 0,
                "registered_persons": [],
                "failed_persons": [],
                "errors": []
            }
            
            # Buscar subdirectorios (cada uno representa una persona)
            person_dirs = [d for d in root_path.iterdir() if d.is_dir()]
            
            if not person_dirs:
                error_msg = f"No se encontraron subdirectorios en: {root_path}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            logger.info(f"Encontrados {len(person_dirs)} directorios de personas")
            
            for person_dir in person_dirs:
                person_name = person_dir.name
                logger.info(f"Procesando persona: {person_name}")
                
                # Verificar si ya existe
                if person_name.lower() in self.face_recognizer.face_database:
                    logger.warning(f"Persona '{person_name}' ya existe, omitiendo...")
                    results["failed_persons"].append({
                        "name": person_name,
                        "reason": "Ya existe en la base de datos"
                    })
                    continue
                
                # Registrar persona
                success = self.enroll_person_from_folder(person_name, str(person_dir))
                
                if success:
                    results["registered_persons"].append(person_name)
                    logger.info(f"✓ {person_name} registrado exitosamente")
                else:
                    results["failed_persons"].append({
                        "name": person_name,
                        "reason": "Error en el procesamiento"
                    })
                    logger.error(f"✗ Error registrando {person_name}")
                
                results["total_persons"] += 1
            
            # Resumen final
            success_count = len(results["registered_persons"])
            failed_count = len(results["failed_persons"])
            
            logger.info(f"\n=== RESUMEN ===")
            logger.info(f"Total procesadas: {results['total_persons']}")
            logger.info(f"Registradas exitosamente: {success_count}")
            logger.info(f"Fallidas: {failed_count}")
            
            if results["registered_persons"]:
                logger.info("Personas registradas:")
                for person in results["registered_persons"]:
                    logger.info(f"  - {person}")
            
            if results["failed_persons"]:
                logger.info("Personas fallidas:")
                for person_info in results["failed_persons"]:
                    logger.info(f"  - {person_info['name']}: {person_info['reason']}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error en registro masivo: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def enroll_single_image(self, person_name: str, image_path: str) -> bool:
        """
        Registrar persona desde una sola imagen
        
        Args:
            person_name: Nombre de la persona
            image_path: Ruta de la imagen
            
        Returns:
            bool: True si se registró exitosamente
        """
        try:
            # Cargar imagen
            image = self.image_processor.load_image(image_path)
            if image is None:
                logger.error(f"No se pudo cargar la imagen: {image_path}")
                return False
            
            # Detectar rostro
            bbox, landmarks = self.face_detector.detect_face_and_landmarks(image)
            
            if bbox is None:
                logger.error("No se detectó rostro en la imagen")
                return False
            
            # Extraer rostro
            face_crop = self.face_detector.crop_face(image, bbox, size=Config.FACE_CROP_SIZE)
            
            if face_crop is None:
                logger.error("No se pudo extraer rostro")
                return False
            
            # Verificar calidad
            is_good, metrics = self.image_validator.check_image_quality(face_crop)
            
            if not is_good:
                logger.warning("Imagen de baja calidad, continuando...")
            
            # Registrar (con una sola imagen, no es muy confiable)
            success = self.face_recognizer.add_person_to_database(
                person_name, [face_crop], method='average'
            )
            
            if success:
                # Agregar a database manager
                embeddings = self.face_recognizer.face_database[person_name]
                metadata = {
                    'source_image': image_path,
                    'registration_method': 'single_image',
                    'quality_metrics': metrics
                }
                
                self.database_manager.add_person(person_name, embeddings, metadata)
                self.database_manager.save_database()
                
                logger.info(f"Persona '{person_name}' registrada desde imagen única")
                logger.warning("NOTA: El registro con una sola imagen es menos confiable")
                return True
            else:
                logger.error("Error registrando embedding")
                return False
                
        except Exception as e:
            logger.error(f"Error registrando desde imagen única: {e}")
            return False

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Registrar rostros en el sistema")
    parser.add_argument("--person_name", "-n", type=str, help="Nombre de la persona")
    parser.add_argument("--folder", "-f", type=str, help="Carpeta con imágenes de la persona")
    parser.add_argument("--image", "-i", type=str, help="Imagen única de la persona")
    parser.add_argument("--batch_dir", "-b", type=str, help="Directorio con múltiples personas")
    parser.add_argument("--max_images", "-m", type=int, default=20, help="Máximo número de imágenes por persona")
    
    args = parser.parse_args()
    
    # Crear directorios necesarios
    Config.create_directories()
    
    # Inicializar enrollador
    enroller = FaceEnroller()
    
    try:
        if args.batch_dir:
            # Registro masivo
            logger.info(f"Iniciando registro masivo desde: {args.batch_dir}")
            results = enroller.enroll_from_directory_structure(args.batch_dir)
            
            if results["success"]:
                print(f"\n✓ Registro masivo completado")
                print(f"Personas registradas: {len(results['registered_persons'])}")
            else:
                print(f"\n✗ Error en registro masivo: {results.get('error', 'Error desconocido')}")
        
        elif args.person_name and args.folder:
            # Registro desde carpeta
            logger.info(f"Registrando {args.person_name} desde carpeta: {args.folder}")
            success = enroller.enroll_person_from_folder(args.person_name, args.folder, args.max_images)
            
            if success:
                print(f"\n✓ {args.person_name} registrado exitosamente")
            else:
                print(f"\n✗ Error registrando {args.person_name}")
        
        elif args.person_name and args.image:
            # Registro desde imagen única
            logger.info(f"Registrando {args.person_name} desde imagen: {args.image}")
            success = enroller.enroll_single_image(args.person_name, args.image)
            
            if success:
                print(f"\n✓ {args.person_name} registrado exitosamente")
                print("⚠️  ADVERTENCIA: Registro con una sola imagen es menos confiable")
            else:
                print(f"\n✗ Error registrando {args.person_name}")
        
        else:
            parser.print_help()
            print("\nEjemplos de uso:")
            print("  python enroll_faces.py -n 'Juan Perez' -f /path/to/juan_images/")
            print("  python enroll_faces.py -n 'Maria' -i /path/to/maria.jpg")
            print("  python enroll_faces.py -b /path/to/all_persons/")
    
    except Exception as e:
        logger.error(f"Error en función principal: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()