"""
Script para entrenar el modelo anti-spoofing
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config, TrainingConfig
from models.anti_spoof import AntiSpoofNet, AntiSpoofTrainer
from utils.image_processing import ImageProcessor, DataAugmentor

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AntiSpoofDataset(Dataset):
    """Dataset para entrenamiento anti-spoofing"""
    
    def __init__(self, real_images_dir: str, fake_images_dir: str, 
                 transform=None, augment=False):
        """
        Inicializar dataset
        
        Args:
            real_images_dir: Directorio con imágenes reales
            fake_images_dir: Directorio con imágenes falsas (fotos de fotos, pantallas, etc.)
            transform: Transformaciones a aplicar
            augment: Si aplicar data augmentation
        """
        self.transform = transform
        self.augment = augment
        self.augmentor = DataAugmentor()
        
        # Cargar rutas de imágenes
        self.image_paths = []
        self.labels = []
        
        # Imágenes reales (label = 1)
        real_dir = Path(real_images_dir)
        if real_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                real_images = list(real_dir.glob(ext)) + list(real_dir.glob(ext.upper()))
                for img_path in real_images:
                    self.image_paths.append(str(img_path))
                    self.labels.append(1)  # Real
        
        # Imágenes falsas (label = 0)
        fake_dir = Path(fake_images_dir)
        if fake_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                fake_images = list(fake_dir.glob(ext)) + list(fake_dir.glob(ext.upper()))
                for img_path in fake_images:
                    self.image_paths.append(str(img_path))
                    self.labels.append(0)  # Fake
        
        logger.info(f"Dataset cargado: {len(self.image_paths)} imágenes")
        logger.info(f"Reales: {sum(self.labels)}, Falsas: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Cargar imagen
            image_path = self.image_paths[idx]
            image = cv2.imread(image_path)
            
            if image is None:
                # Imagen corrupta, devolver imagen negra
                image = np.zeros((128, 128, 3), dtype=np.uint8)
            
            # Redimensionar a tamaño fijo
            image = cv2.resize(image, (128, 128))
            
            # Data augmentation si está habilitado
            if self.augment and np.random.random() > 0.5:
                # Aplicar transformaciones aleatorias
                if np.random.random() > 0.5:
                    angle = np.random.uniform(-15, 15)
                    image = self.augmentor.rotate_image(image, angle)
                
                if np.random.random() > 0.5:
                    image = self.augmentor.flip_image(image, 1)
                
                if np.random.random() > 0.5:
                    factor = np.random.uniform(0.8, 1.2)
                    image = self.augmentor.change_brightness(image, factor)
                
                if np.random.random() > 0.7:
                    image = self.augmentor.add_noise(image, 'gaussian')
            
            # Convertir BGR a RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Aplicar transformaciones de PyTorch
            if self.transform:
                image = self.transform(image)
            
            label = self.labels[idx]
            return image, label
            
        except Exception as e:
            logger.error(f"Error cargando imagen {idx}: {e}")
            # Devolver imagen negra en caso de error
            dummy_image = np.zeros((128, 128, 3), dtype=np.uint8)
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0

def create_synthetic_dataset(output_dir: str, num_real: int = 1000, num_fake: int = 1000):
    """
    Crear dataset sintético para pruebas
    
    Args:
        output_dir: Directorio de salida
        num_real: Número de imágenes reales sintéticas
        num_fake: Número de imágenes falsas sintéticas
    """
    try:
        output_path = Path(output_dir)
        real_dir = output_path / "real"
        fake_dir = output_path / "fake"
        
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creando dataset sintético en {output_dir}")
        
        # Crear imágenes "reales" (patrones complejos)
        for i in tqdm(range(num_real), desc="Generando imágenes reales"):
            # Crear patrón complejo que simula texturas de piel
            image = np.random.randint(100, 200, (128, 128, 3), dtype=np.uint8)
            
            # Agregar texturas
            noise = np.random.normal(0, 15, (128, 128, 3))
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
            
            # Agregar gradientes suaves
            y, x = np.ogrid[:128, :128]
            gradient = (x + y) / 255.0
            for c in range(3):
                image[:, :, c] = np.clip(image[:, :, c] * (0.8 + 0.4 * gradient), 0, 255)
            
            cv2.imwrite(str(real_dir / f"real_{i:04d}.jpg"), image)
        
        # Crear imágenes "falsas" (patrones simples, uniformes)
        for i in tqdm(range(num_fake), desc="Generando imágenes falsas"):
            # Crear patrones uniformes que simulan pantallas/fotos
            if i % 3 == 0:
                # Patrón uniforme
                color = np.random.randint(50, 200, 3)
                image = np.full((128, 128, 3), color, dtype=np.uint8)
            elif i % 3 == 1:
                # Patrón de rayas (simula pantalla)
                image = np.zeros((128, 128, 3), dtype=np.uint8)
                for y in range(0, 128, 4):
                    image[y:y+2, :, :] = np.random.randint(100, 255, 3)
            else:
                # Patrón de cuadrícula
                image = np.random.randint(80, 120, (128, 128, 3), dtype=np.uint8)
                image[::8, :, :] = 255
                image[:, ::8, :] = 255
            
            cv2.imwrite(str(fake_dir / f"fake_{i:04d}.jpg"), image)
        
        logger.info(f"Dataset sintético creado: {num_real} reales, {num_fake} falsas")
        
    except Exception as e:
        logger.error(f"Error creando dataset sintético: {e}")

def plot_training_history(history: dict, save_path: str = None):
    """
    Graficar historial de entrenamiento
    
    Args:
        history: Historial de entrenamiento
        save_path: Ruta para guardar la gráfica
    """
    try:
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history['train_losses'], label='Train Loss')
        ax1.plot(history['val_losses'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(history['train_accuracies'], label='Train Accuracy')
        ax2.plot(history['val_accuracies'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfica guardada: {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error graficando historial: {e}")

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> dict:
    """
    Evaluar modelo en conjunto de prueba
    
    Args:
        model: Modelo a evaluar
        test_loader: DataLoader de prueba
        device: Dispositivo de cómputo
        
    Returns:
        dict: Métricas de evaluación
    """
    model.eval()
    
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluando"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Calcular métricas detalladas
            for i in range(len(target)):
                if target[i] == 1 and predicted[i] == 1:
                    true_positives += 1
                elif target[i] == 0 and predicted[i] == 1:
                    false_positives += 1
                elif target[i] == 0 and predicted[i] == 0:
                    true_negatives += 1
                elif target[i] == 1 and predicted[i] == 0:
                    false_negatives += 1
    
    accuracy = 100 * correct / total
    
    # Calcular métricas adicionales
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives
    }
    
    return metrics

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Entrenar modelo anti-spoofing")
    parser.add_argument("--real_dir", "-r", type=str, required=True, 
                       help="Directorio con imágenes reales")
    parser.add_argument("--fake_dir", "-f", type=str, required=True,
                       help="Directorio con imágenes falsas")
    parser.add_argument("--epochs", "-e", type=int, default=TrainingConfig.NUM_EPOCHS,
                       help="Número de épocas")
    parser.add_argument("--batch_size", "-b", type=int, default=TrainingConfig.BATCH_SIZE,
                       help="Tamaño del batch")
    parser.add_argument("--learning_rate", "-lr", type=float, default=TrainingConfig.LEARNING_RATE,
                       help="Tasa de aprendizaje")
    parser.add_argument("--output_model", "-o", type=str, default=str(Config.ANTI_SPOOF_MODEL_PATH),
                       help="Ruta del modelo de salida")
    parser.add_argument("--create_synthetic", action="store_true",
                       help="Crear dataset sintético de prueba")
    parser.add_argument("--synthetic_dir", type=str, default="synthetic_data",
                       help="Directorio para dataset sintético")
    
    args = parser.parse_args()
    
    try:
        # Crear directorios necesarios
        Config.create_directories()
        
        # Crear dataset sintético si se solicita
        if args.create_synthetic:
            logger.info("Creando dataset sintético...")
            create_synthetic_dataset(args.synthetic_dir)
            print(f"\nDataset sintético creado en: {args.synthetic_dir}")
            print("Usa los siguientes directorios para entrenar:")
            print(f"  --real_dir {args.synthetic_dir}/real")
            print(f"  --fake_dir {args.synthetic_dir}/fake")
            return
        
        # Verificar que existan los directorios
        if not os.path.exists(args.real_dir):
            logger.error(f"Directorio de imágenes reales no existe: {args.real_dir}")
            return
        
        if not os.path.exists(args.fake_dir):
            logger.error(f"Directorio de imágenes falsas no existe: {args.fake_dir}")
            return
        
        # Configurar device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando device: {device}")
        
        # Definir transformaciones
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Crear dataset
        logger.info("Cargando dataset...")
        dataset = AntiSpoofDataset(
            real_images_dir=args.real_dir,
            fake_images_dir=args.fake_dir,
            transform=transform,
            augment=True
        )
        
        if len(dataset) == 0:
            logger.error("Dataset vacío")
            return
        
        # Dividir dataset
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        logger.info(f"División del dataset:")
        logger.info(f"  Entrenamiento: {len(train_dataset)}")
        logger.info(f"  Validación: {len(val_dataset)}")
        logger.info(f"  Prueba: {len(test_dataset)}")
        
        # Crear data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        
        # Crear modelo
        model = AntiSpoofNet(input_size=128, num_classes=2)
        
        # Crear entrenador
        trainer = AntiSpoofTrainer(model, device=device)
        
        # Configurar learning rate si se especifica
        if args.learning_rate != TrainingConfig.LEARNING_RATE:
            trainer.optimizer = torch.optim.Adam(
                model.parameters(), lr=args.learning_rate, weight_decay=1e-4
            )
        
        # Entrenar modelo
        logger.info(f"Iniciando entrenamiento por {args.epochs} épocas...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            save_path=args.output_model
        )
        
        # Evaluar en conjunto de prueba
        logger.info("Evaluando modelo en conjunto de prueba...")
        test_metrics = evaluate_model(model, test_loader, device)
        
        # Mostrar resultados
        print("\n" + "="*50)
        print("RESULTADOS DEL ENTRENAMIENTO")
        print("="*50)
        print(f"Mejor accuracy de validación: {history['best_val_accuracy']:.2f}%")
        print(f"Accuracy en prueba: {test_metrics['accuracy']:.2f}%")
        print(f"Precision: {test_metrics['precision']:.3f}")
        print(f"Recall: {test_metrics['recall']:.3f}")
        print(f"F1-Score: {test_metrics['f1_score']:.3f}")
        print(f"Modelo guardado en: {args.output_model}")
        
        # Graficar historial
        plot_save_path = str(Path(args.output_model).parent / "training_history.png")
        plot_training_history(history, plot_save_path)
        
        logger.info("Entrenamiento completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()