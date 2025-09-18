"""
Modelo Anti-Spoofing para detectar ataques con fotos/videos
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import logging
from typing import Tuple, Optional

from config.config import Config
logger = logging.getLogger(__name__)

class AntiSpoofNet(nn.Module):
    """Red neuronal convolucional para detección anti-spoofing"""
    
    def __init__(self, input_size: int = 128, num_classes: int = 2):
        """
        Inicializar la red anti-spoofing
        
        Args:
            input_size: Tamaño de entrada (asume imágenes cuadradas)
            num_classes: Número de clases (2: fake, real)
        """
        super(AntiSpoofNet, self).__init__()
        
        # Capas convolucionales
        self.conv_layers = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Bloque 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.4)
        )
        
        # Capas fully connected
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos de la red"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AntiSpoofDetector:

    """Detector anti-spoofing que utiliza la red neuronal"""

    

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Inicializar detector anti-spoofing

        Args:
            model_path: Ruta al modelo pre-entrenado
            device: 'cuda', 'cpu' o 'auto'
        """
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Inicializar modelo
        self.model = AntiSpoofNet().to(self.device)
        self.model.eval()

        # Estado del modelo
        self.model_loaded = False

        # Cargar modelo pre-entrenado si se proporciona
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model_loaded = True
                logger.info(f"Modelo anti-spoofing cargado desde {model_path}")
            except Exception as e:
                logger.error(f"Error cargando modelo: {e}")
        else:
            logger.warning("No se encontró modelo pre-entrenado, usando modelo sin entrenar")

        # Transformaciones para preprocesamiento
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])

        ])

        # Suavizado temporal y tolerancia a falsos negativos
        smooth_frames = max(1, getattr(Config, 'ANTI_SPOOF_SMOOTH_FRAMES', 1))
        self.use_smoothing = getattr(Config, 'ANTI_SPOOF_SMOOTHING', False)
        history_size = smooth_frames if self.use_smoothing else 1
        self.confidence_history = deque(maxlen=history_size)

        tolerance = getattr(Config, 'ANTI_SPOOF_FAIL_TOLERANCE', smooth_frames)
        if self.use_smoothing:
            tolerance = max(2, tolerance)
        else:
            tolerance = max(1, tolerance)
        self.low_confidence_frames = 0
        self.low_confidence_patience = tolerance
    
    def preprocess_image(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocesar imagen para el modelo
        
        Args:
            face_image: Imagen BGR del rostro
            
        Returns:
            torch.Tensor: Tensor preprocesado
        """
        try:
            # Convertir BGR a RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Aplicar transformaciones
            face_tensor = self.transform(face_rgb)
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            return face_tensor
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            return None
    def _update_confidence(self, confidence: float) -> float:
        """Actualizar historial de confianza y obtener valor suavizado."""
        self.confidence_history.append(confidence)

        if self.use_smoothing:
            return float(np.mean(self.confidence_history))
        return float(confidence)

    def _apply_low_confidence_logic(self, smoothed_confidence: float) -> Tuple[bool, float]:
        """Aplicar lógica de tolerancia ante predicciones de baja confianza."""
        threshold = getattr(Config, 'ANTI_SPOOF_THRESHOLD', 0.5)

        if smoothed_confidence >= threshold:
            self.low_confidence_frames = 0
            return True, smoothed_confidence

        self.low_confidence_frames += 1
        is_real = self.low_confidence_frames < self.low_confidence_patience
        return is_real, smoothed_confidence

    def detect_spoofing(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """
        Detectar si la imagen es un ataque de spoofing

        Args:
            face_image: Imagen BGR del rostro

        Returns:
            tuple: (is_real, confidence)
        """
        try:
            if not self.model_loaded:
                # Si no hay modelo entrenado, no bloquear al usuario
                return True, 1.0

            # Preprocesar imagen
            face_tensor = self.preprocess_image(face_image)
            if face_tensor is None:
                smoothed = self._update_confidence(0.0)
                return self._apply_low_confidence_logic(smoothed)

            # Inferencia
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)

                # Asumiendo que índice 0 = fake, índice 1 = real
                prob_real = probabilities[0, 1].item()

            smoothed = self._update_confidence(prob_real)
            is_real, final_confidence = self._apply_low_confidence_logic(smoothed)

            # Si la confianza vuelve a subir, reiniciar contador de fallos
            if prob_real >= getattr(Config, 'ANTI_SPOOF_THRESHOLD', 0.5):
                self.low_confidence_frames = 0

            return is_real, final_confidence

        except Exception as e:
            logger.error(f"Error en detección anti-spoofing: {e}")
            smoothed = self._update_confidence(0.0)
            return self._apply_low_confidence_logic(smoothed)
    
    def batch_detect(self, face_images: list) -> list:
        """
        Detección en lote

        Args:
            face_images: Lista de imágenes BGR

        Returns:
            list: Lista de tuplas (is_real, confidence)
        """
        results = []

        try:
            if not self.model_loaded:
                return [(True, 1.0)] * len(face_images)

            # Preprocesar todas las imágenes
            batch_tensors = []
            valid_indices = []

            for i, face_image in enumerate(face_images):
                tensor = self.preprocess_image(face_image)
                if tensor is not None:
                    batch_tensors.append(tensor.squeeze(0))
                    valid_indices.append(i)
            
            if not batch_tensors:
                return [(False, 0.0)] * len(face_images)
            
            # Crear batch
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Inferencia en lote
            with torch.no_grad():
                outputs = self.model(batch)
                probabilities = F.softmax(outputs, dim=1)
                
                # Procesar resultados
                batch_results = []
                for prob in probabilities:
                    prob_real = prob[1].item()
                    is_real = prob_real > Config.ANTI_SPOOF_THRESHOLD
                
                    batch_results.append((is_real, prob_real))
            
            # Mapear resultados a índices originales
            result_idx = 0
            for i in range(len(face_images)):
                if i in valid_indices:
                    results.append(batch_results[result_idx])
                    result_idx += 1
                else:
                    results.append((False, 0.0))
                    
        except Exception as e:
            logger.error(f"Error en detección en lote: {e}")
            results = [(False, 0.0)] * len(face_images)
        
        return results


class AntiSpoofTrainer:
    """Entrenador para el modelo anti-spoofing"""
    
    def __init__(self, model: AntiSpoofNet, device: str = 'auto'):
        """
        Inicializar entrenador
        
        Args:
            model: Modelo a entrenar
            device: Dispositivo de entrenamiento
        """
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        
        # Configurar optimizador y criterio
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=0.001, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5)
        
        # Métricas de entrenamiento
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Entrenar una época
        
        Args:
            train_loader: DataLoader de entrenamiento
            
        Returns:
            tuple: (loss_promedio, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Estadísticas
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validar modelo
        
        Args:
            val_loader: DataLoader de validación
            
        Returns:
            tuple: (loss_promedio, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 50, save_path: str = None) -> dict:
        """
        Entrenar modelo completo
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            num_epochs: Número de épocas
            save_path: Ruta para guardar el mejor modelo
            
        Returns:
            dict: Historial de entrenamiento
        """
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 10
        
        logger.info(f"Iniciando entrenamiento por {num_epochs} épocas")
        
        for epoch in range(num_epochs):
            # Entrenar
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validar
            val_loss, val_acc = self.validate(val_loader)
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Guardar métricas
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.train_accuracies.append(train_acc)

            self.val_accuracies.append(val_acc)

            

            # Log progreso

            logger.info(f'Época {epoch+1}/{num_epochs}:')

            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            

            # Guardar mejor modelo

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                patience_counter = 0

                

                if save_path:

                    torch.save(self.model.state_dict(), save_path)

                    logger.info(f'Mejor modelo guardado en {save_path}')

            else:

                patience_counter += 1

            

            # Early stopping

            if patience_counter >= max_patience:

                logger.info(f'Early stopping en época {epoch+1}')

                break

        

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': best_val_acc
        }
