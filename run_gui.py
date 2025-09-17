#!/usr/bin/env python3
"""
Script para ejecutar la interfaz gráfica del sistema de reconocimiento facial
"""
import sys
import os
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Agregar el directorio actual al path para imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def check_dependencies():
    """Verificar dependencias necesarias"""
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import mediapipe
    except ImportError:
        missing_deps.append("mediapipe")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    return missing_deps

def show_dependency_error(missing_deps):
    """Mostrar error de dependencias faltantes"""
    root = tk.Tk()
    root.withdraw()  # Ocultar ventana principal
    
    deps_text = "\n".join(f"  - {dep}" for dep in missing_deps)
    
    message = f"""Dependencias faltantes:

{deps_text}

Instale las dependencias ejecutando:
pip install -r requirements.txt

O instale manualmente:
pip install {' '.join(missing_deps)}"""
    
    messagebox.showerror("Dependencias Faltantes", message)
    root.destroy()

def check_project_structure():
    """Verificar estructura del proyecto"""
    required_dirs = ["config", "models", "utils", "data"]
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not (current_dir / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        root = tk.Tk()
        root.withdraw()
        
        dirs_text = "\n".join(f"  - {dir_name}/" for dir_name in missing_dirs)
        
        message = f"""Estructura de proyecto incompleta.

Directorios faltantes:
{dirs_text}

Asegúrese de ejecutar desde el directorio raíz del proyecto."""
        
        messagebox.showerror("Estructura de Proyecto", message)
        root.destroy()
        return False
    
    return True

def create_directories():
    """Crear directorios necesarios"""
    try:
        from config.config import Config
        Config.create_directories()
        return True
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Error creando directorios: {e}")
        root.destroy()
        return False

def main():
    """Función principal"""
    print("🚀 Iniciando Sistema de Reconocimiento Facial con Interfaz Gráfica")
    print("=" * 60)
    
    # Verificar dependencias
    print("📦 Verificando dependencias...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print("❌ Dependencias faltantes encontradas")
        show_dependency_error(missing_deps)
        return 1
    
    print("✅ Todas las dependencias están instaladas")
    
    # Verificar estructura del proyecto
    print("📁 Verificando estructura del proyecto...")
    if not check_project_structure():
        return 1
    
    print("✅ Estructura del proyecto verificada")
    
    # Crear directorios necesarios
    print("📂 Creando directorios necesarios...")
    if not create_directories():
        return 1
    
    print("✅ Directorios creados/verificados")
    
    # Importar y ejecutar GUI
    try:
        print("🖥️  Cargando interfaz gráfica...")
        from gui_main import FaceRecognitionGUI
        
        print("✅ Interfaz cargada exitosamente")
        print("🎯 Iniciando aplicación...")
        
        app = FaceRecognitionGUI()
        app.run()
        
        print("👋 Aplicación cerrada correctamente")
        return 0
        
    except ImportError as e:
        root = tk.Tk()
        root.withdraw()
        
        message = f"""Error importando módulos:
{e}

Asegúrese de que todos los archivos estén presentes:
  - gui_main.py
  - config/config.py
  - models/*.py
  - utils/*.py"""
        
        messagebox.showerror("Error de Importación", message)
        root.destroy()
        return 1
        
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        
        message = f"""Error ejecutando la aplicación:
{e}

Revise la consola para más detalles."""
        
        messagebox.showerror("Error de Ejecución", message)
        root.destroy()
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Aplicación interrumpida por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")
        sys.exit(1)