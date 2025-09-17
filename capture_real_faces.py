import cv2
import os
from pathlib import Path

def capture_real_faces():
    # Crear directorio
    real_dir = Path("training_data/real")
    real_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = len(list(real_dir.glob("*.jpg")))  # Continuar numeración
    
    print("=== CAPTURA DE ROSTROS REALES ===")
    print("Instrucciones:")
    print("- Posiciona diferentes personas frente a la cámara")
    print("- Varía ángulos e iluminación")
    print("- Presiona ESPACIO para capturar")
    print("- Presiona Q para salir")
    print(f"Imágenes actuales: {count}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mostrar información
        cv2.putText(frame, f"Rostros reales: {count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "ESPACIO: Capturar | Q: Salir", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Capturar Rostros Reales", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            filename = real_dir / f"real_{count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            count += 1
            print(f"Rostro real guardado: {filename}")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total rostros reales: {count}")

if __name__ == "__main__":
    capture_real_faces()