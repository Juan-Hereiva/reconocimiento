import cv2
import os
from pathlib import Path

def capture_fake_attacks():
    fake_dir = Path("training_data/fake")
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = len(list(fake_dir.glob("*.jpg")))
    
    print("=== CAPTURA DE ATAQUES FALSOS ===")
    print("Tipos de ataques a capturar:")
    print("1. Fotos impresas en papel")
    print("2. Fotos en pantalla de móvil")
    print("3. Fotos en monitor de computadora")
    print("4. Videos reproducidos en pantalla")
    print("5. Fotos de fotos (foto de una foto)")
    print()
    print("- Apunta la cámara al ataque")
    print("- Presiona ESPACIO para capturar")
    print("- Presiona Q para salir")
    print(f"Ataques actuales: {count}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.putText(frame, f"Ataques falsos: {count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "ESPACIO: Capturar | Q: Salir", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Apunta a: foto/pantalla/papel", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Capturar Ataques Falsos", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            filename = fake_dir / f"fake_{count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            count += 1
            print(f"Ataque falso guardado: {filename}")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total ataques falsos: {count}")

if __name__ == "__main__":
    capture_fake_attacks()