import cv2
import os

# Nombre de la categoría (ejemplo: "resistor", "capacitor", "chip")
category = input("Introduce el nombre del objeto que estás capturando: ")

# Crear carpeta para la categoría si no existe
save_path = f"data/{category}"
os.makedirs(save_path, exist_ok=True)

# Abrir la cámara
cap = cv2.VideoCapture(0)  # Usa el backend predeterminado

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
else:
    print(f"📸 Capturando imágenes para la categoría: {category}")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer el frame de la cámara.")
            break

        # Mostrar el video en vivo
        cv2.imshow("Captura de imágenes", frame)

        # Detectar teclas presionadas
        key = cv2.waitKey(1) & 0xFF

        # Guardar imagen cuando se presiona 's'
        if key == ord('s'):
            image_path = f"{save_path}/{count}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"✅ Imagen guardada: {image_path}")
            count += 1

        # Salir con la tecla 'q'
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()