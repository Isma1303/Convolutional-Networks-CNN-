import cv2
import os

# Nombre de la categoría (ejemplo: "resistor", "capacitor", "chip")
category = input("Introduce el nombre del objeto que estás capturando: ")

# Crear carpeta para la categoría si no existe
save_path = f"data/{category}"
os.makedirs(save_path, exist_ok=True)

# Abrir la cámara
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
else:
    print(f"📸 Capturando imágenes para la categoría: {category}")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mostrar el video en vivo
        cv2.imshow("Captura de imágenes", frame)

        # Guardar imagen cuando se presiona 's'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            try:
                # Preguntar al usuario si desea capturar la imagen
                confirm = input("¿Deseas capturar esta imagen? (sí/no): ").strip().lower()
                if confirm in ['sí', 'si', 's']:
                    image_path = f"{save_path}/{count}.jpg"
                    cv2.imwrite(image_path, frame)
                    print(f"✅ Imagen guardada: {image_path}")
                    count += 1
                else:
                    print("❌ Imagen descartada.")
            except Exception as e:
                print(f"Error al capturar la imagen: {e}")

        # Salir con la tecla 'q'
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()