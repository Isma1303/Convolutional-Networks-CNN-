import cv2

# Intentar abrir la cámara con backend de macOS
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
else:
    print("✅ Cámara detectada. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()  # Captura un frame de la cámara
        if not ret:
            print("⚠️ Error al capturar el video.")
            break
        
        cv2.imshow("Cámara en Vivo", frame)  # Muestra la imagen en una ventana
        
        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
