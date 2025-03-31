import cv2
import torch
import numpy as np
from cnn_model import CNNModel
import time

# Cargar modelo entrenado
model = CNNModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Verificar si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Inicializar cámara
cap = cv2.VideoCapture(0)

try:
    if not cap.isOpened():
        raise Exception("❌ No se pudo abrir la cámara")
    else:
        print("✅ Cámara detectada. Presiona 'q' para salir.")
except Exception as e:
    print(e)

# Sistema de etiquetas (ejemplo)
labels = {0: "Gato", 1: "Perro", 2: "Pájaro"}

# Límite de FPS
fps_limit = 30
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Controlar FPS
        if (time.time() - start_time) < (1 / fps_limit):
            continue
        start_time = time.time()

        # Preprocesamiento de la imagen
        img = cv2.resize(frame, (64, 64))
        img = img / 255.0  # Normalizar valores entre 0 y 1
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

        # Predicción con manejo de errores
        try:
            output = model(img)
            prediction = torch.argmax(output, dim=1).item()
            prediction_label = labels.get(prediction, "Desconocido")
        except Exception as e:
            print(f"Error durante la predicción: {e}")
            prediction_label = "Error"

        print(f"Predicción: {prediction_label}")

        # Mostrar la predicción en la ventana
        cv2.rectangle(frame, (5, 5), (300, 40), (0, 0, 0), -1)  # Fondo negro
        cv2.putText(frame, f"Predicción: {prediction_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error inesperado: {e}")
finally:
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()