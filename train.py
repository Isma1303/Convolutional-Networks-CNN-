import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model import CNNModel
import cv2
import numpy as np

# Inicializar modelo
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def capture_data():
    # Intentar abrir la cámara
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return None, None
    else:
        print("✅ Cámara detectada. Presiona 'q' para salir.")

    data, labels = [], []
    
    for i in range(100):  # Capturar 100 imágenes para entrenamiento
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Error al capturar el video.")
            continue
        
        # Mostrar el cuadro capturado
        cv2.imshow("Capturando", frame)
        print(f"✅ Capturando imagen {i+1}/100")  # Imprimir progreso en consola
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
            print("🛑 Captura interrumpida por el usuario.")
            break
        
        img = cv2.resize(frame, (64, 64))
        img = np.transpose(img, (2, 0, 1))
        data.append(img)
        labels.append(0)  # Placeholder para las etiquetas
    
    cap.release()
    cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Entrenamiento
data, labels = capture_data()
if data is not None and labels is not None:
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "model.pth")
else:
    print("❌ No se pudo realizar el entrenamiento debido a problemas con la captura de datos.")
