import cv2
import torch
import clip
from PIL import Image
import numpy as np

# Cargar modelo CLIP preentrenado
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model, preprocess = clip.load("ViT-B/32", device=device)
except Exception as e:
    raise Exception(f"❌ Error al cargar el modelo CLIP: {e}")

# Definir etiquetas
labels = ["gato", "perro", "pájaro", "celular", "computadora", "vaso"]
try:
    text_inputs = clip.tokenize(labels).to(device)
except Exception as e:
    raise Exception(f"❌ Error al tokenizar las etiquetas: {e}")

# Inicializar cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("❌ No se pudo abrir la cámara")
print("✅ Cámara detectada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al leer el frame de la cámara.")
        break
    
    # Convertir la imagen para CLIP
    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_preprocessed = preprocess(img_pil).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Error al procesar la imagen: {e}")
        continue
    
    # Realizar predicción
    try:
        with torch.no_grad():
            image_features = model.encode_image(img_preprocessed)
            text_features = model.encode_text(text_inputs)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            predicted_label = labels[similarity.argmax().item()]
    except Exception as e:
        print(f"❌ Error al realizar la predicción: {e}")
        continue
    
    # Mostrar resultado en consola
    print(f"🔍 Predicción: {predicted_label}")
    
    # Mostrar resultado en la ventana
    cv2.putText(frame, f"Predicción: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Reconocimiento con CLIP", frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()