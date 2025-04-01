import serial  # Comunicación con Arduino
import time
import cv2
import torch
import clip
from PIL import Image
import numpy as np
import mysql.connector
from datetime import datetime
from connection import get_connection


import platform
import glob

def get_arduino_port():
    """Detecta el puerto del Arduino según el sistema operativo."""
    system = platform.system()

    if system == "Windows":
        return "COM3"  # Cambia esto si en tu PC el puerto es diferente
    elif system == "Darwin":  # macOS
        ports = glob.glob("/dev/tty.usbmodem*")
        return ports[0] if ports else None
    elif system == "Linux":
        ports = glob.glob("/dev/ttyUSB*")
        return ports[0] if ports else None
    else:
        return None

# Detectar el puerto de Arduino
arduino_port = get_arduino_port()

if arduino_port is None:
    raise Exception("No se encontró un puerto de Arduino.")

# Configurar conexión con Arduino
arduino = serial.Serial(arduino_port, 9600, timeout=1)
time.sleep(2)  # Esperar a que Arduino se inicialice
print(f"Conectado a Arduino en {arduino_port}")


# Cargar modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_text_descriptions_from_db():
    """Obtiene las descripciones de la base de datos"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        if conn is None:
            raise Exception("No se pudo conectar a la base de datos")
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT DISTINCT text_data FROM text_data")
        text_descriptions = [row['text_data'] for row in cursor.fetchall()]
        
        return text_descriptions if text_descriptions else ["perro", "gato", "coche"]
        
    except mysql.connector.Error as err:
        print(f"Error de base de datos: {err}")
        return ["perro", "gato", "coche"]
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

def save_prediction_to_db(predicted_label):
    """Guarda la predicción en la base de datos"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        if conn is None:
            print("No se pudo conectar a la base de datos")
            return False
        
        cursor = conn.cursor()
        query = "INSERT INTO predictions (predicted_label, created_at) VALUES (%s, %s)"
        cursor.execute(query, (predicted_label, datetime.now().strftime('%d-%m-%Y')))
        conn.commit()
        return True
        
    except mysql.connector.Error as err:
        print(f"Error al guardar predicción: {err}")
        return False
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

def main():
    text_descriptions = get_text_descriptions_from_db()
    text_inputs = clip.tokenize(text_descriptions).to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("No se pudo abrir la cámara")
    print("Cámara detectada. Presiona 'q' para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer el frame de la cámara.")
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_preprocessed = preprocess(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(img_preprocessed)
                text_features = model.encode_text(text_inputs)
                similarity = (image_features @ text_features.T).softmax(dim=-1)
                predicted_index = similarity.argmax().item()
                predicted_label = text_descriptions[predicted_index]
                save_prediction_to_db(predicted_label)

            # Enviar predicción a Arduino
            arduino.write(f"{predicted_label}\n".encode())

            # Mostrar en consola y en la ventana
            print(f"Predicción: {predicted_label}")
            cv2.putText(frame, f"Predicción: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Reconocimiento con CLIP", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arduino.close()

if __name__ == "__main__":
    main()