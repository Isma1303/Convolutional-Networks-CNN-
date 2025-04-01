import cv2
import torch
import clip
from PIL import Image
import numpy as np
import mysql.connector
from datetime import datetime
from connection import get_connection



# Cargar modelo CLIP preentrenado
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model, preprocess = clip.load("ViT-B/32", device=device)
except Exception as e:
    raise Exception(f"Error al cargar el modelo CLIP: {e}")

def get_text_descriptions_from_db():
    """Obtiene las descripciones de texto desde la base de datos"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        if conn is None:
            raise Exception("No se pudo conectar a la base de datos")
        
        cursor = conn.cursor(dictionary=True)
        
        # Obtener todas las descripciones de texto únicas
        query = "SELECT DISTINCT text_data FROM text_data"
        cursor.execute(query)
        
        text_descriptions = [row['text_data'] for row in cursor.fetchall()]
        
        if not text_descriptions:
            raise Exception("No se encontraron descripciones en la base de datos")
            
        return text_descriptions
        
    except mysql.connector.Error as err:
        print(f"Error de base de datos: {err}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

def save_prediction_to_db(predicted_label):
    """Guarda la predicción en la base de datos con marca de tiempo"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        if conn is None:
            print("No se pudo conectar a la base de datos para guardar la predicción")
            return False
        
        cursor = conn.cursor()
        
        # Obtener la fecha actual en formato DD-MM-YYYY
        current_date = datetime.now().strftime('%d-%m-%Y')
        
        # Insertar la predicción
        query = "INSERT INTO predictions (predicted_label, created_at) VALUES (%s, %s)"
        cursor.execute(query, (predicted_label, current_date))
        
        conn.commit()
        return True
        
    except mysql.connector.Error as err:
        print(f"Error al guardar predicción: {err}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

def main():
    # Obtener descripciones desde la base de datos
    text_descriptions = get_text_descriptions_from_db()
    if not text_descriptions:
        # Usar valores por defecto si falla la conexión a la base de datos
        text_descriptions = [
            "perro", "gato", "coche", "persona", "arbol",
            "casa", "bicicleta", "avion", "barco", "telefono"
        ]
        print("Usando descripciones por defecto")
    
    # Tokenizar las descripciones de texto
    text_inputs = clip.tokenize(text_descriptions).to(device)
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("No se pudo abrir la camara")
    print("Camara detectada. Presiona 'q' para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer el frame de la camara.")
                break
            
            # Convertir la imagen para CLIP
            try:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                img_preprocessed = preprocess(img_pil).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Error al procesar la imagen: {e}")
                continue
            
            # Realizar predicción
            try:
                with torch.no_grad():
                    image_features = model.encode_image(img_preprocessed)
                    text_features = model.encode_text(text_inputs)
                    similarity = (image_features @ text_features.T).softmax(dim=-1)
                    
                    # Obtener el índice de la predicción más alta
                    predicted_index = similarity.argmax().item()
                    predicted_label = text_descriptions[predicted_index]
                    
                    # Guardar la predicción en la base de datos
                    save_prediction_to_db(predicted_label)
                    
            except Exception as e:
                print(f"Error al realizar la predicción: {e}")
                continue
            
            # Mostrar resultado en consola
            print(f"Prediccion: {predicted_label}")
            
            # Mostrar resultado en la ventana
            cv2.putText(frame, f"Prediccion: {predicted_label}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Reconocimiento con CLIP", frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()