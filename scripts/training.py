import os
import tensorflow as tf
from scripts.preprocesamiento_datos import cargar_y_preprocesar_datos
from models.cnn_modelo1 import crear_modelo_cnn

def entrenar_modelo(ruta_datos_entrenamiento, ruta_datos_validacion, tamano_lote, tamano_imagen, num_epocas):
    """
    Entrena un modelo CNN.

    Args:
        ruta_datos_entrenamiento (str): Ruta al directorio de datos de entrenamiento.
        ruta_datos_validacion (str): Ruta al directorio de datos de validación.
        tamano_lote (int): Tamaño del lote.
        tamano_imagen (tuple): Tamaño de las imágenes (ancho, alto).
        num_epocas (int): Número de épocas de entrenamiento.
    """

    conjunto_datos_entrenamiento = cargar_y_preprocesar_datos(ruta_datos_entrenamiento, tamano_lote, tamano_imagen)
    conjunto_datos_validacion = cargar_y_preprocesar_datos(ruta_datos_validacion, tamano_lote, tamano_imagen)

    num_clases = len(conjunto_datos_entrenamiento.class_indices)
    modelo = crear_modelo_cnn(tamano_imagen, num_clases)

    modelo.fit(conjunto_datos_entrenamiento,
              epochs=num_epocas,
              validation_data=conjunto_datos_validacion)

if __name__ == "__main__":
    ruta_datos_entrenamiento = "data/train"
    ruta_datos_validacion = "data/validation"
    tamano_lote = 32
    tamano_imagen = (150, 150)
    num_epocas = 10

    entrenar_modelo(ruta_datos_entrenamiento, ruta_datos_validacion, tamano_lote, tamano_imagen, num_epocas)