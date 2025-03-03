import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def cargar_y_preprocesar_datos(ruta_datos, tamano_lote, tamano_imagen):
    """Carga y preprocesa imágenes usando ImageDataGenerator."""
    generador_imagenes = ImageDataGenerator(rescale=1./255)
    conjunto_datos = generador_imagenes.flow_from_directory(
        ruta_datos,
        target_size=tamano_imagen,
        batch_size=tamano_lote,
        class_mode='categorical',
        shuffle=False  # Importante para la matriz de confusión
    )
    return conjunto_datos

def evaluar_modelo(ruta_modelo, ruta_datos_prueba, tamano_lote, tamano_imagen):
    """Evalúa un modelo CNN."""
    # Cargar el modelo entrenado
    modelo = tf.keras.models.load_model(ruta_modelo)

    # Cargar y preprocesar el conjunto de datos de prueba
    conjunto_datos_prueba = cargar_y_preprocesar_datos(ruta_datos_prueba, tamano_lote, tamano_imagen)

    # Evaluar el modelo
    evaluacion = modelo.evaluate(conjunto_datos_prueba)
    print(f"Pérdida en la prueba: {evaluacion[0]:.4f}, Precisión en la prueba: {evaluacion[1]:4f}")

    # Obtener las predicciones
    predicciones = modelo.predict(conjunto_datos_prueba)
    y_pred = np.argmax(predicciones, axis=1)

    # Obtener las etiquetas verdaderas
    y_true = conjunto_datos_prueba.classes

    # Generar el informe de clasificación
    print(classification_report(y_true, y_pred, target_names=conjunto_datos_prueba.class_indices.keys()))

    # Generar la matriz de confusión
    matriz_confusion = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=conjunto_datos_prueba.class_indices.keys(),
                yticklabels=conjunto_datos_prueba.class_indices.keys())
    plt.xlabel('Predicciones')
    plt.ylabel('Etiquetas Verdaderas')
    plt.title('Matriz de Confusión')
    plt.show()

if __name__ == "__main__":
    ruta_modelo = "models/cnn_model1.h5"  # Asegúrate de que este sea el nombre correcto de tu modelo
    ruta_datos_prueba = "data/test"  # Ajusta la ruta a tu conjunto de datos de prueba
    tamano_lote = 32
    tamano_imagen = (150, 150)

    evaluar_modelo(ruta_modelo, ruta_datos_prueba, tamano_lote, tamano_imagen)