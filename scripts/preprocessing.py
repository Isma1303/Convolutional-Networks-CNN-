import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def cargar_y_preprocesar_datos(ruta_datos, tamano_lote, tamano_imagen):
    """
    Carga y preprocesa imágenes usando ImageDataGenerator.

    Args:
        ruta_datos (str): Ruta al directorio de datos (train o validation).
        tamano_lote (int): Tamaño del lote para el generador.
        tamano_imagen (tuple): Tamaño de las imágenes (ancho, alto).

    Returns:
        tf.data.Dataset: Conjunto de datos preprocesado.
    """

    generador_imagenes = ImageDataGenerator(rescale=1./255) # Reescalar los valores de píxeles a [0, 1]

    conjunto_datos = generador_imagenes.flow_from_directory(
        ruta_datos,
        target_size=tamano_imagen,
        batch_size=tamano_lote,
        class_mode='categorical' # Para clasificación multiclase
    )

    return conjunto_datos