# filepath: /Users/alejandro/Desktop/Convolutional-Networks-CNN-/models/cnn_model1.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def crear_modelo_cnn(tamano_imagen, num_clases):
    """
    Crea un modelo CNN secuencial.

    Args:
        tamano_imagen (tuple): Tamaño de las imágenes (ancho, alto).
        num_clases (int): Número de clases en el conjunto de datos.

    Returns:
        tf.keras.models.Sequential: Modelo CNN compilado.
    """

    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(tamano_imagen[0], tamano_imagen[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_clases, activation='softmax')
    ])
    modelo.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return modelo