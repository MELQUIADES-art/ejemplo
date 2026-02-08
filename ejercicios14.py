# ============================================================
# Clasificación de dígitos escritos a mano (MNIST)
# CNN de alta precisión con TensorFlow / Keras
# ============================================================

# ----------------------------
# Importación de librerías
# ----------------------------
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------
# Limpieza de sesión y semillas
# (para reproducibilidad)
# ----------------------------
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# ----------------------------
# Carga del dataset MNIST
# ----------------------------
# El dataset contiene imágenes de dígitos escritos a mano (0–9)
# Tamaño de cada imagen: 28x28 píxeles en escala de grises
(datos_entrenamiento, etiquetas_entrenamiento), (datos_prueba, etiquetas_prueba) = keras.datasets.mnist.load_data()

# ----------------------------
# Normalización de los datos
# ----------------------------
# Se escalan los valores de los píxeles al rango [0, 1]
datos_entrenamiento = datos_entrenamiento / 255.0
datos_prueba = datos_prueba / 255.0

# ----------------------------
# Separación en entrenamiento y validación
# ----------------------------
# Se reservan las últimas 5000 imágenes para validación
imagenes_train = datos_entrenamiento[:-5000]
imagenes_valid = datos_entrenamiento[-5000:]

etiquetas_train = etiquetas_entrenamiento[:-5000]
etiquetas_valid = etiquetas_entrenamiento[-5000:]

# ----------------------------
# Ajuste de dimensiones
# ----------------------------
# Se añade el canal (1) para compatibilidad con Conv2D
imagenes_train = imagenes_train[..., np.newaxis]
imagenes_valid = imagenes_valid[..., np.newaxis]
imagenes_test = datos_prueba[..., np.newaxis]

# ----------------------------
# Construcción del modelo CNN
# ----------------------------
modelo_cnn = keras.Sequential([

    # Primera capa convolucional
    layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        input_shape=(28, 28, 1)
    ),

    # Segunda capa convolucional
    layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal"
    ),

    # Capa de pooling para reducir dimensionalidad
    layers.MaxPooling2D(),

    # Conversión de mapas de características a vector
    layers.Flatten(),

    # Regularización con Dropout (25%)
    layers.Dropout(0.25),

    # Capa densa totalmente conectada
    layers.Dense(
        units=128,
        activation="relu",
        kernel_initializer="he_normal"
    ),

    # Dropout más agresivo (50%)
    layers.Dropout(0.5),

    # Capa de salida con Softmax (10 clases)
    layers.Dense(10, activation="softmax")
])

# ----------------------------
# Compilación del modelo
# ----------------------------
modelo_cnn.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="nadam",
    metrics=["accuracy"]
)

# ----------------------------
# Entrenamiento del modelo
# ----------------------------
historial = modelo_cnn.fit(
    imagenes_train,
    etiquetas_train,
    epochs=10,
    validation_data=(imagenes_valid, etiquetas_valid)
)

# ----------------------------
# Evaluación en el conjunto de prueba
# ----------------------------
resultado_prueba = modelo_cnn.evaluate(imagenes_test, etiquetas_prueba)

print("\nResultados en el conjunto de prueba:")
print(f"Pérdida: {resultado_prueba[0]:.4f}")
print(f"Precisión: {resultado_prueba[1]*100:.2f}%")
