# Importamos librerías necesarias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Cargamos el dataset MNIST
(X_entrenamiento_completo, y_entrenamiento_completo), (X_prueba, y_prueba) = tf.keras.datasets.mnist.load_data()

# El set de entrenamiento contiene 60,000 imágenes en escala de grises de 28x28 píxeles
print(X_entrenamiento_completo.shape)  # (60000, 28, 28)

# Cada intensidad de píxel está representada como un byte (0 a 255)
print(X_entrenamiento_completo.dtype)  # dtype('uint8')

# Separamos el set de validación (5,000) y el set de entrenamiento más pequeño
# Escalamos los píxeles a valores entre 0 y 1
X_validacion, X_entrenamiento = X_entrenamiento_completo[:5000] / 255., X_entrenamiento_completo[5000:] / 255.
y_validacion, y_entrenamiento = y_entrenamiento_completo[:5000], y_entrenamiento_completo[5000:]
X_prueba = X_prueba / 255.

# Mostramos una imagen usando Matplotlib con mapa de colores 'binary'
plt.imshow(X_entrenamiento[0], cmap="binary")
plt.axis('off')
plt.show()

# Las etiquetas son los IDs de las clases (0 a 9)
print(y_entrenamiento[:10])

# Tamaño de los sets de validación y prueba
print(X_validacion.shape)  # (5000, 28, 28)
print(X_prueba.shape)      # (10000, 28, 28)

# Visualizamos varias imágenes del dataset
n_filas = 4
n_columnas = 10
plt.figure(figsize=(n_columnas * 1.2, n_filas * 1.2))
for fila in range(n_filas):
    for columna in range(n_columnas):
        indice = n_columnas * fila + columna
        plt.subplot(n_filas, n_columnas, indice + 1)
        plt.imshow(X_entrenamiento[indice], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(y_entrenamiento[indice])
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# Construimos una red densa simple y buscamos la tasa de aprendizaje óptima
# Creamos un callback que aumente exponencialmente la tasa de aprendizaje y guarde la pérdida
K = tf.keras.backend

class TasaAprendizajeExponencial(tf.keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.tasas = []
        self.perdidas = []

    def on_batch_end(self, batch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy() * self.factor
        self.model.optimizer.learning_rate = lr
        self.tasas.append(lr)
        self.perdidas.append(logs["loss"])

# Reiniciamos sesión y fijamos semillas
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# Definimos el modelo
modelo = tf.keras.Sequential([
    layers.Flatten(input_shape=[28, 28]),
    layers.Dense(300, activation="relu"),
    layers.Dense(100, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Empezamos con una tasa de aprendizaje pequeña y la incrementamos un 0.5% por iteración
optimizador = keras.optimizers.SGD(learning_rate=1e-3)
modelo.compile(loss="sparse_categorical_crossentropy", optimizer=optimizador,
               metrics=["accuracy"])
callback_lr_exponencial = TasaAprendizajeExponencial(factor=1.005)

# Entrenamos el modelo por 1 época para evaluar el efecto de la tasa de aprendizaje
historial = modelo.fit(X_entrenamiento, y_entrenamiento, epochs=1,
                       validation_data=(X_validacion, y_validacion),
                       callbacks=[callback_lr_exponencial])

# Graficamos la pérdida en función de la tasa de aprendizaje
plt.plot(callback_lr_exponencial.tasas, callback_lr_exponencial.perdidas)
plt.gca().set_xscale('log')
plt.hlines(min(callback_lr_exponencial.perdidas), min(callback_lr_exponencial.tasas), max(callback_lr_exponencial.tasas))
plt.axis([min(callback_lr_exponencial.tasas), max(callback_lr_exponencial.tasas), 0, callback_lr_exponencial.perdidas[0]])
plt.grid()
plt.xlabel("Tasa de aprendizaje")
plt.ylabel("Pérdida")

# Reiniciamos sesión y volvemos a definir el modelo con la tasa óptima (0.3)
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

modelo = keras.Sequential([
    layers.Flatten(input_shape=[28, 28]),
    layers.Dense(300, activation="relu"),
    layers.Dense(100, activation="relu"),
    layers.Dense(10, activation="softmax")
])

optimizador = keras.optimizers.SGD(learning_rate=3e-1)
modelo.compile(loss="sparse_categorical_crossentropy", optimizer=optimizador,
               metrics=["accuracy"])

# Definimos directorio para logs y callbacks
indice_corrida = 1
directorio_logs = Path() / "mis_logs_mnist" / f"corrida_{indice_corrida:03d}"

early_stopping_cb = EarlyStopping(patience=20)
checkpoint_cb = ModelCheckpoint("mi_modelo_mnist.keras", save_best_only=True)
tensorboard_cb = TensorBoard(directorio_logs)

# Entrenamos el modelo con todos los callbacks
historial = modelo.fit(X_entrenamiento, y_entrenamiento, epochs=100,
                       validation_data=(X_validacion, y_validacion),
                       callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])

# Cargamos el mejor modelo guardado y evaluamos en el set de prueba
modelo = keras.models.load_model("mi_modelo_mnist.keras")
modelo.evaluate(X_prueba, y_prueba)  # Obtenemos precisión >98%

# Visualizamos curvas de aprendizaje usando TensorBoard
%tensorboard --logdir=./mis_logs_mnist
