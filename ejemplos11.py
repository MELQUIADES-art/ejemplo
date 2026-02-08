# =========================================================
# 📦 IMPORTACIONES
# =========================================================
import math
import numpy as np
from pathlib import Path
import tensorflow as tf
import keras.layers

# Fijamos semilla para reproducibilidad
tf.random.set_seed(42)

# =========================================================
# 📚 CARGA Y PREPARACIÓN DEL DATASET CIFAR-10
# =========================================================
(datos_entrenamiento_completo, etiquetas_entrenamiento_completo), (datos_prueba, etiquetas_prueba) = tf.keras.datasets.cifar10.load_data()

# Conjunto de validación (primeras 5000 imágenes)
datos_validacion = datos_entrenamiento_completo[:5000]
etiquetas_validacion = etiquetas_entrenamiento_completo[:5000]

# Resto para entrenamiento
datos_entrenamiento = datos_entrenamiento_completo[5000:]
etiquetas_entrenamiento = etiquetas_entrenamiento_completo[5000:]

# =========================================================
# 🧠 MODELO 1 — DNN PROFUNDA (He + Swish)
# =========================================================
modelo_profundo = tf.keras.Sequential()
modelo_profundo.add(tf.keras.layers.Flatten(input_shape=[32, 32, 3]))

for _ in range(20):
    modelo_profundo.add(tf.keras.layers.Dense(
        100,
        activation="swish",
        kernel_initializer="he_normal"
    ))

modelo_profundo.add(tf.keras.layers.Dense(10, activation="softmax"))

optimizador_nadam = tf.keras.optimizers.Nadam(learning_rate=5e-5)

modelo_profundo.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizador_nadam,
    metrics=["accuracy"]
)

# Callbacks
parada_temprana = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
guardar_modelo = tf.keras.callbacks.ModelCheckpoint("modelo_dnn.keras", save_best_only=True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(Path("logs_dnn"))

modelo_profundo.fit(
    datos_entrenamiento, etiquetas_entrenamiento,
    epochs=100,
    validation_data=(datos_validacion, etiquetas_validacion),
    callbacks=[parada_temprana, guardar_modelo, tensorboard_cb]
)

modelo_profundo.evaluate(datos_validacion, etiquetas_validacion)

# =========================================================
# 🧠 MODELO 2 — DNN + BATCH NORMALIZATION
# =========================================================
tf.random.set_seed(42)

modelo_bn = tf.keras.Sequential()
modelo_bn.add(tf.keras.layers.Flatten(input_shape=[32, 32, 3]))

for _ in range(20):
    modelo_bn.add(tf.keras.layers.Dense(100, kernel_initializer="he_normal"))
    modelo_bn.add(tf.keras.layers.BatchNormalization())
    modelo_bn.add(tf.keras.layers.Activation("swish"))

modelo_bn.add(tf.keras.layers.Dense(10, activation="softmax"))

modelo_bn.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Nadam(learning_rate=5e-4),
    metrics=["accuracy"]
)

modelo_bn.fit(
    datos_entrenamiento, etiquetas_entrenamiento,
    epochs=100,
    validation_data=(datos_validacion, etiquetas_validacion),
    callbacks=[parada_temprana]
)

modelo_bn.evaluate(datos_validacion, etiquetas_validacion)

# =========================================================
# 🧠 MODELO 3 — SELU (Self-Normalizing Network)
# =========================================================
tf.random.set_seed(42)

modelo_selu = tf.keras.Sequential()
modelo_selu.add(tf.keras.layers.Flatten(input_shape=[32, 32, 3]))

for _ in range(20):
    modelo_selu.add(tf.keras.layers.Dense(
        100,
        activation="selu",
        kernel_initializer="lecun_normal"
    ))

modelo_selu.add(tf.keras.layers.Dense(10, activation="softmax"))

modelo_selu.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Nadam(learning_rate=7e-4),
    metrics=["accuracy"]
)

# Normalización requerida por SELU
media = datos_entrenamiento.mean(axis=0)
desviacion = datos_entrenamiento.std(axis=0)

datos_entrenamiento_norm = (datos_entrenamiento - media) / desviacion
datos_validacion_norm = (datos_validacion - media) / desviacion
datos_prueba_norm = (datos_prueba - media) / desviacion

modelo_selu.fit(
    datos_entrenamiento_norm, etiquetas_entrenamiento,
    epochs=100,
    validation_data=(datos_validacion_norm, etiquetas_validacion),
    callbacks=[parada_temprana]
)

modelo_selu.evaluate(datos_validacion_norm, etiquetas_validacion)

# =========================================================
# 🧠 MODELO 4 — SELU + ALPHA DROPOUT
# =========================================================
tf.random.set_seed(42)

modelo_alpha = tf.keras.Sequential()
modelo_alpha.add(tf.keras.layers.Flatten(input_shape=[32, 32, 3]))

for _ in range(20):
    modelo_alpha.add(tf.keras.layers.Dense(
        100,
        activation="selu",
        kernel_initializer="lecun_normal"
    ))

modelo_alpha.add(keras.layers.AlphaDropout(rate=0.1))
modelo_alpha.add(tf.keras.layers.Dense(10, activation="softmax"))

modelo_alpha.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Nadam(learning_rate=5e-4),
    metrics=["accuracy"]
)

modelo_alpha.fit(
    datos_entrenamiento_norm, etiquetas_entrenamiento,
    epochs=100,
    validation_data=(datos_validacion_norm, etiquetas_validacion),
    callbacks=[parada_temprana]
)

modelo_alpha.evaluate(datos_validacion_norm, etiquetas_validacion)

# =========================================================
# 🎲 MC DROPOUT (INFERENCIA ESTOCÁSTICA)
# =========================================================
class AlphaDropoutMC(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

modelo_mc = tf.keras.Sequential([
    AlphaDropoutMC(capa.rate) if isinstance(capa, keras.layers.AlphaDropout) else capa
    for capa in modelo_alpha.layers
])

def predecir_mc_probabilidades(modelo, datos, repeticiones=10):
    preds = [modelo.predict(datos) for _ in range(repeticiones)]
    return np.mean(preds, axis=0)

def predecir_mc_clases(modelo, datos, repeticiones=10):
    probas = predecir_mc_probabilidades(modelo, datos, repeticiones)
    return probas.argmax(axis=1)

predicciones_mc = predecir_mc_clases(modelo_mc, datos_validacion_norm)
precision_mc = (predicciones_mc == etiquetas_validacion[:, 0]).mean()
print("Precisión con MC Dropout:", precision_mc)
