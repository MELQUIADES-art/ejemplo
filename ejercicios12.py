# ============================================================
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ============================================================

import numpy as np
import tensorflow as tf
from collections import OrderedDict
from tqdm import trange

# ============================================================
# FUNCIÓN AUXILIAR PARA EXTRAER BATCHES ALEATORIOS
# ============================================================

def obtener_batch_aleatorio(X, y, tamaño_batch=32):
    """
    Devuelve un batch aleatorio de los datos de entrenamiento.
    """
    indices = np.random.randint(0, len(X), tamaño_batch)
    return X[indices], y[indices]

# ============================================================
# CAPA PERSONALIZADA: NORMALIZACIÓN POR CAPAS (LAYER NORM)
# ============================================================

class NormalizacionPorCapas(tf.keras.layers.Layer):
    """
    Implementación personalizada de Layer Normalization.
    """

    def __init__(self, epsilon=0.001, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, forma_entrada):
        # Parámetro de escala (α), inicializado en 1
        self.alpha = self.add_weight(
            name="alpha",
            shape=forma_entrada[-1:],
            initializer="ones",
            dtype=tf.float32
        )
        # Parámetro de desplazamiento (β), inicializado en 0
        self.beta = self.add_weight(
            name="beta",
            shape=forma_entrada[-1:],
            initializer="zeros",
            dtype=tf.float32
        )

    def call(self, entradas):
        # Cálculo de la media y la varianza por instancia
        media, varianza = tf.nn.moments(entradas, axes=-1, keepdims=True)
        desviacion = tf.sqrt(varianza + self.epsilon)

        # Fórmula de Layer Normalization
        return self.alpha * (entradas - media) / desviacion + self.beta

    def get_config(self):
        config_base = super().get_config()
        return {**config_base, "epsilon": self.epsilon}

# ============================================================
# COMPARACIÓN CON LayerNormalization DE KERAS
# ============================================================

# Datos de ejemplo
X_entrenamiento = np.random.rand(100, 20).astype(np.float32)

capa_personalizada = NormalizacionPorCapas()
capa_keras = tf.keras.layers.LayerNormalization()

error_medio = tf.reduce_mean(
    tf.keras.losses.MeanAbsoluteError()(
        capa_keras(X_entrenamiento),
        capa_personalizada(X_entrenamiento)
    )
)

print("Error medio absoluto:", error_medio.numpy())

# ============================================================
# CARGA Y PREPROCESAMIENTO DEL DATASET FASHION MNIST
# ============================================================

(X_total, y_total), (X_prueba, y_prueba) = tf.keras.datasets.fashion_mnist.load_data()

X_total = X_total.astype(np.float32) / 255.0
X_prueba = X_prueba.astype(np.float32) / 255.0

X_validacion = X_total[:5000]
X_entrenamiento = X_total[5000:]

y_validacion = y_total[:5000]
y_entrenamiento = y_total[5000:]

# ============================================================
# MODELO BASE PARA ENTRENAMIENTO PERSONALIZADO
# ============================================================

tf.keras.utils.set_random_seed(42)

modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

# ============================================================
# PARÁMETROS DE ENTRENAMIENTO
# ============================================================

epocas = 5
tamaño_batch = 32
pasos_por_epoca = len(X_entrenamiento) // tamaño_batch

optimizador = tf.keras.optimizers.Nadam(learning_rate=0.01)
funcion_perdida = tf.keras.losses.sparse_categorical_crossentropy

media_perdida = tf.keras.metrics.Mean()
metricas = [tf.keras.metrics.SparseCategoricalAccuracy()]

# ============================================================
# BUCLE DE ENTRENAMIENTO PERSONALIZADO
# ============================================================

with trange(1, epocas + 1, desc="Entrenamiento completo") as barra_epocas:
    for epoca in barra_epocas:
        with trange(1, pasos_por_epoca + 1, desc=f"Época {epoca}/{epocas}") as barra_pasos:
            for paso in barra_pasos:
                X_batch, y_batch = obtener_batch_aleatorio(
                    X_entrenamiento, y_entrenamiento, tamaño_batch
                )

                with tf.GradientTape() as cinta:
                    predicciones = modelo(X_batch)
                    perdida_principal = tf.reduce_mean(
                        funcion_perdida(y_batch, predicciones)
                    )
                    perdida_total = tf.add_n([perdida_principal] + modelo.losses)

                gradientes = cinta.gradient(
                    perdida_total, modelo.trainable_variables
                )
                optimizador.apply_gradients(
                    zip(gradientes, modelo.trainable_variables)
                )

                estado = OrderedDict()
                media_perdida(perdida_total)
                estado["pérdida"] = media_perdida.result().numpy()

                for metrica in metricas:
                    metrica(y_batch, predicciones)
                    estado[metrica.name] = metrica.result().numpy()

                barra_pasos.set_postfix(estado)

            # Evaluación en validación
            pred_validacion = modelo(X_validacion)
            estado["val_pérdida"] = np.mean(
                funcion_perdida(y_validacion, pred_validacion)
            )
            estado["val_exactitud"] = np.mean(
                tf.keras.metrics.sparse_categorical_accuracy(
                    tf.constant(y_validacion, dtype=tf.float32),
                    pred_validacion
                )
            )
            barra_pasos.set_postfix(estado)

        for metrica in [media_perdida] + metricas:
            metrica.reset_state()

# ============================================================
# ENTRENAMIENTO CON DOS OPTIMIZADORES (CAPAS INFERIORES Y SUPERIORES)
# ============================================================

capas_inferiores = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="relu"),
])

capas_superiores = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="softmax"),
])

modelo_doble_opt = tf.keras.Sequential([
    capas_inferiores,
    capas_superiores
])

optimizador_inferior = tf.keras.optimizers.SGD(learning_rate=1e-4)
optimizador_superior = tf.keras.optimizers.Nadam(learning_rate=1e-3)

# ============================================================
# BUCLE DE ENTRENAMIENTO CON DOS OPTIMIZADORES
# ============================================================

with trange(1, epocas + 1, desc="Entrenamiento con doble optimizador") as barra_epocas:
    for epoca in barra_epocas:
        with trange(1, pasos_por_epoca + 1, desc=f"Época {epoca}/{epocas}") as barra_pasos:
            for paso in barra_pasos:
                X_batch, y_batch = obtener_batch_aleatorio(
                    X_entrenamiento, y_entrenamiento, tamaño_batch
                )

                with tf.GradientTape(persistent=True) as cinta:
                    predicciones = modelo_doble_opt(X_batch)
                    perdida = tf.reduce_mean(
                        funcion_perdida(y_batch, predicciones)
                    )

                for capas, optimizador in [
                    (capas_inferiores, optimizador_inferior),
                    (capas_superiores, optimizador_superior)
                ]:
                    gradientes = cinta.gradient(
                        perdida, capas.trainable_variables
                    )
                    optimizador.apply_gradients(
                        zip(gradientes, capas.trainable_variables)
                    )

                del cinta
