# ============================================================
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ============================================================

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Audio, display

# ============================================================
# 1. DATASET QUICKDRAW (SKETCHRNN)
# ============================================================

# Descarga del dataset QuickDraw en formato TFRecords
url_datos = "http://download.tensorflow.org/data/"
nombre_archivo = "quickdraw_tutorial_dataset_v1.tar.gz"

ruta_descarga = tf.keras.utils.get_file(
    nombre_archivo,
    url_datos + nombre_archivo,
    cache_dir=".",
    extract=True
)

# Localización de la carpeta de datos
if "_extracted" in ruta_descarga:
    carpeta_quickdraw = Path(ruta_descarga)
else:
    carpeta_quickdraw = Path(ruta_descarga).parent

# Archivos de entrenamiento y evaluación
archivos_entrenamiento = sorted(
    [str(p) for p in carpeta_quickdraw.glob("training.tfrecord-*")]
)

archivos_evaluacion = sorted(
    [str(p) for p in carpeta_quickdraw.glob("eval.tfrecord-*")]
)

# Carga de nombres de clases
with open(carpeta_quickdraw / "training.tfrecord.classes") as f:
    clases = [linea.strip().lower() for linea in f.readlines()]

numero_clases = len(clases)

# ============================================================
# FUNCIÓN PARA PARSEAR LOS TFRECORDS
# ============================================================

def parsear_ejemplos(lote_datos):
    """
    Convierte un batch de TFRecords en sketches, longitudes y etiquetas
    """
    descripcion = {
        "ink": tf.io.VarLenFeature(tf.float32),
        "shape": tf.io.FixedLenFeature([2], tf.int64),
        "class_index": tf.io.FixedLenFeature([1], tf.int64)
    }

    ejemplos = tf.io.parse_example(lote_datos, descripcion)
    dibujos_planos = tf.sparse.to_dense(ejemplos["ink"])
    dibujos = tf.reshape(dibujos_planos, [tf.size(lote_datos), -1, 3])
    longitudes = ejemplos["shape"][:, 0]
    etiquetas = ejemplos["class_index"][:, 0]

    return dibujos, longitudes, etiquetas


def crear_dataset_quickdraw(
    rutas,
    tam_lote=32,
    buffer_shuffle=None,
    hilos_parseo=5,
    hilos_lectura=5,
    cache=False
):
    """
    Construye un tf.data.Dataset para QuickDraw
    """
    ds = tf.data.TFRecordDataset(
        rutas,
        num_parallel_reads=hilos_lectura
    )

    if cache:
        ds = ds.cache()

    if buffer_shuffle:
        ds = ds.shuffle(buffer_shuffle)

    ds = ds.batch(tam_lote)
    ds = ds.map(parsear_ejemplos, num_parallel_calls=hilos_parseo)

    return ds.prefetch(1)


# Creación de datasets
dataset_entrenamiento = crear_dataset_quickdraw(
    archivos_entrenamiento,
    buffer_shuffle=10000
)

dataset_validacion = crear_dataset_quickdraw(
    archivos_evaluacion[:5]
)

dataset_prueba = crear_dataset_quickdraw(
    archivos_evaluacion[5:]
)

# ============================================================
# VISUALIZACIÓN DE SKETCHES
# ============================================================

def dibujar_sketch(sketch, etiqueta=None):
    """
    Dibuja un sketch individual
    """
    origen = np.array([[0., 0., 0.]])
    sketch = np.r_[origen, sketch]

    indices_fin_trazo = np.argwhere(sketch[:, -1] == 1.)[:, 0]
    coordenadas = sketch[:, :2].cumsum(axis=0)
    trazos = np.split(coordenadas, indices_fin_trazo + 1)

    if etiqueta is not None:
        plt.title(clases[etiqueta.numpy()])
    else:
        plt.title("Adivina la clase")

    plt.plot(coordenadas[:, 0], -coordenadas[:, 1], "y:")
    for trazo in trazos:
        plt.plot(trazo[:, 0], -trazo[:, 1], ".-")

    plt.axis("off")


def dibujar_varios_sketches(sketches, longitudes, etiquetas):
    """
    Dibuja varios sketches en una cuadrícula
    """
    n = len(sketches)
    columnas = 4
    filas = (n - 1) // columnas + 1

    plt.figure(figsize=(columnas * 3, filas * 3.5))

    for i, s, l, e in zip(range(n), sketches, longitudes, etiquetas):
        plt.subplot(filas, columnas, i + 1)
        dibujar_sketch(s[:l], e)

    plt.show()

# ============================================================
# PREPROCESADO: RECORTE DE SKETCHES LARGOS
# ============================================================

def recortar_sketches_largos(dataset, longitud_maxima=100):
    """
    Recorta sketches a una longitud máxima fija
    """
    return dataset.map(
        lambda x, _, y: (x[:, :longitud_maxima], y)
    )


dataset_entrenamiento_recortado = recortar_sketches_largos(dataset_entrenamiento)
dataset_validacion_recortado = recortar_sketches_largos(dataset_validacion)
dataset_prueba_recortado = recortar_sketches_largos(dataset_prueba)

# ============================================================
# MODELO DE CLASIFICACIÓN SKETCHRNN
# ============================================================

modelo_sketch = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 5, strides=2, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(64, 5, strides=2, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(128, 3, strides=2, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(numero_clases, activation="softmax")
])

optimizador = tf.keras.optimizers.SGD(
    learning_rate=1e-2,
    clipnorm=1.0
)

modelo_sketch.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizador,
    metrics=["accuracy", "sparse_top_k_categorical_accuracy"]
)

# Entrenamiento
modelo_sketch.fit(
    dataset_entrenamiento_recortado,
    epochs=2,
    validation_data=dataset_validacion_recortado
)

# Guardado del modelo
modelo_sketch.save("modelo_sketchrnn.keras")

# ============================================================
# 2. DATASET BACH CHORALES
# ============================================================

ruta_bach = tf.keras.utils.get_file(
    "jsb_chorales.tgz",
    "https://github.com/ageron/data/raw/main/jsb_chorales.tgz",
    cache_dir=".",
    extract=True
)

if "_extracted" in ruta_bach:
    carpeta_bach = Path(ruta_bach) / "jsb_chorales"
else:
    carpeta_bach = Path(ruta_bach).with_name("jsb_chorales")

archivos_train = sorted(carpeta_bach.glob("train/chorale_*.csv"))
archivos_valid = sorted(carpeta_bach.glob("valid/chorale_*.csv"))
archivos_test = sorted(carpeta_bach.glob("test/chorale_*.csv"))

def cargar_corales(rutas):
    """
    Carga los corales Bach desde CSV
    """
    return [pd.read_csv(r).values.tolist() for r in rutas]

corales_train = cargar_corales(archivos_train)
corales_valid = cargar_corales(archivos_valid)
corales_test = cargar_corales(archivos_test)

# ============================================================
# SÍNTESIS DE AUDIO
# ============================================================

def notas_a_frecuencias(notas):
    return 2 ** ((np.array(notas) - 69) / 12) * 440

def frecuencias_a_muestras(frecuencias, tempo, sample_rate):
    duracion = 60 / tempo
    frecuencias = (duracion * frecuencias).round() / duracion
    n_muestras = int(duracion * sample_rate)
    tiempo = np.linspace(0, duracion, n_muestras)

    ondas = np.sin(2 * np.pi * frecuencias.reshape(-1, 1) * tiempo)
    ondas *= (frecuencias > 9).reshape(-1, 1)

    return ondas.reshape(-1)

def acordes_a_audio(acordes, tempo, sample_rate):
    freqs = notas_a_frecuencias(acordes)
    freqs = np.r_[freqs, freqs[-1:]]
    mezcla = np.mean(
        [frecuencias_a_muestras(m, tempo, sample_rate) for m in freqs.T],
        axis=0
    )

    fade = np.linspace(1, 0, sample_rate * 60 // tempo) ** 2
    mezcla[-len(fade):] *= fade

    return mezcla

def reproducir_acordes(acordes, tempo=160, amplitud=0.1, sample_rate=44100):
    audio = amplitud * acordes_a_audio(acordes, tempo, sample_rate)
    return display(Audio(audio, rate=sample_rate))
