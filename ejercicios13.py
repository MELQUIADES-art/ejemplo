# Importaciones necesarias
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import ExitStack
import tensorflow_datasets as tfds
from tensorflow.train import Example, Features, Feature, BytesList, Int64List
from datetime import datetime

# ========================================
# 1. Cargar Fashion MNIST y dividir conjuntos
# ========================================
(X_entrenamiento_completo, y_entrenamiento_completo), (X_prueba, y_prueba) = tf.keras.datasets.fashion_mnist.load_data()

X_validacion, X_entrenamiento = X_entrenamiento_completo[:5000], X_entrenamiento_completo[5000:]
y_validacion, y_entrenamiento = y_entrenamiento_completo[:5000], y_entrenamiento_completo[5000:]

# ========================================
# 2. Crear datasets de TensorFlow y mezclar entrenamiento
# ========================================
tf.random.set_seed(42)
dataset_entrenamiento = tf.data.Dataset.from_tensor_slices((X_entrenamiento, y_entrenamiento))
dataset_entrenamiento = dataset_entrenamiento.shuffle(len(X_entrenamiento), seed=42)
dataset_validacion = tf.data.Dataset.from_tensor_slices((X_validacion, y_validacion))
dataset_prueba = tf.data.Dataset.from_tensor_slices((X_prueba, y_prueba))

# ========================================
# 3. Función para crear Example protobuf
# ========================================
def crear_ejemplo(imagen, etiqueta):
    """
    Convierte una imagen y su etiqueta en un Example de TensorFlow para TFRecord
    """
    imagen_serializada = tf.io.serialize_tensor(imagen)
    return Example(
        features=Features(
            feature={
                "imagen": Feature(bytes_list=BytesList(value=[imagen_serializada.numpy()])),
                "etiqueta": Feature(int64_list=Int64List(value=[etiqueta])),
            }
        )
    )

# ========================================
# 4. Guardar datasets en múltiples archivos TFRecord
# ========================================
def guardar_tfrecords(nombre, dataset, num_archivos=10):
    """
    Guarda un dataset en múltiples TFRecord en modo round-robin
    """
    rutas = ["{}.tfrecord-{:05d}-of-{:05d}".format(nombre, i, num_archivos)
             for i in range(num_archivos)]
    with ExitStack() as stack:
        escritores = [stack.enter_context(tf.io.TFRecordWriter(ruta)) for ruta in rutas]
        for idx, (imagen, etiqueta) in dataset.enumerate():
            shard = idx % num_archivos
            ejemplo = crear_ejemplo(imagen, etiqueta)
            escritores[shard].write(ejemplo.SerializeToString())
    return rutas

rutas_entrenamiento = guardar_tfrecords("fashion_mnist_entrenamiento", dataset_entrenamiento)
rutas_validacion = guardar_tfrecords("fashion_mnist_validacion", dataset_validacion)
rutas_prueba = guardar_tfrecords("fashion_mnist_prueba", dataset_prueba)

# ========================================
# 5. Crear pipeline eficiente con tf.data
# ========================================
def preprocesar_tfrecord(tfrecord):
    """
    Parsea un TFRecord y devuelve la imagen y la etiqueta
    """
    descripciones = {
        "imagen": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "etiqueta": tf.io.FixedLenFeature([], tf.int64, default_value=-1)
    }
    ejemplo = tf.io.parse_single_example(tfrecord, descripciones)
    imagen = tf.io.parse_tensor(ejemplo["imagen"], out_type=tf.uint8)
    imagen = tf.reshape(imagen, shape=[28, 28])
    return imagen, ejemplo["etiqueta"]

def crear_dataset_tfrecord(rutas, n_hilos_lectura=5, buffer_shuffle=None,
                           n_hilos_parseo=5, tam_batch=32, cache=True):
    """
    Crea un dataset de TFRecord con lectura paralela, batch y prefetch
    """
    dataset = tf.data.TFRecordDataset(rutas, num_parallel_reads=n_hilos_lectura)
    if cache:
        dataset = dataset.cache()
    if buffer_shuffle:
        dataset = dataset.shuffle(buffer_shuffle)
    dataset = dataset.map(preprocesar_tfrecord, num_parallel_calls=n_hilos_parseo)
    dataset = dataset.batch(tam_batch)
    return dataset.prefetch(1)

dataset_entrenamiento_tf = crear_dataset_tfrecord(rutas_entrenamiento, buffer_shuffle=60000)
dataset_validacion_tf = crear_dataset_tfrecord(rutas_validacion)
dataset_prueba_tf = crear_dataset_tfrecord(rutas_prueba)

# ========================================
# 6. Visualizar algunas imágenes
# ========================================
for imagen_batch, etiqueta_batch in dataset_entrenamiento_tf.take(1):
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(imagen_batch[i].numpy(), cmap="binary")
        plt.axis("off")
        plt.title(str(etiqueta_batch[i].numpy()))
plt.show()

# ========================================
# 7. Modelo Keras para Fashion MNIST
# ========================================
tf.random.set_seed(42)

normalizacion = tf.keras.layers.Normalization(input_shape=[28, 28])

muestras_imagenes = dataset_entrenamiento_tf.take(100).map(lambda x, y: x)
imagenes_numpy = np.concatenate(list(muestras_imagenes.as_numpy_iterator()), axis=0).astype(np.float32)
normalizacion.adapt(imagenes_numpy)

modelo_fashion = tf.keras.Sequential([
    normalizacion,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

modelo_fashion.compile(loss="sparse_categorical_crossentropy",
                       optimizer="nadam", metrics=["accuracy"])

# Configurar TensorBoard
logs = Path() / "logs_fashion_mnist" / datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch=10)

# Entrenar modelo
modelo_fashion.fit(dataset_entrenamiento_tf, epochs=5,
                   validation_data=dataset_validacion_tf,
                   callbacks=[tensorboard_cb])

# ========================================
# 8. Descargar y preparar IMDB
# ========================================
root_imdb = "https://ai.stanford.edu/~amaas/data/sentiment/"
archivo_imdb = "aclImdb_v1.tar.gz"
ruta_imdb = tf.keras.utils.get_file(archivo_imdb, root_imdb + archivo_imdb, extract=True, cache_dir=".")
if "_extracted" in ruta_imdb:
    path_imdb = Path(ruta_imdb) / "aclImdb"
else:
    path_imdb = Path(ruta_imdb).with_name("aclImdb")

# Función para mostrar estructura de carpetas
def mostrar_estructura(ruta, nivel=0, indent=4, max_archivos=3):
    if nivel == 0:
        print(f"{ruta}/")
        nivel += 1
    subrutas = sorted(ruta.iterdir())
    carpetas = [p for p in subrutas if p.is_dir()]
    archivos = [p for p in subrutas if p not in carpetas]
    indent_str = " " * indent * nivel
    for carpeta in carpetas:
        print(f"{indent_str}{carpeta.name}/")
        mostrar_estructura(carpeta, nivel + 1, indent)
    for archivo in archivos[:max_archivos]:
        print(f"{indent_str}{archivo.name}")
    if len(archivos) > max_archivos:
        print(f"{indent_str}...")

mostrar_estructura(path_imdb)

# ========================================
# 9. Cargar rutas de reviews
# ========================================
def rutas_reviews(dirpath):
    return [str(path) for path in dirpath.glob("*.txt")]

train_pos = rutas_reviews(path_imdb / "train" / "pos")
train_neg = rutas_reviews(path_imdb / "train" / "neg")
test_valid_pos = rutas_reviews(path_imdb / "test" / "pos")
test_valid_neg = rutas_reviews(path_imdb / "test" / "neg")

# Dividir test en validación y prueba
np.random.shuffle(test_valid_pos)
np.random.shuffle(test_valid_neg)

test_pos = test_valid_pos[:5000]
test_neg = test_valid_neg[:5000]
valid_pos = test_valid_pos[5000:]
valid_neg = test_valid_neg[5000:]

# ========================================
# 10. Crear datasets tf.data para IMDB
# ========================================
batch_size = 32

def imdb_dataset(filepaths_positive, filepaths_negative, n_hilos=5):
    dataset_neg = tf.data.TextLineDataset(filepaths_negative, num_parallel_reads=n_hilos)
    dataset_neg = dataset_neg.map(lambda review: (review, 0))
    dataset_pos = tf.data.TextLineDataset(filepaths_positive, num_parallel_reads=n_hilos)
    dataset_pos = dataset_pos.map(lambda review: (review, 1))
    return tf.data.Dataset.concatenate(dataset_pos, dataset_neg)

dataset_entrenamiento_imdb = imdb_dataset(train_pos, train_neg).shuffle(25000, seed=42).batch(batch_size).prefetch(1)
dataset_validacion_imdb = imdb_dataset(valid_pos, valid_neg).batch(batch_size).prefetch(1)
dataset_prueba_imdb = imdb_dataset(test_pos, test_neg).batch(batch_size).prefetch(1)

# ========================================
# 11. Modelo de clasificación binaria con TF-IDF
# ========================================
max_tokens = 1000
reviews_muestra = dataset_entrenamiento_imdb.map(lambda review, label: review)
text_vectorization = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_mode="tf_idf")
text_vectorization.adapt(reviews_muestra)

modelo_imdb_tfidf = tf.keras.Sequential([
    text_vectorization,
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

modelo_imdb_tfidf.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
modelo_imdb_tfidf.fit(dataset_entrenamiento_imdb, epochs=5, validation_data=dataset_validacion_imdb)

# ========================================
# 12. Modelo con Embedding y rescaling
# ========================================
embedding_size = 20
text_vectorization_int = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_mode="int")
text_vectorization_int.adapt(reviews_muestra)

def calcular_mean_embedding(inputs):
    """
    Calcula la media de embeddings y multiplica por la raíz del número de palabras
    """
    no_pad = tf.math.count_nonzero(inputs, axis=-1)
    n_palabras = tf.math.count_nonzero(no_pad, axis=-1, keepdims=True)
    sqrt_n_palabras = tf.math.sqrt(tf.cast(n_palabras, tf.float32))
    return tf.reduce_sum(inputs, axis=1) / sqrt_n_palabras

modelo_imdb_embedding = tf.keras.Sequential([
    text_vectorization_int,
    tf.keras.layers.Embedding(input_dim=max_tokens, output_dim=embedding_size, mask_zero=True),
    tf.keras.layers.Lambda(calcular_mean_embedding),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

modelo_imdb_embedding.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
modelo_imdb_embedding.fit(dataset_entrenamiento_imdb, epochs=5, validation_data=dataset_validacion_imdb)

# ========================================
# 13. Cargar IMDB fácilmente con TFDS
# ========================================
datasets_tfds = tfds.load(name="imdb_reviews")
dataset_entrenamiento_tfds, dataset_prueba_tfds = datasets_tfds["train"], datasets_tfds["test"]

for ejemplo in dataset_entrenamiento_tfds.take(1):
    print(ejemplo["text"])
    print(ejemplo["label"])
