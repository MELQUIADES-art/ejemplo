# ============================================================
# IMPORTACIONES NECESARIAS
# ============================================================

import numpy as np
import tensorflow as tf
from datetime import date
import torch
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

# ============================================================
# EJERCICIO 8: GRAMÁTICAS DE REBER EMBEBIDAS
# ============================================================

# Gramática de Reber por defecto
gramatica_reber_base = [
    [("B", 1)],                  # estado 0 -> B -> estado 1
    [("T", 2), ("P", 3)],         # estado 1 -> T o P
    [("S", 2), ("X", 4)],         # estado 2 -> S o X
    [("T", 3), ("V", 5)],         # estado 3
    [("X", 3), ("S", 6)],         # estado 4
    [("P", 4), ("V", 6)],         # estado 5
    [("E", None)]                 # estado final
]

# Gramática de Reber embebida
gramatica_reber_embebida = [
    [("B", 1)],
    [("T", 2), ("P", 3)],
    [(gramatica_reber_base, 4)],
    [(gramatica_reber_base, 5)],
    [("T", 6)],
    [("P", 6)],
    [("E", None)]
]

# ------------------------------------------------------------
# Generar una cadena válida a partir de una gramática
# ------------------------------------------------------------

def generar_cadena(gramatica):
    estado = 0
    salida = []
    while estado is not None:
        indice = np.random.randint(len(gramatica[estado]))
        produccion, estado = gramatica[estado][indice]
        if isinstance(produccion, list):
            produccion = generar_cadena(produccion)
        salida.append(produccion)
    return "".join(salida)

# ------------------------------------------------------------
# Generar una cadena corrupta (no válida)
# ------------------------------------------------------------

CARACTERES_POSIBLES = "BEPSTVX"

def generar_cadena_corrupta(gramatica, caracteres=CARACTERES_POSIBLES):
    cadena_correcta = generar_cadena(gramatica)
    posicion = np.random.randint(len(cadena_correcta))
    caracter_original = cadena_correcta[posicion]
    caracter_erroneo = np.random.choice(
        sorted(set(caracteres) - set(caracter_original))
    )
    return (
        cadena_correcta[:posicion] +
        caracter_erroneo +
        cadena_correcta[posicion + 1:]
    )

# ------------------------------------------------------------
# Convertir una cadena en IDs numéricos
# ------------------------------------------------------------

def cadena_a_ids(cadena, caracteres=CARACTERES_POSIBLES):
    return [caracteres.index(c) for c in cadena]

# ------------------------------------------------------------
# Crear dataset balanceado (50% válido, 50% inválido)
# ------------------------------------------------------------

def crear_dataset_reber(tamano):
    cadenas_buenas = [
        cadena_a_ids(generar_cadena(gramatica_reber_embebida))
        for _ in range(tamano // 2)
    ]
    cadenas_malas = [
        cadena_a_ids(generar_cadena_corrupta(gramatica_reber_embebida))
        for _ in range(tamano - tamano // 2)
    ]
    todas = cadenas_buenas + cadenas_malas
    X = tf.ragged.constant(todas, ragged_rank=1)
    y = np.array([[1.]] * len(cadenas_buenas) +
                 [[0.]] * len(cadenas_malas))
    return X, y

# ============================================================
# MODELO RNN PARA CLASIFICAR CADENAS REBER
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)

X_entrenamiento, y_entrenamiento = crear_dataset_reber(10000)
X_validacion, y_validacion = crear_dataset_reber(2000)

modelo_reber = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[None], dtype=tf.int32, ragged=True),
    tf.keras.layers.Embedding(input_dim=len(CARACTERES_POSIBLES), output_dim=5),
    tf.keras.layers.GRU(30),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

optimizador = tf.keras.optimizers.SGD(
    learning_rate=0.02, momentum=0.95, nesterov=True
)

modelo_reber.compile(
    loss="binary_crossentropy",
    optimizer=optimizador,
    metrics=["accuracy"]
)

modelo_reber.fit(
    X_entrenamiento,
    y_entrenamiento,
    epochs=20,
    validation_data=(X_validacion, y_validacion)
)

# ============================================================
# EJERCICIO 9: CONVERSIÓN DE FECHAS CON ENCODER-DECODER
# ============================================================

MESES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def generar_fechas_aleatorias(n):
    minimo = date(1000, 1, 1).toordinal()
    maximo = date(9999, 12, 31).toordinal()
    ordinales = np.random.randint(maximo - minimo, size=n) + minimo
    fechas = [date.fromordinal(o) for o in ordinales]
    entradas = [MESES[f.month - 1] + " " + f.strftime("%d, %Y") for f in fechas]
    salidas = [f.isoformat() for f in fechas]
    return entradas, salidas

CARACTERES_ENTRADA = "".join(sorted(set("".join(MESES) + "0123456789, ")))
CARACTERES_SALIDA = "0123456789-"

def texto_a_ids(texto, caracteres):
    return [caracteres.index(c) for c in texto]

def preparar_textos(textos, caracteres):
    ids = [texto_a_ids(t, caracteres) for t in textos]
    X = tf.ragged.constant(ids, ragged_rank=1)
    return (X + 1).to_tensor()

def crear_dataset_fechas(n):
    x, y = generar_fechas_aleatorias(n)
    return (
        preparar_textos(x, CARACTERES_ENTRADA),
        preparar_textos(y, CARACTERES_SALIDA)
    )

X_entrenamiento, Y_entrenamiento = crear_dataset_fechas(10000)
X_validacion, Y_validacion = crear_dataset_fechas(2000)

# ============================================================
# MODELO SEQ2SEQ SIMPLE
# ============================================================

modelo_fechas = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(CARACTERES_ENTRADA) + 1, 32),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.RepeatVector(Y_entrenamiento.shape[1]),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dense(len(CARACTERES_SALIDA) + 1, activation="softmax")
])

modelo_fechas.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="nadam",
    metrics=["accuracy"]
)

modelo_fechas.fit(
    X_entrenamiento,
    Y_entrenamiento,
    epochs=20,
    validation_data=(X_validacion, Y_validacion)
)

# ============================================================
# EJERCICIO 11: GENERACIÓN DE TEXTO CON GPT (PYTORCH)
# ============================================================

modelo_gpt = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")
tokenizador_gpt = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

texto_inicio = "This royal throne of kings, this sceptred isle"
entrada_codificada = tokenizador_gpt.encode(
    texto_inicio,
    return_tensors="pt",
    add_special_tokens=False
)

salidas = modelo_gpt.generate(
    input_ids=entrada_codificada,
    do_sample=True,
    max_length=50,
    top_p=0.9,
    temperature=1.0,
    num_return_sequences=5
)

for secuencia in salidas:
    print(tokenizador_gpt.decode(secuencia))
    print("-" * 80)
