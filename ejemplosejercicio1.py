from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")  # Scikit-Learn ≥1.0.1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuración de fuentes para gráficos
plt.rc('font', size=12)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# Hacer resultados reproducibles
np.random.seed(42)

#preparar datos
from sklearn.linear_model import LinearRegression

ruta_datos = "https://github.com/ageron/data/raw/main/"
satisfaccion_vida = pd.read_csv(ruta_datos + "lifesat/lifesat.csv")

# Selección de columnas
pib_per_capita = satisfaccion_vida[["GDP per capita (USD)"]].values
satisfaccion = satisfaccion_vida[["Life satisfaction"]].values

#modelo regresion lineal
from sklearn.neighbors import KNeighborsRegressor

modelo = KNeighborsRegressor(n_neighbors=3)

# Entrenar el modelo
modelo.fit(pib_per_capita, satisfaccion)

# Predicción de ejemplo (Chipre)
nuevo_pib = np.array([[37655]])  # Ejemplo de PIB de Chipre
print(modelo.predict(nuevo_pib))  # [[6.33333333]]


ruta_dataset = Path() / "datasets" / "lifesat"
ruta_dataset.mkdir(parents=True, exist_ok=True)

for archivo in ("oecd_bli.csv", "gdp_per_capita.csv"):
    if not (ruta_dataset / archivo).is_file():
        print("Descargando", archivo)
        url = ruta_datos + "lifesat/" + archivo
        import urllib.request
        urllib.request.urlretrieve(url, ruta_dataset / archivo)

# Cargar datasets
oecd_bli = pd.read_csv(ruta_dataset / "oecd_bli.csv")
gdp_per_capita_df = pd.read_csv(ruta_dataset / "gdp_per_capita.csv")

# Filtrar PIB solo año 2020
anio_pib = 2020
columna_pib = "GDP per capita (USD)"
columna_satisfaccion = "Life satisfaction"

gdp_per_capita_df = gdp_per_capita_df[gdp_per_capita_df["Year"] == anio_pib]
gdp_per_capita_df = gdp_per_capita_df.drop(["Code", "Year"], axis=1)
gdp_per_capita_df.columns = ["Country", columna_pib]
gdp_per_capita_df.set_index("Country", inplace=True)

# Filtrar OECD BLI solo para la columna "Life satisfaction"
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

# Merge datasets
datos_paises_completos = pd.merge(left=oecd_bli, right=gdp_per_capita_df,
                                  left_index=True, right_index=True)
datos_paises_completos.sort_values(by=columna_pib, inplace=True)
datos_paises_completos = datos_paises_completos[[columna_pib, columna_satisfaccion]]


pib_min = 23_500
pib_max = 62_500

datos_paises_filtrados = datos_paises_completos[
    (datos_paises_completos[columna_pib] >= pib_min) &
    (datos_paises_completos[columna_pib] <= pib_max)
]

# Guardar datasets filtrados y completos
datos_paises_filtrados.to_csv(ruta_dataset / "lifesat.csv")
datos_paises_completos.to_csv(ruta_dataset / "lifesat_completo.csv")


datos_paises_filtrados.plot(kind='scatter', x=columna_pib, y=columna_satisfaccion,
                            figsize=(5, 3), grid=True)

min_satisfaccion = 4
max_satisfaccion = 9

posicion_texto = {
    "Turkey": (29_500, 4.2),
    "Hungary": (28_000, 6.9),
    "France": (40_000, 5),
    "New Zealand": (28_000, 8.2),
    "Australia": (50_000, 5.5),
    "United States": (59_000, 5.3),
    "Denmark": (46_000, 8.5)
}

for pais, pos_text in posicion_texto.items():
    x_data = datos_paises_filtrados[columna_pib].loc[pais]
    y_data = datos_paises_filtrados[columna_satisfaccion].loc[pais]
    pais_mostrar = "U.S." if pais == "United States" else pais
    plt.annotate(pais_mostrar, xy=(x_data, y_data),
                 xytext=pos_text, fontsize=12,
                 arrowprops=dict(facecolor='black', width=0.5,
                                 shrink=0.08, headwidth=5))
    plt.plot(x_data, y_data, "ro")

plt.axis([pib_min, pib_max, min_satisfaccion, max_satisfaccion])
guardar_figura('grafico_dispersión_pib_satisfaccion')
plt.show()


from sklearn import linear_model

X_sample = datos_paises_filtrados[[columna_pib]].values
y_sample = datos_paises_filtrados[[columna_satisfaccion]].values

regresion_lineal = linear_model.LinearRegression()
regresion_lineal.fit(X_sample, y_sample)

intercepto = regresion_lineal.intercept_[0]
pendiente = regresion_lineal.coef_.ravel()[0]
print(f"θ0={intercepto:.2f}, θ1={pendiente:.2e}")


pib_chipre = gdp_per_capita_df[columna_pib].loc["Cyprus"]
prediccion_chipre = regresion_lineal.predict([[pib_chipre]])[0,0]
print(prediccion_chipre)  # 6.30 aproximadamente


from sklearn import preprocessing, pipeline

Xfull = np.c_[datos_paises_completos[columna_pib]]
yfull = np.c_[datos_paises_completos[columna_satisfaccion]]

poly = preprocessing.PolynomialFeatures(degree=10, include_bias=False)
scaler = preprocessing.StandardScaler()
lin_reg2 = linear_model.LinearRegression()

pipeline_modelo = pipeline.Pipeline([
    ('poly', poly),
    ('scal', scaler),
    ('lin', lin_reg2)
])
pipeline_modelo.fit(Xfull, yfull)

X_plot = np.linspace(0, 115_000, 1000)
curva = pipeline_modelo.predict(X_plot[:, np.newaxis])
plt.plot(X_plot, curva)

plt.axis([0, 115_000, min_satisfaccion, max_satisfaccion])
guardar_figura('overfitting_modelo')
plt.show()
