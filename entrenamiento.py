print("Hola, este script se ejecuta en Azure ML 🚀")

for i in range(3):
    print(f"Iteración {i}")
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy import stats
from sklearn.preprocessing import RobustScaler, MinMaxScaler
datos=pd.read_excel("datos.xlsx")
datos["Extensión"]=np.array(datos["Extensión"],dtype='str')
datos["año"]=np.array(datos["año"],dtype='str')
def funcionTiempo(x):
    if (x.count('h')==0):
        if(x.count('m')==0):
            x='00:00:'+x
            x=x.replace('s','')
        else:
            x='00:'+x
            if(x.count('s')==0):
                x=x.replace('m',':')
                x=x+'00'
            else:
                x=x.replace('m ',':')
                x=x.replace('s','')
    else:
        if(x.count('m')==0):
            if(x.count('s')==0):
                x=x.replace('h',':00:00')
            else:
                x=x.replace('h ',':00:')
                x=x.replace('s','')
        else:
            if(x.count('s')==0):
                x=x.replace('h ',':')
                x=x.replace('m',':00')
            else:
                x=x.replace('h ',':')
                x=x.replace('m ',':')
                x=x.replace('s','')

    h, m, s = map(int, x.split(':'))
    x = timedelta(hours=h, minutes=m, seconds=s)
    return x
datos["Duración"]=datos["Duración"].apply(funcionTiempo)
datos["Entrantes Duración"]=datos["Entrantes Duración"].apply(funcionTiempo)
datos["Internas Duración"]=datos["Internas Duración"].apply(funcionTiempo)
datos["Salientes Duración"]=datos["Salientes Duración"].apply(funcionTiempo)
info_df = pd.DataFrame({
    'Columna': datos.columns,
    'Tipo de dato': datos.dtypes,
    'Valores no nulos': datos.notnull().sum(),
    'Valores nulos': datos.isnull().sum(),
    'Porcentaje nulos (%)': round(datos.isnull().mean() * 100, 2)
})

print(info_df)
print(datos.head(10))
