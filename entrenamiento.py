print("Hola, este script se ejecuta en Azure ML 🚀")
from math import sqrt
for i in range(3):
    print(f"Iteración {i}")
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-1,1,1000)
y=np.log(np.abs(x))
z=np.sqrt(np.abs(x)+1)
plt.plot(x,y,'r')
plt.plot(x,z,'g')
import pandas as pd
m={
    'x':x,
    'y':y,
    'z':z
}
dt=pd.DataFrame(m,columns=['x','y','z'])
print(dt.describe())
print(dt.hist())
print(dt.boxplot())
