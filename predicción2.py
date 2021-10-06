

import pandas as pd

datos = pd.read_excel('/content/MarathonData (1).xlsx')

datos

datos["Name"]

datos.info()

datos['Wall21'] = pd.to_numeric(datos['Wall21'],errors='coerce')

datos.describe()

datos

datos2 = datos.rename(columns = {'Marathon': 'Ciudad', 'km4week': 'Precio', 'sp4week': 'Rebaja', 'CrossTraining': 'NAN', 'Wall21': 'Devoluciones', 'MarathonTime': 'Reparaciones'}, inplace = False)
datos2 #Cambiar el nombre de las columnas

datos

datos2

datos2.hist()

datos2 = datos2.drop(columns=['Name'])
datos2 = datos2.drop(columns=['id'])
datos2 = datos2.drop(columns=['Ciudad'])
datos2 = datos2.drop(columns=['CATEGORY'])
datos2

datos2.isna().sum() #Faltan datos

datos2

datos2["NAN"] = datos2["NAN"].fillna(0)
datos2  #Se rellenan con ceros

datos2 = datos2.dropna(how='any')
datos2

datos3 = datos2.drop(['NAN'], axis=1)

datos3

datos3['Category'].unique()

valores_categoria = {"Category":  {'MAM':1, 'M45':2, 'M40':3, 'M50':4, 'M55':5,'WAM':6}}
datos3.replace(valores_categoria, inplace=True)
datos3

import matplotlib.pyplot as plt
plt.scatter(x = datos3['Precio'], y=datos3['Reparaciones'])
plt.title('Precio Vs Reparaciones Time')
plt.xlabel('Precio')
plt.ylabel('Reparaciones')
plt.show()

datos3 = datos3.query('Rebaja<1000')

plt.scatter(x = datos3['Rebaja'], y= datos3['Reparaciones'])
plt.title('Rebaja Vs Reparaciones')
plt.xlabel('Rebaja')
plt.ylabel('Reparaciones')
plt.show()

plt.scatter(x = datos3['Devoluciones'], y= datos3['Reparaciones'])
plt.title('Devoluciones Vs Reparaciones')
plt.xlabel('Devoluciones')
plt.ylabel('Reparaciones')
plt.show()

datos3

"""Entrenamiento del modelo de predicción

"""

datos_entrenamiento = datos3.sample(frac=0.8,random_state=0)
datos_test = datos3.drop(datos_entrenamiento.index)

datos_entrenamiento

datos_test

"""Marcamos el valor a predecir"""

etiquetas_entrenamiento = datos_entrenamiento.pop('Reparaciones')
etiquetas_test = datos_test.pop('Reparaciones')

etiquetas_entrenamiento

etiquetas_test

datos_entrenamiento

from sklearn.linear_model import LinearRegression
modelo = LinearRegression()
modelo.fit(datos_entrenamiento,etiquetas_entrenamiento)

predicción = modelo.predict(datos_test)
predicción

import numpy as np
from sklearn.metrics import mean_squared_error
error = np.sqrt(mean_squared_error(etiquetas_test, predicciones))
print("Error porcentual : %f" % (error*100))

datos3

nuevo_corredor = pd.DataFrame(np.array([[1,400,20,0,1.4]]),columns=['Category', 'Precio','Rebaja', 'Reparaciones','Devoluciones'])
nuevo_corredor

modelo.predict(nuevo_corredor)
