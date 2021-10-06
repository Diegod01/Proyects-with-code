
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data = pd.read_excel('Satisfacción.xlsx')
Data.head(11)

Data.describe()

Data['Experiencia'].unique()

Data['Experiencia'].replace(('Excelente','Promedio', 'Horrible'), (3,2,1), inplace = True)   #Convertir datos a numerico

Data['Experiencia'].unique()

Data['Experiencia'].head()

Data.describe()

Opiniones_de_clientes = [3,2,1]
Nombres_de_las_opiniones = ['Excelente','Promedio', 'Horrible']

plt.pie(Opiniones_de_clientes, labels = Nombres_de_las_opiniones, autopct="%0.1f %%" )
plt.show()

"""50% Creé que el servicio es Excelente (Satisfecho)
33.3% Creé que el el servicio es Promedio (Satisfacción Promedio)
16.7% Creé que el el servicio es Horrible (No satisfecho)
"""
