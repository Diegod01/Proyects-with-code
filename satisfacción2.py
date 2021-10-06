

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data2 = pd.read_excel('Satisfacción2.xlsx')
Data2.head(21)

Data2['Experiencias'].unique()

Data2['Experiencias'].replace(('Excelente','Muy Bueno', 'Bueno', 'Promedio', 'Malo', 'Muy Malo', 'Horrible'), (7,6,5,4,3,2,1), inplace = True)   #Convertir datos a numerico

Data2['Experiencias'].unique()

Data2['Experiencias'].head()

Data2.describe()

Opiniones_de_clientes = [7,6,5,4,3,2,1]
Nombres_de_las_opiniones = ['Excelente','Muy Bueno', 'Bueno', 'Promedio', 'Malo', 'Muy Malo', 'Horrible']

plt.pie(Opiniones_de_clientes, labels = Nombres_de_las_opiniones, autopct="%0.1f %%" )
plt.show()

