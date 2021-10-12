

import pandas as pd

data = pd.read_csv('/content/onlinedeliverydata.csv')

data

#Pregunta 1: ¿Cuantos pedidos recibimos?

data.shape

#Respuesta 1: 388

data.dtypes

list(data.columns)

data1 = data.drop(['Marital Status'], axis=1)

data1 = data.drop(['Monthly Income','Educational Qualifications'], axis=1)

data1 = data.drop(['Marital Status','Family size',  'latitude', 'longitude',    'Pin code', 'Medium (P1)',  'Medium (P2)',  'Meal(P1)', 'Meal(P2)', 'Perference(P1)', 'Perference(P2)', 'Ease and convenient', 'Time saving',   'More restaurant choices',  'Easy Payment option',  'More Offers and Discount', 'Good Food quality',    'Good Tracking system', 'Self Cooking', 'Health Concern',   'Late Delivery',    'Poor Hygiene', 'Bad past experience',  'Unavailability',   'Unaffordable', 'Long delivery time',   'Delay of delivery person getting assigned',    'Delay of delivery person picking up food', 'Wrong order delivered', 'Missing item',    'Order placed by mistake',  'Influence of time',    'Order Time',   'Maximum wait time',    'Residence in busy location',   'Google Maps Accuracy', 'Good Road Condition',  'Low quantity low time',    'Delivery person ability',  'Influence of rating',  'Less Delivery time',   'High Quality of package',  'Number of calls',  'Politeness',   'Temperature',  'Good Quantity',    'Output',   'Reviews'], axis=1)

data1

data1 = data.drop(['Monthly Income','Educational Qualifications'], axis=1)

data5 = data1.to_excel('PedidosYa_new.xlsx')

data5

data7 = pd.read_excel('/content/PedidosYa_new.xlsx')

data7.drop(['Marital Status','Family size',  'latitude', 'longitude', 'Pin code', 'Medium (P1)',  'Medium (P2)',  'Meal(P1)', 'Meal(P2)', 'Perference(P1)', 'Perference(P2)', 'Ease and convenient', 'Time saving',   'More restaurant choices',  'Easy Payment option',  'More Offers and Discount', 'Good Food quality',    'Good Tracking system', 'Self Cooking', 'Health Concern',   'Late Delivery',    'Poor Hygiene', 'Bad past experience',  'Unavailability',   'Unaffordable', 'Long delivery time',   'Delay of delivery person getting assigned',    'Delay of delivery person picking up food', 'Wrong order delivered', 'Missing item',    'Order placed by mistake',  'Influence of time',    'Order Time',   'Maximum wait time',    'Residence in busy location',   'Google Maps Accuracy', 'Good Road Condition',  'Low quantity low time',    'Delivery person ability',  'Influence of rating',  'Less Delivery time',   'High Quality of package',  'Number of calls',  'Politeness',   'Temperature',  'Good Quantity',    'Output',   'Reviews'], axis=1)

data1 = data7

data7 = data1.to_excel('PEDIDOSYA.xlsx')

df = pd.read_excel('/content/PEDIDOSYA.xlsx')

df



data6 = pd.Series(22)

data6.value_counts()

#Pregunta 2: Qué rango de edad es el que más nos compra?

tab = pd.crosstab(index=data5["Age"],columns="frecuencia")
print(tab)

#Respuesta 2: Los que más compran tienen 23 años

#Pregunta: ¿Quien compra más, hombres o mujeres?

tab = pd.crosstab(index=data5["Gender"],columns="frecuencia")
print(tab)

#Respuesta 2: Hombres

#Pregunta 3: ¿Que ocupación tienen la mayoría de los que nos compran?

tab = pd.crosstab(index=df["Occupation"],columns="frecuencia")
print(tab)

tab = pd.crosstab(index=df["Age"],columns="frecuencia")
print(tab)

#Los que más nos compran son estudiantes

"""Correlación"""

#Pregunta 4: ¿Qué relación hay entre la ocupación y las compras?

df



import matplotlib.pyplot as plt



df.isnull().values.any()

df.isnull().sum()

df = pd.read_excel("/content/PEDIDOSYA.xlsx", na_values = ['NA', 'N/A','-'])

df

tab = pd.crosstab(index=df['Educational Qualifications'],columns="frecuencia")

print(tab)

import pandas as PD
import matplotlib.pyplot as plt
import seaborn as sns

df

plt.scatter(df['Age'], df['Educational Qualifications'])
plt.title('Correlación')
plt.xlabel('var1')
plt.ylabel('var2')
plt.show()

sns.pairplot(df)

df1 = df.rename(columns={'Unnamed:0' : 'Ventas'})

df1



df.corr('pearson')

plt.scatter(df['Unnamed: 0'], df['Age'])
plt.title('Correlación')
plt.xlabel('var1')
plt.ylabel('var2')
plt.show()

import pandas as pd

#Hay una correlación entre las ventas y la edad de los consumidores

df = pd.read_excel('/content/PEDIDOSYA.xlsx')

df

#Preparamos el modelado del modelo de ML para predecir y responder la siguiente pregunta:

# 1: ¿Qué edad tendra el siguiente comprador?

import numpy as np
import pandas as pd

datos = pd.read_excel('/content/PEDIDOSYA.xlsx')

datos.head(20)

df.Gender.replace(('Female','Male'), (0,1), inplace=True)

tab = pd.crosstab(index=datos["Occupation"],columns="frecuencia")
print(tab)

df.Occupation.replace(('Employee','House wife','Self Employeed','Student'), (0, 1, 2, 3), inplace=True)

datos.head(5)

tab = pd.crosstab(index=df["Age"],columns="frecuencia")
print(tab)

df

tab = pd.crosstab(index=df["Gender"],columns="frecuencia")
print(tab)

df.head(16)

import numpy as np 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


Ocupación [3,3,3,3,3,0,3,3,3,3,3,3,3,3,2,3] #A
Genero = [0,0,1,0,1,0,0,0,0,1,1,1,0,0,0,0]

x = Ocupación
y = Genero

x = mean_data = np.array(x)
X=x[:,np.newaxis]

while True:
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    mlr=MLPRegressor(solver="lbfgs",alpha=1e-5,hidden_layer_sizes=(3,3), random_state=1)
    mlr.fit(X_train, y_train)
    print(mlr.score(X_train, y_train))
    if mlr.score(X_train,y_train) > 0.10:
        break

print("Basándonos en consumidores anteriores, el siguiente cliente que tendra ' ' años: ")
print(mlr.predict(np.array(70).reshape(1, 1)))

#La edad del siguiente cliente sera 20 años
