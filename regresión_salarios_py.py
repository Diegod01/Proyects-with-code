

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('/content/salarios.csv')

dataset.head()

X = dataset.iloc[:,:-1].values
y =  dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regresor = LinearRegression()

regresor.fit(X_train, Y_train)

y_pred = regresor.predict(X_test)

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regresor.predict(X_train), color='blue')
plt.title['Salario VS Experiencia']
plt.xlabel('Años de experiencia')
plt.ylabel('Salario')
plt.show()

"""A más años más salario"""





