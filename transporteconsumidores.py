

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/My Uber Drives - 2016.csv')

df

df.shape

df.columns.values

df.columns=['Fecha_incial','Fecha_final','Categoria','Inicio','FInal','MIllas','Objetivo']

df.columns

nulls=df.isnull().sum()

nulls

df.Purpose = df.replace(to_replace=np.nan,value='Missing_record')

df.Purpose.value_counts()

df.Purpose.isnull().sum()

df=df.dropna(axis=0)

df.head()

df.Objetivo.value_counts().to_frame()

df.Categoria.value_counts().to_frame()

df.columns.values

df['Fecha_incial']=pd.to_datetime(df['Fecha_incial'])

df.head()

df.groupby(['Inicio'])['Categoria'].count()

df.Categoria.value_counts().plot(kind='bar',figsize=(15,10))

"""La mayoría de viajes se usan para negocios"""

df.groupby('Objetivo').mean().plot(kind='bar',figsize=(15,10))

"""La mayoría de viajes se hacen por la razón de que los ususarios deben viajar diariamente."""



df.columns.values



plt.figure(figsize=(15,15))
pd.Series(df['Fecha_incial']).value_counts()[:25].plot(kind = "pie")
plt.title("travels")
plt.xticks(rotation = 50)

plt.figure(figsize=(15,15))
pd.Series(df['Fecha_final']).value_counts()[:25].plot(kind = "pie")
plt.title("travels")
plt.xticks(rotation = 50)

plt.figure(figsize=(15,15))
pd.Series(df['Categoria']).value_counts()[:25].plot(kind = "pie")
plt.title("travels")
plt.xticks(rotation = 50)

plt.figure(figsize=(15,15))
pd.Series(df['Inicio']).value_counts()[:25].plot(kind = "pie")
plt.title("travels")
plt.xticks(rotation = 50)

df.columns.values



plt.figure(figsize=(15,15))
pd.Series(df['FInal']).value_counts()[:25].plot(kind = "pie")
plt.title("travels")
plt.xticks(rotation = 50)

plt.figure(figsize=(15,15))
pd.Series(df['Objetivo']).value_counts()[:50].plot(kind = "pie")
plt.title("travels")
plt.xticks(rotation = 50)

