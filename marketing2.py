


import pandas as pd

dataset = pd.read_excel('/content/marketing_campaign.xlsx')

dataset

DF.shape

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df.columns = df.columns.str.replace(' ', '')

# transform Income column to a numerical
df['Income'] = df['Income'].str.replace('$', '')
df['Income'] = df['Income'].str.replace(',', '').astype('float')

dataset.head()

print(dataset['Marital_Status'].value_counts())

dataset['Marital_Status'] = dataset['Marital_Status'].replace(['Alone','YOLO','Absurd'],'Single')
dataset.Marital_Status.value_counts()

print(dataset['Education'].value_counts())

num_coln = dataset.select_dtypes(include=np.number).columns.tolist()
bins=10
j=1
fig = plt.figure(figsize = (20, 30))
for i in num_coln:
    plt.subplot(7,4,j)
    plt.boxplot(dataset[i])
    j=j+1
    plt.xlabel(i)
    plt.legend()
plt.show()

dataset.drop(dataset[(dataset['Income']>200000)|(dataset['Year_Birth']<1920)].index,inplace=True)

dataset.shape

dataset.rename(columns = {'Year_Birth':'Age'}, inplace = True)
dataset['Age'] = dataset.Age.apply(lambda x: 2021-x)

dataset['MntTotal'] = np.sum(dataset.filter(regex='Mnt'), axis=1)

dataset['TotalPurchases'] = np.sum(dataset.filter(regex='Purchases'),axis=1)

dataset['AvgWeb'] = round(dataset['NumWebPurchases']/dataset['NumWebVisitsMonth'],2)
dataset.fillna({'AvgWeb' : 0},inplace=True) # Handling for cases where division by 0 may yield unwanted results
dataset.replace(np.inf,0,inplace=True)

dataset['Children'] = dataset['Kidhome'] + dataset['Teenhome']

dataset['Dt_Customer'] = pd.to_datetime(dataset['Dt_Customer'] )

dataset.rename(columns = {'Dt_Customer':'TotalEnrollDays'}, inplace = True)

dataset['TotalEnrollDays'] = [int(str(dataset['TotalEnrollDays'][x])[:4]) for x in dataset.index]

dataset.head()

plt.figure(figsize=(15,20))
mask = np.triu(dataset.corr())
sns.heatmap(round(dataset.corr(),2),
            annot=True,
            vmin=-1,vmax=1,center=0,
            cmap='bwr',
            mask=mask,
            cbar_kws = {'orientation':'horizontal'}
           )

