

import pandas as pd

df = pd.read_excel('/content/marketing_campaign.xlsx')

df.head(5)

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
# %matplotlib inline
import seaborn as sns
import matplotlib
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df.head(10).style.set_properties(**{"background-color": "#fffc99","color": "black", "border-color": "black"})

"""# **CLEANING**"""

df = df.copy()
df.head(5).style.set_properties(**{"background-color": "#fffc99","color": "black", "border-color": "black"})

print("Shape of the DataFrame is :",df.shape)

print("Columns in DataFrame is :\n",df.columns)

print("Print a Summary of a Dataframe is :",df.info())

df.describe().style.set_properties(**{"background-color": "#fffc99","color": "black", "border-color": "black"})

def missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    Percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, Percentage], axis=1, keys=['Total', 'Percentage'])
missing_data(df).style.set_properties(**{"background-color": "#fffc99","color": "black", "border-color": "black"})

import missingno as mn
mn.matrix(df)

df['Income']=df['Income'].fillna(df['Income'].median())

df.isna().any()

df[df.duplicated()] #0

df=df.drop(columns=["Z_CostContact", "Z_Revenue"],axis=1)
df.head(5).style.set_properties(**{"background-color": "#fffc99","color": "black", "border-color": "black"})

"""# ***UNIVARIATE ANALYSIS***"""

print("Unique categories present in the Year_Birth:",df["Year_Birth"].value_counts())

df['Education'].unique()

print("Unique categories present in the Year_Birth:",df["Education"].value_counts())

df['Education'] = df['Education'].replace(['PhD','2n Cycle','Graduation', 'Master'],'Post Graduate')  
df['Education'] = df['Education'].replace(['Basic'], 'Under Graduate')

print("Unique categories present in the Education:",df["Education"].value_counts())
print('\n')


df['Education'].value_counts().plot(kind='bar',color = 'mediumblue',edgecolor = "black",linewidth = 3)
plt.title("Frequency Of Each Category in the Education Variable \n")

df['Marital_Status'].unique()

print("Unique categories present in the Year_Birth:",df["Marital_Status"].value_counts())

df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Relationship')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')

print("Unique categories present in the Marital_Status:",df['Marital_Status'].value_counts())
print("\n")

df['Marital_Status'].value_counts().plot(kind='bar',color = 'MediumBlue',edgecolor = "black",linewidth = 3)
plt.title("Frequency Of Each Category in the Marital_Status Variable \n")

df['Income'].max()

df['Income'].min()

df['Income'].mean()

sns.distplot(df["Income"],color = 'Mediumblue')
plt.show()
df["Income"].plot.box(figsize=(16,5),color = 'MediumBlue')
plt.show()

df['Kids'] = df['Kidhome'] + df['Teenhome']

print("Unique categories present in the Kids:",df['Kids'].value_counts())
print("\n")

#VISUALIZING THE "Kids"
df['Kids'].value_counts().plot(kind='bar',color = 'mediumblue',edgecolor = "black",linewidth = 3)
plt.title("Frequency Of Each Category in the Kids Variable \n")

df['Expenses'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['Expenses'].head(10)

sns.distplot(df["Expenses"],color = 'mediumblue')
plt.show()
df["Expenses"].plot.box(figsize=(16,5))
plt.show()

df['TotalAcceptedCmp'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5']

print("Unique categories present in the TotalAcceptedCmp:",df['TotalAcceptedCmp'].value_counts())
print("\n")


df['TotalAcceptedCmp'].value_counts().plot(kind='bar',color = 'mediumblue',edgecolor = "black",linewidth = 3)
plt.title("Frequency Of Each Category in the TotalAcceptedCmp Variable \n")

df['NumTotalPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']
df['NumTotalPurchases'].unique()

sns.distplot(df["NumTotalPurchases"],color = 'mediumblue')
plt.show()
df["NumTotalPurchases"].plot.box(figsize=(16,5),color = 'mediumblue')
plt.show()

# Borrar columnas para reducir la complejidad del modelo

col_del = ["ID","AcceptedCmp1" , "AcceptedCmp2", "AcceptedCmp3" , "AcceptedCmp4","AcceptedCmp5","NumWebVisitsMonth", "NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases" , "Kidhome", "Teenhome","MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df=df.drop(columns=col_del,axis=1)
df.head(10).style.set_properties(**{"background-color": "#fffc99","color": "black", "border-color": "black"})

x = df.columns 
for i in x:
     print(i)

df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['First_day'] = '01-01-2015'
df['First_day'] = pd.to_datetime(df.First_day)
df['Day_engaged'] = (df['First_day'] - df['Dt_Customer']).dt.days

.df['Age'] = (pd.Timestamp('now').year) - (pd.to_datetime(df['Dt_Customer']).dt.year)

print("Unique categories present in the Age:",df['Age'].value_counts())
print("\n")

df['Age'].value_counts().plot(kind='bar',color = 'mediumblue',edgecolor = "black",linewidth = 3)
plt.title("Frequency Of Each Category in the Age Variable \n")

#El 24,86% son clientes de 7 años. El 53,08% son clientes de 8 años. El 22,05% son clientes de 9 años

df.head(5).style.set_properties(**{"background-color": "#fffc99","color": "black", "border-color": "black"})

df=df.drop(columns=["Dt_Customer", "First_day", "Year_Birth", "Dt_Customer", "Recency", "Complain","Response"],axis=1)
df.head(5).style.set_properties(**{"background-color": "#fffc99","color": "black", "border-color": "black"})

df.shape

import pandas

import pandas as pd

pandas.__version__

# Commented out IPython magic to ensure Python compatibility.
!pip uninstall pandas
!pip install "pandas==0.25.3"
# %pylab inline

pandas.__version__

from pandas_profiling import ProfileReport
prof = ProfileReport(df)

prof

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.ticker as ticker

def barplot_data(listx,listy,sizex=18,sizey=6,stepx=6,stepy=6,numberx=True,ox=True):   
    fig, ax = plt.subplots()
    fig.set_figwidth(sizex)
    fig.set_figheight(sizey)
    b = ax.bar(listx, listy, color='orange')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(round(sizey/stepy)))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(round(sizex/stepx)))
    if (numberx):
        for rect in b:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

    plt.show()

barplot_data(list(df.Year_Birth.unique()),list(df.groupby('Year_Birth').count()['ID']),stepy=1)

plt.show()

df = df[df['Year_Birth'] > 1940]
df

barplot_data(list(df.Year_Birth.unique()),list(df.groupby('Year_Birth').count()['ID']),stepy=1)

